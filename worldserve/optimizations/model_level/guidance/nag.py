"""
Normalized Attention Guidance (NAG) for WorldServe / Oasis.

Reference: arxiv 2505.21179  "Normalized Attention Guidance: Enhancing
Classifier-Free Guidance for Diffusion Models without Introducing Noise"
(Chen et al., 2025).

Motivation
----------
In few-step distilled models (4-step Oasis, 3-step MG2) the per-step ODE
integration interval ``h`` is large, so guidance corrections applied at the
*output* level can push latents far outside the training distribution.  NAG
moves guidance into the *attention* layer: it extrapolates attention maps
beyond the conditional and then projects them back onto the probability
simplex (L1-normalise over keys).  This keeps guidance within valid attention
distributions regardless of the guidance scale.

Per-layer formula (applied after softmax inside each attention module):

    attn_guided = attn_cond + eta * (attn_cond - attn_uncond)
    attn_guided = attn_guided / attn_guided.sum(dim=-1, keepdim=True)

where ``eta`` is the guidance coefficient (analogous to CFG weight minus 1).

Implementation notes
--------------------
- Forward hooks are installed on all ``nn.Module`` instances whose name
  contains 'attn', 'attention', or 'self_attn'.
- The hook intercepts the *output* of each attention module and expects the
  first return value to be the attention-weighted value tensor (shape
  (B, heads, Q, C/heads) or (B, Q, C)).  For more complex attention modules
  that return tuples the hook unpacks and repacks transparently.
- Because NAG requires paired cond/uncond attention outputs the wrapper
  batches them: the first half of the batch is treated as cond, the second
  half as uncond, matching standard CFG batch doubling.
- ``@torch._dynamo.disable`` is applied to hook callables so that
  torch.compile / CUDA-graph capture does not trace through them (the dynamic
  normalisation branch upsets static-graph assumptions).
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Thread-local flag for APGWrapper sequential-mode signalling
# ---------------------------------------------------------------------------

_nag_thread_local = threading.local()


def set_nag_sequential_mode(enabled: bool) -> None:
    """Called by APGWrapper to indicate sequential (not batch-doubled) CFG."""
    _nag_thread_local.sequential_mode = enabled


def is_nag_sequential_mode() -> bool:
    return getattr(_nag_thread_local, "sequential_mode", False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_attention_module(name: str, module: nn.Module) -> bool:
    """Return True if *name* suggests an attention sub-module."""
    lower = name.lower()
    return any(k in lower for k in ("attn", "attention", "self_attn"))


# ---------------------------------------------------------------------------
# Hook implementation (decorated to opt out of torch.compile tracing)
# ---------------------------------------------------------------------------

@torch._dynamo.disable
def _nag_hook_fn(
    module: nn.Module,
    input: Tuple[Any, ...],
    output: Any,
    *,
    eta: float,
    enabled_flag: List[bool],
) -> Any:
    """
    Forward hook that applies NAG normalisation to the attention output.

    The hook is registered on individual attention modules and fires after
    each forward pass.  It expects the module to output either:

      - A plain Tensor (attention output), shape (..., Q, D).
      - A tuple whose first element is such a Tensor.

    When the batch is doubled (first half = cond, second half = uncond) the
    hook applies:
        out_cond  += eta * (out_cond - out_uncond)
        out_cond   = L1-normalise over last dim

    If the batch is not doubled (single-pass inference) the hook is a no-op.

    Parameters
    ----------
    module : nn.Module
        The attention module (unused beyond identification).
    input : tuple
        Inputs to the module (unused).
    output : Tensor or tuple
        Module output to be modified in-place conceptually (we return a new
        tensor / tuple).
    eta : float
        NAG guidance coefficient.
    enabled_flag : list[bool]
        Single-element mutable list used as an "enabled" flag so the closure
        can be toggled without re-registering the hook.
    """
    if not enabled_flag[0]:
        return output  # pass through when disabled

    # Unpack tuple outputs.
    is_tuple = isinstance(output, tuple)
    if is_tuple:
        attn_out = output[0]
        rest = output[1:]
    else:
        attn_out = output
        rest = None

    if not isinstance(attn_out, Tensor):
        return output  # unknown output format — skip

    # CFG detection: either batch is doubled (standard CFG) or
    # APGWrapper signals sequential mode via thread-local flag.
    if not is_nag_sequential_mode() and attn_out.shape[0] % 2 != 0:
        return output

    if is_nag_sequential_mode():
        # Sequential mode: APGWrapper runs cond/uncond separately.
        # In this case NAG normalizes the full output (not split).
        # The NAG normalization still improves distribution but
        # can't do the cond-uncond extrapolation.
        normed = attn_out / (attn_out.sum(dim=-1, keepdim=True).abs() + 1e-8)
        if is_tuple:
            return (normed,) + rest
        return normed

    batch = attn_out.shape[0]
    half = batch // 2
    out_cond = attn_out[:half]
    out_uncond = attn_out[half:]

    # NAG extrapolation then L1 normalisation over the last dimension.
    out_guided = out_cond + eta * (out_cond - out_uncond)
    # Clamp negatives before normalising to avoid degenerate distributions.
    out_guided = out_guided.clamp(min=0.0)
    denom = out_guided.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out_guided = out_guided / denom

    # Reconstruct: replace cond half with guided output, keep uncond intact.
    new_attn = torch.cat([out_guided, out_uncond], dim=0)

    if is_tuple:
        return (new_attn,) + rest
    return new_attn


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class NAGHook:
    """
    Normalized Attention Guidance hook manager.

    Installs forward hooks on all attention modules in *model*, applies NAG
    normalisation during inference, and provides enable/disable/remove controls.

    Parameters
    ----------
    model : nn.Module
        The target model (e.g. Oasis DiT).  Hook installation happens at
        construction time.
    eta : float
        Guidance coefficient in attention space.  ``eta=1.5`` means 50 %
        extrapolation beyond the conditional attention.  Typical range 1–3.
    target_modules : list[str] or None
        If given, only install hooks on modules whose *full qualified name*
        is in this list.  If None, ``find_attention_modules`` is used to
        automatically discover all attention sub-modules.
    """

    def __init__(
        self,
        model: nn.Module,
        eta: float = 1.5,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.eta = eta

        # Shared mutable flag — toggled by enable() / disable().
        self._enabled: List[bool] = [True]

        # Registry: {full_module_name: hook_handle}
        self._hooks: Dict[str, Any] = {}

        # Discover and install hooks.
        modules_to_hook = self._resolve_target_modules(target_modules)
        self._install_hooks(modules_to_hook)

    # ------------------------------------------------------------------
    # Module discovery
    # ------------------------------------------------------------------

    @staticmethod
    def find_attention_modules(model: nn.Module) -> Dict[str, nn.Module]:
        """
        Walk *model* and return ``{name: module}`` for all sub-modules
        whose qualified name suggests an attention layer.

        Matches names containing 'attn', 'attention', or 'self_attn'
        (case-insensitive).

        Parameters
        ----------
        model : nn.Module
            Model to inspect.

        Returns
        -------
        dict[str, nn.Module]
            Ordered dictionary of matching modules.
        """
        found: Dict[str, nn.Module] = {}
        for name, module in model.named_modules():
            if name and _is_attention_module(name, module):
                found[name] = module
        return found

    def _resolve_target_modules(
        self, target_modules: Optional[List[str]]
    ) -> Dict[str, nn.Module]:
        """Return the modules to hook based on ``target_modules`` arg."""
        if target_modules is None:
            discovered = self.find_attention_modules(self.model)
            if not discovered:
                warnings.warn(
                    "NAGHook: no attention modules found in model.  "
                    "NAG will have no effect.  Pass explicit target_modules "
                    "if your attention modules use non-standard names.",
                    stacklevel=3,
                )
            return discovered

        # Resolve by name.
        all_named: Dict[str, nn.Module] = dict(self.model.named_modules())
        resolved: Dict[str, nn.Module] = {}
        for name in target_modules:
            if name in all_named:
                resolved[name] = all_named[name]
            else:
                warnings.warn(
                    f"NAGHook: requested module '{name}' not found in model; skipping.",
                    stacklevel=3,
                )
        return resolved

    # ------------------------------------------------------------------
    # Hook installation / removal
    # ------------------------------------------------------------------

    def _install_hooks(self, modules: Dict[str, nn.Module]) -> None:
        """Register forward hooks on each module in *modules*."""
        for name, module in modules.items():
            handle = module.register_forward_hook(
                lambda mod, inp, out, _name=name: _nag_hook_fn(
                    mod,
                    inp,
                    out,
                    eta=self.eta,
                    enabled_flag=self._enabled,
                )
            )
            self._hooks[name] = handle

    def remove_hooks(self) -> None:
        """Remove all registered hooks.  After this call the object is inert."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Enable / disable toggle
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Re-enable NAG normalisation (hooks remain installed)."""
        self._enabled[0] = True

    def disable(self) -> None:
        """
        Temporarily disable NAG normalisation without removing hooks.
        Hooks remain installed but become pass-through until ``enable()``
        is called.
        """
        self._enabled[0] = False

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NAGHook(eta={self.eta}, "
            f"hooks={list(self._hooks.keys())}, "
            f"enabled={self._enabled[0]})"
        )

    def __del__(self) -> None:
        """Best-effort cleanup when the object is garbage-collected."""
        try:
            self.remove_hooks()
        except Exception:
            pass
