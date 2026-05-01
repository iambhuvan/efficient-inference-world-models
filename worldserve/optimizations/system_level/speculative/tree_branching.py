"""
Tree-based action branching for WorldServe (Phase 5 — NOVEL CONTRIBUTION).

Predicts K likely next player actions and pre-computes draft frames for each
in parallel. When the actual input arrives, if it matches one of the predicted
actions, the pre-computed frame is served instantly (no forward pass needed).
Otherwise, falls back to a standard forward pass.

Key insight: GPU parallelism means K=4 batched forward passes cost ~2.5-3x a
single forward, not 4x. Combined with ~60% action repeat rate in Minecraft,
this yields significant latency reduction.

Action prediction strategies:
  - 'repeat_last': Predict last action + top-K most common from history (~60% hit rate)
  - 'mlp': Learned MLP on last N actions for distribution prediction

Architecture assumptions (Matrix-Game 2.0):
  - 30 DiT blocks, Wan2.1 backbone
  - Keyboard cross-attention + mouse concatenation
  - Rolling KV window of 6 frames
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HW5-derived worker primitives for true multi-GPU parallel branch execution
# ---------------------------------------------------------------------------

class _Task:
    """Wraps a compute function preserving grad-mode state (from HW5/worker.py)."""
    def __init__(self, compute: Callable) -> None:
        self._compute = compute
        self._grad_enabled = torch.is_grad_enabled()

    def compute(self) -> Any:
        with torch.set_grad_enabled(self._grad_enabled):
            return self._compute()


def _gpu_worker(in_queue: Queue, out_queue: Queue, device: torch.device) -> None:
    """Per-device worker thread loop (from HW5/worker.py)."""
    ctx = torch.cuda.device(device) if device.type == "cuda" else _nullctx()
    with ctx:
        while True:
            task = in_queue.get()
            if task is None:
                break
            try:
                result = task.compute()
            except Exception:
                out_queue.put((False, sys.exc_info()))
                continue
            out_queue.put((True, result))
    out_queue.put((False, None))


def _create_gpu_workers(
    devices: List[torch.device],
) -> Tuple[List[Queue], List[Queue]]:
    """Spawn one daemon thread per unique device (from HW5/worker.py)."""
    in_queues: List[Queue] = []
    out_queues: List[Queue] = []
    spawned: Dict[torch.device, Tuple[Queue, Queue]] = {}

    for device in devices:
        # Normalise: cuda with no index → current device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        if device not in spawned:
            iq: Queue = Queue()
            oq: Queue = Queue()
            spawned[device] = (iq, oq)
            t = Thread(target=_gpu_worker, args=(iq, oq, device), daemon=True)
            t.start()

        iq, oq = spawned[device]
        in_queues.append(iq)
        out_queues.append(oq)

    return in_queues, out_queues


class _nullctx:
    """Minimal no-op context manager for CPU devices."""
    def __enter__(self) -> "_nullctx":
        return self
    def __exit__(self, *_: Any) -> None:
        pass


@dataclass
class TreeBranchStats:
    """Tracks hit rate and latency for tree-based action branching."""
    total_frames: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_latencies_ms: List[float] = field(default_factory=list)
    miss_latencies_ms: List[float] = field(default_factory=list)
    batch_draft_times_ms: List[float] = field(default_factory=list)
    sequential_estimate_ms: List[float] = field(default_factory=list)


class ActionMLP(nn.Module):
    """
    Learned action predictor: MLP that takes last N actions and predicts
    a probability distribution over the next action.
    """

    def __init__(
        self,
        action_dim: int,
        history_len: int = 10,
        hidden_dim: int = 128,
        num_actions: int = 256,
    ) -> None:
        """
        Args:
            action_dim: Dimensionality of each action vector.
            history_len: Number of past actions to condition on.
            hidden_dim: Hidden layer size.
            num_actions: Size of discrete action space (for classification head).
        """
        super().__init__()
        self.action_dim = action_dim
        self.history_len = history_len
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(action_dim * history_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        Predict action distribution from history.

        Args:
            action_history: (B, history_len, action_dim) tensor of recent actions.

        Returns:
            (B, num_actions) log-probability distribution over next action.
        """
        B = action_history.shape[0]
        flat = action_history.reshape(B, -1)
        logits = self.net(flat)
        return F.log_softmax(logits, dim=-1)


class TreeActionBrancher:
    """
    Tree-based action branching for speculative frame generation.

    Pre-computes K likely action outcomes in parallel, then serves the
    matching frame instantly when the actual player input arrives.
    """

    def __init__(
        self,
        k_actions: int = 4,
        action_predictor: str = "repeat_last",
        action_mlp: Optional[ActionMLP] = None,
        action_dim: int = 0,
        num_discrete_actions: int = 256,
    ) -> None:
        """
        Args:
            k_actions: Number of action hypotheses to draft in parallel.
            action_predictor: Strategy for predicting likely actions.
                'repeat_last': Use last action + top-K from history (~60% Minecraft hit rate).
                'mlp': Use learned MLP predictor.
            action_mlp: Pre-trained ActionMLP instance (required if action_predictor='mlp').
            action_dim: Dimensionality of action vectors (for MLP predictor).
            num_discrete_actions: Size of discrete action space.
        """
        self.k_actions = k_actions
        self.action_predictor = action_predictor
        self.action_mlp = action_mlp
        self.action_dim = action_dim
        self.num_discrete_actions = num_discrete_actions
        self.stats = TreeBranchStats()
        self._single_fwd_ms: float = 0.0  # measured on first frame for honest speedup

        if action_predictor == "mlp" and action_mlp is None:
            logger.warning(
                "action_predictor='mlp' but no action_mlp provided. "
                "Falling back to 'repeat_last'."
            )
            self.action_predictor = "repeat_last"

    # ------------------------------------------------------------------
    # Action prediction
    # ------------------------------------------------------------------

    def predict_likely_actions(
        self,
        action_history: List[Any],
        k: Optional[int] = None,
    ) -> List[Any]:
        """
        Predict K likely next actions based on history.

        Strategies:
          - repeat_last: Returns last action + top-(K-1) most common actions from history.
            This exploits the ~60% action repeat rate in Minecraft.
          - mlp: Uses learned MLP to predict action distribution, returns top-K.

        Args:
            action_history: List of recent actions (tensors, ints, or dicts).
            k: Number of predictions (defaults to self.k_actions).

        Returns:
            List of K predicted actions (same type as history elements).
        """
        k = k or self.k_actions

        if not action_history:
            logger.warning("Empty action history; returning empty predictions.")
            return []

        if self.action_predictor == "repeat_last":
            return self._predict_repeat_last(action_history, k)
        elif self.action_predictor == "mlp":
            return self._predict_mlp(action_history, k)
        else:
            raise ValueError(f"Unknown action_predictor: {self.action_predictor!r}")

    def _predict_repeat_last(
        self,
        action_history: List[Any],
        k: int,
    ) -> List[Any]:
        """
        Predict using repeat-last + frequency heuristic.

        Returns: [last_action, 2nd_most_common, 3rd_most_common, ...] up to K.
        """
        last_action = action_history[-1]
        predictions = [last_action]

        if k <= 1:
            return predictions

        # Count action frequencies for non-tensor actions
        if isinstance(last_action, (int, str)):
            counter = Counter(action_history)
            # Get top actions, excluding the last one (already included)
            for action, _count in counter.most_common(k + 1):
                if len(predictions) >= k:
                    break
                if action != last_action:
                    predictions.append(action)

        elif isinstance(last_action, torch.Tensor):
            # For tensor actions: find unique actions by rounding/hashing
            seen_hashes = {_tensor_hash(last_action)}

            # Sort history by frequency using approximate hashing
            hash_counter: Counter = Counter()
            hash_to_action: Dict[int, Any] = {}
            for act in action_history:
                h = _tensor_hash(act)
                hash_counter[h] += 1
                hash_to_action[h] = act

            for h, _count in hash_counter.most_common(k + 1):
                if len(predictions) >= k:
                    break
                if h not in seen_hashes:
                    predictions.append(hash_to_action[h])
                    seen_hashes.add(h)

        elif isinstance(last_action, dict):
            # For dict actions: use string representation for dedup
            seen = {str(last_action)}
            counter = Counter(str(a) for a in action_history)
            action_by_str = {str(a): a for a in action_history}

            for action_str, _count in counter.most_common(k + 1):
                if len(predictions) >= k:
                    break
                if action_str not in seen:
                    predictions.append(action_by_str[action_str])
                    seen.add(action_str)

        # Pad with last action if not enough unique actions
        while len(predictions) < k:
            predictions.append(last_action)

        return predictions[:k]

    def _predict_mlp(
        self,
        action_history: List[Any],
        k: int,
    ) -> List[Any]:
        """Predict using learned MLP on action history."""
        assert self.action_mlp is not None

        # Prepare input tensor
        history_len = self.action_mlp.history_len
        action_dim = self.action_mlp.action_dim

        # Pad or truncate history
        if len(action_history) >= history_len:
            recent = action_history[-history_len:]
        else:
            # Pad with zeros
            pad_count = history_len - len(action_history)
            if isinstance(action_history[0], torch.Tensor):
                zero = torch.zeros_like(action_history[0])
            else:
                zero = 0
            recent = [zero] * pad_count + action_history

        # Convert to tensor
        if isinstance(recent[0], torch.Tensor):
            history_tensor = torch.stack(recent).unsqueeze(0)  # (1, history_len, action_dim)
        else:
            history_tensor = torch.tensor(recent, dtype=torch.float32).unsqueeze(0)
            if history_tensor.dim() == 2:
                history_tensor = history_tensor.unsqueeze(-1)  # (1, history_len, 1)

        device = next(self.action_mlp.parameters()).device
        history_tensor = history_tensor.to(device)

        # Predict
        self.action_mlp.eval()
        with torch.no_grad():
            log_probs = self.action_mlp(history_tensor)  # (1, num_actions)
            top_k = torch.topk(log_probs[0], k=k)

        # Convert back to action format
        predictions = []
        for idx in top_k.indices.tolist():
            if isinstance(action_history[0], torch.Tensor):
                # Create action tensor from index (model-specific encoding)
                act = torch.tensor([idx], dtype=action_history[0].dtype, device=action_history[0].device)
                predictions.append(act)
            else:
                predictions.append(idx)

        return predictions

    # ------------------------------------------------------------------
    # Batched draft forward
    # ------------------------------------------------------------------

    def batch_draft(
        self,
        model: nn.Module,
        x: torch.Tensor,
        likely_actions: List[Any],
        kv_cache: Optional[Any] = None,
        kv_cache_manager: Optional[Any] = None,
        timestep: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> "Tuple[List[torch.Tensor], Optional[Any]]":
        """
        Run batched forward pass for K action hypotheses in parallel.

        Repeats the input K times and batches all action hypotheses into a single
        forward pass. GPU parallelism means K=4 costs ~2.5-3x a single forward,
        not 4x (due to CUDA kernel launch overhead amortization).

        Args:
            model: The DiT model.
            x: (B, C, T, H, W) or (B, S, D) input latent tensor (single-batch).
            likely_actions: K predicted actions to draft.
            kv_cache: Optional KV cache to use during forward.
            kv_cache_manager: Optional KV cache manager for snapshot/restore.
            timestep: Current diffusion timestep tensor.
            **kwargs: Additional model arguments.

        Returns:
            Tuple of (List of K output tensors, cache_snapshot or None).
        """
        K = len(likely_actions)
        if K == 0:
            return [], None

        # Snapshot KV cache state before batched forward so we can rollback
        # non-selected branches on cache miss.
        cache_snapshot = None
        if kv_cache_manager is not None and hasattr(kv_cache_manager, "snapshot"):
            cache_snapshot = kv_cache_manager.snapshot()

        # Repeat input for K actions
        x_repeated = x.repeat(K, *([1] * (x.dim() - 1)))  # (K*B, ...)

        # Prepare batched action conditioning
        batched_actions = self._batch_actions(likely_actions, batch_size=x.shape[0])

        # Prepare timestep
        if timestep is not None:
            timestep_repeated = timestep.repeat(K, *([1] * (timestep.dim() - 1)))
        else:
            timestep_repeated = None

        # Handle KV cache repetition
        if kv_cache is not None and hasattr(kv_cache, "repeat_for_batch"):
            kv_cache_repeated = kv_cache.repeat_for_batch(K)
        else:
            kv_cache_repeated = kv_cache

        # Forward pass — use autocast so DiT's sinusoidal t_embedder (float32
        # internal) doesn't clash with float16 weight matrices.
        model.eval()
        _amp_dtype = x_repeated.dtype if x_repeated.dtype in (torch.float16, torch.bfloat16) else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=_amp_dtype):
            model_kwargs = dict(kwargs)
            if batched_actions is not None:
                model_kwargs["actions"] = batched_actions
            if kv_cache_repeated is not None:
                model_kwargs["kv_cache"] = kv_cache_repeated
            if timestep_repeated is not None:
                model_kwargs["timestep"] = timestep_repeated

            if isinstance(x_repeated, dict):
                output = model(**x_repeated, **model_kwargs)
            else:
                try:
                    output = model(x_repeated, **model_kwargs)
                except TypeError:
                    # Oasis DiT interface: model(x, t_BT, external_cond=actions)
                    # Remap 'actions'→'external_cond'; pass 'timestep' positionally.
                    fallback_kwargs: Dict[str, Any] = {
                        k: v for k, v in model_kwargs.items()
                        if k not in ("actions", "timestep", "kv_cache")
                    }
                    action_val = model_kwargs.get("actions")
                    # Only map to external_cond if it is a proper feature tensor (dim>=2),
                    # not a 1-D index tensor from discrete action space.
                    if isinstance(action_val, torch.Tensor) and action_val.dim() >= 2:
                        fallback_kwargs["external_cond"] = action_val
                    t_arg = model_kwargs.get("timestep")
                    if t_arg is not None and isinstance(t_arg, torch.Tensor):
                        KB = x_repeated.shape[0]
                        T  = x_repeated.shape[1] if x_repeated.dim() >= 2 else 1
                        if t_arg.dim() == 1 and t_arg.shape[0] == KB:
                            t_arg = t_arg.unsqueeze(1).expand(KB, T).to(x_repeated.dtype).contiguous()
                        try:
                            output = model(x_repeated, t_arg, **fallback_kwargs)
                        except TypeError:
                            output = model(x_repeated, **fallback_kwargs)
                    else:
                        output = model(x_repeated, **fallback_kwargs)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Split output back into K results
        B = x.shape[0]
        draft_frames = []
        for i in range(K):
            draft_frames.append(output[i * B : (i + 1) * B])

        return draft_frames, cache_snapshot

    def parallel_branch_draft(
        self,
        model: nn.Module,
        x: torch.Tensor,
        likely_actions: List[Any],
        timestep: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Run K action branches using the HW5-derived worker pattern.

        Multi-GPU (K GPUs available): dispatches each branch to a dedicated
        GPU thread via _Task + _create_gpu_workers, achieving true parallelism.
        The diagonal scheduling insight from HW5 _clock_cycles() applies here:
        all K tasks are submitted simultaneously so they overlap on separate devices.

        Single-GPU fallback: runs branches sequentially — no fake speedup claimed.

        Args:
            model: The DiT model.
            x: Input latent (single-batch).
            likely_actions: K predicted actions.
            timestep: Current diffusion timestep tensor.
            **kwargs: Extra model args.

        Returns:
            (draft_frames, elapsed_ms): List of K output tensors and wall time.
        """
        K = len(likely_actions)
        if K == 0:
            return [], 0.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if num_gpus >= K:
            # ── TRUE PARALLEL (multi-GPU): HW5 worker pattern ──────────────
            devices = [torch.device("cuda", i % num_gpus) for i in range(K)]
            in_queues, out_queues = _create_gpu_workers(devices)

            # Copy the model to each branch device so all parameters live on
            # the same device as the input tensors.  cuda:0 uses the original
            # to avoid doubling VRAM on that card; other devices get deep copies.
            import copy
            model_per_device: Dict[torch.device, nn.Module] = {}
            primary_device = next(model.parameters()).device
            for dev in devices:
                if dev == primary_device and dev not in model_per_device:
                    model_per_device[dev] = model
                elif dev not in model_per_device:
                    m_copy = copy.deepcopy(model).to(dev)
                    m_copy.eval()
                    model_per_device[dev] = m_copy

            # Submit K tasks simultaneously (mirrors HW5 pipe.compute schedule)
            for action, iq, device in zip(likely_actions, in_queues, devices):
                x_dev = x.to(device)
                t_dev = timestep.to(device) if isinstance(timestep, torch.Tensor) else timestep
                act_dev = action.to(device) if isinstance(action, torch.Tensor) else action
                kw_copy = dict(kwargs)
                m_dev = model_per_device[device]

                def _branch(_x=x_dev, _t=t_dev, _a=act_dev, _kw=kw_copy, _m=m_dev):
                    return self._safe_forward(_m, _x, _t, _a, **_kw)

                iq.put(_Task(_branch))

            # Collect results (preserving order)
            draft_frames: List[torch.Tensor] = [None] * K  # type: ignore[list-item]
            for i, oq in enumerate(out_queues):
                ok, payload = oq.get()
                if not ok:
                    exc_info = payload
                    raise RuntimeError(f"Branch {i} failed") from exc_info[1].with_traceback(exc_info[2])
                # Move back to primary device
                result = payload
                if isinstance(result, torch.Tensor):
                    result = result.to(x.device)
                draft_frames[i] = result

            # Shut down workers
            for iq in in_queues:
                iq.put(None)

        else:
            # ── SEQUENTIAL (single-GPU): honest — no parallelism ────────────
            draft_frames = []
            for action in likely_actions:
                out = self._safe_forward(model, x, timestep, action, **kwargs)
                draft_frames.append(out)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return draft_frames, elapsed_ms

    def _batch_actions(
        self,
        actions: List[Any],
        batch_size: int,
    ) -> Optional[Any]:
        """
        Combine K actions into a batched tensor for parallel forward.

        Args:
            actions: List of K actions.
            batch_size: Original batch size B.

        Returns:
            Batched action tensor of shape (K*B, ...), or None if actions are not tensorizable.
        """
        if not actions:
            return None

        if isinstance(actions[0], torch.Tensor):
            # Each action: (B, ...) or scalar
            expanded = []
            for act in actions:
                if act.dim() == 0:
                    act = act.unsqueeze(0).expand(batch_size)
                elif act.shape[0] != batch_size:
                    act = act.unsqueeze(0).expand(batch_size, *act.shape)
                expanded.append(act)
            return torch.cat(expanded, dim=0)

        elif isinstance(actions[0], (int, float)):
            t = torch.tensor(actions, dtype=torch.long)
            return t.repeat_interleave(batch_size)

        elif isinstance(actions[0], dict):
            # Batch dict actions: concatenate each key
            batched_dict = {}
            for key in actions[0]:
                vals = []
                for act in actions:
                    v = act[key]
                    if isinstance(v, torch.Tensor):
                        if v.shape[0] != batch_size:
                            v = v.unsqueeze(0).expand(batch_size, *v.shape)
                        vals.append(v)
                if vals:
                    batched_dict[key] = torch.cat(vals, dim=0)
            return batched_dict if batched_dict else None

        else:
            logger.warning("Cannot batch actions of type %s", type(actions[0]))
            return None

    # ------------------------------------------------------------------
    # Selection and verification
    # ------------------------------------------------------------------

    def select_and_verify(
        self,
        draft_frames: List[torch.Tensor],
        actual_action: Any,
        likely_actions: List[Any],
        model: Optional[nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Check if actual action matches a predicted action. If so, serve instantly.

        Args:
            draft_frames: List of K pre-computed output frames.
            actual_action: The actual player action that arrived.
            likely_actions: The K predicted actions that were drafted.
            model: DiT model (needed if cache miss requires full forward).
            x: Input tensor (needed for cache miss).
            timestep: Timestep tensor (needed for cache miss).
            **kwargs: Additional model arguments.

        Returns:
            Tuple of:
              - output: The output frame tensor.
              - was_hit: True if the actual action matched a prediction (instant).
        """
        # Check for match
        match_idx = self._find_matching_action(actual_action, likely_actions)

        if match_idx is not None and 0 <= match_idx < len(draft_frames):
            # Cache hit: serve pre-computed frame instantly
            return draft_frames[match_idx], True
        else:
            # Cache miss: need full forward with actual action
            if model is not None and x is not None:
                model.eval()
                with torch.no_grad():
                    output = self._safe_forward(
                        model, x, timestep, actual_action, **kwargs
                    )
                return output, False
            else:
                raise ValueError(
                    "Cache miss but model/x not provided for fallback forward pass."
                )

    def _find_matching_action(
        self,
        actual: Any,
        predictions: List[Any],
    ) -> Optional[int]:
        """
        Find index of actual action in predictions list.

        Returns None if no match found.
        """
        for i, pred in enumerate(predictions):
            if _actions_match(actual, pred):
                return i
        return None

    # ------------------------------------------------------------------
    # Internal: model-agnostic forward call
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_forward(
        model: nn.Module,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor],
        action: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Call model with graceful fallback for different DiT interfaces.

        Tries three calling conventions in order:
          1. model(x, actions=action, timestep=timestep, **kwargs)
          2. model(x, t_BT) — Oasis DiT positional (t expanded to (B,T))
          3. model(x) — bare forward

        Returns the first element if model returns a tuple/list.
        """
        fwd_kwargs: Dict[str, Any] = dict(kwargs)
        if action is not None:
            fwd_kwargs["actions"] = action
        if timestep is not None:
            fwd_kwargs["timestep"] = timestep

        if isinstance(x, dict):
            try:
                out = model(**x, **fwd_kwargs)
            except TypeError:
                fwd_kwargs.pop("actions", None)
                fwd_kwargs.pop("timestep", None)
                out = model(**x, **fwd_kwargs)
        else:
            # Use autocast to handle DiT models whose sinusoidal t_embedder
            # internally computes float32 tensors that then hit float16 weights.
            amp_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                try:
                    out = model(x, **fwd_kwargs)
                except TypeError:
                    # Oasis DiT: positional (x, t_BT); remap actions→external_cond
                    fwd_kwargs2: Dict[str, Any] = {
                        k: v for k, v in fwd_kwargs.items()
                        if k not in ("actions", "timestep")
                    }
                    action_val = fwd_kwargs.get("actions")
                    # Only pass external_cond if it is a proper tensor (not a scalar int)
                    if isinstance(action_val, torch.Tensor):
                        fwd_kwargs2["external_cond"] = action_val
                    t_arg = fwd_kwargs.get("timestep")
                    if t_arg is not None and isinstance(t_arg, torch.Tensor):
                        B = x.shape[0]
                        T = x.shape[1] if x.dim() >= 2 else 1
                        if t_arg.dim() == 1 and t_arg.shape[0] == B:
                            t_arg = t_arg.unsqueeze(1).expand(B, T).to(x.dtype).contiguous()
                        try:
                            out = model(x, t_arg, **fwd_kwargs2)
                        except TypeError:
                            out = model(x, **fwd_kwargs2)
                    else:
                        out = model(x, **fwd_kwargs2)

        return out[0] if isinstance(out, (tuple, list)) else out

    def streaming_loop(
        self,
        model: nn.Module,
        action_stream: List[Any],
        kv_cache: Optional[Any] = None,
        num_frames: int = 30,
        timestep_fn: Optional[Callable[[int], torch.Tensor]] = None,
        x_init: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Full streaming generation loop with tree-based action branching.

        For each frame:
          1. Predict K likely next actions from history
          2. Batch-draft K frames in parallel
          3. When actual action arrives, select matching frame or fallback
          4. Update action history

        Args:
            model: The DiT model.
            action_stream: Pre-recorded or live action sequence (length >= num_frames).
            kv_cache: Optional KV cache manager.
            num_frames: Number of frames to generate.
            timestep_fn: Function mapping frame_idx -> timestep tensor.
                Defaults to returning zeros.
            x_init: Initial latent tensor. If None, uses zeros.
            **kwargs: Additional model arguments.

        Returns:
            Tuple of:
              - output_frames: List of generated frame tensors.
              - stats: Summary statistics dict.
        """
        output_frames: List[torch.Tensor] = []
        action_history: List[Any] = []

        # Initialize
        if x_init is None:
            # Placeholder; real usage would provide initial latent
            logger.warning("No x_init provided; using zeros placeholder.")
            x_init = torch.zeros(1, 4, 1, 32, 32)  # Minimal placeholder

        x = x_init

        for frame_idx in range(min(num_frames, len(action_stream))):
            actual_action = action_stream[frame_idx]
            timestep = (
                timestep_fn(frame_idx) if timestep_fn is not None
                else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            )

            if len(action_history) >= 1:
                # --- Predict and pre-compute using HW5 worker pattern ---
                likely_actions = self.predict_likely_actions(action_history, self.k_actions)

                draft_frames, batch_time_ms = self.parallel_branch_draft(
                    model, x, likely_actions, timestep=timestep, **kwargs
                )
                self.stats.batch_draft_times_ms.append(batch_time_ms)
                self.stats.sequential_estimate_ms.append(
                    self._single_fwd_ms * len(likely_actions)
                )

                # --- Select: return pre-computed draft on hit, forward on miss ---
                t0 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                match_idx = self._find_matching_action(actual_action, likely_actions)
                if match_idx is not None and 0 <= match_idx < len(draft_frames):
                    # Cache hit: serve pre-computed frame directly — NO forward pass
                    output = draft_frames[match_idx]
                    was_hit = True
                else:
                    # Cache miss: run forward with actual action
                    if cache_snapshot is not None and kv_cache is not None and hasattr(kv_cache, "restore"):
                        kv_cache.restore(cache_snapshot)
                    with torch.no_grad():
                        output = self._safe_forward(
                            model, x, timestep, actual_action, **kwargs
                        )
                    was_hit = False

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                select_time_ms = (time.perf_counter() - t0) * 1000.0

                if was_hit:
                    self.stats.cache_hits += 1
                    self.stats.hit_latencies_ms.append(select_time_ms)
                else:
                    self.stats.cache_misses += 1
                    self.stats.miss_latencies_ms.append(select_time_ms)

            else:
                # First frame: no history — run standard forward and time it
                # for honest speedup baseline
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_single = time.perf_counter()
                with torch.no_grad():
                    output = self._safe_forward(
                        model, x, timestep, actual_action, **kwargs
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._single_fwd_ms = (time.perf_counter() - t_single) * 1000.0
                cache_snapshot = None  # initialise for later use

            self.stats.total_frames += 1
            output_frames.append(output)
            action_history.append(actual_action)

            # Use output as next input (autoregressive)
            x = output

        return output_frames, self.get_stats()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return hit rate, average latency, and speedup statistics.

        Returns:
            Dict with:
              - hit_rate: Fraction of frames served from cache
              - total_frames: Total frames generated
              - cache_hits / cache_misses: Counts
              - avg_hit_latency_ms: Average latency for cache hits
              - avg_miss_latency_ms: Average latency for cache misses
              - avg_batch_draft_ms: Average time for batched K-action draft
              - estimated_speedup: vs sequential single-action generation
        """
        stats = self.stats
        total = stats.cache_hits + stats.cache_misses

        hit_rate = stats.cache_hits / total if total > 0 else 0.0

        avg_hit = (
            sum(stats.hit_latencies_ms) / len(stats.hit_latencies_ms)
            if stats.hit_latencies_ms else 0.0
        )
        avg_miss = (
            sum(stats.miss_latencies_ms) / len(stats.miss_latencies_ms)
            if stats.miss_latencies_ms else 0.0
        )
        avg_batch = (
            sum(stats.batch_draft_times_ms) / len(stats.batch_draft_times_ms)
            if stats.batch_draft_times_ms else 0.0
        )

        # Speedup estimate (honest):
        # Baseline: single forward per frame = _single_fwd_ms (measured on frame 0)
        # With tree branching:
        #   - branch cost paid upfront: avg_batch (K sequential or K parallel)
        #   - hit (hit_rate %): zero additional cost — draft frame returned directly
        #   - miss (1-hit_rate %): one additional forward = avg_miss
        # Effective per-frame cost = avg_batch + (1-hit_rate) * avg_miss
        # Speedup = single_fwd / effective_cost
        single_fwd = self._single_fwd_ms if self._single_fwd_ms > 0 else avg_batch
        if avg_batch > 0 and total > 0:
            tree_avg = avg_batch + (1.0 - hit_rate) * avg_miss
            estimated_speedup = single_fwd / tree_avg if tree_avg > 0 else 1.0
        else:
            estimated_speedup = 1.0

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return {
            "total_frames": stats.total_frames,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "hit_rate": hit_rate,
            "avg_hit_latency_ms": avg_hit,
            "avg_miss_latency_ms": avg_miss,
            "avg_batch_draft_ms": avg_batch,
            "single_fwd_ms": round(single_fwd, 2),
            "estimated_speedup_internal": round(estimated_speedup, 3),
            "num_gpus": num_gpus,
            "parallel_mode": "multi-gpu" if num_gpus >= self.k_actions else "sequential",
            "k_actions": self.k_actions,
            "action_predictor": self.action_predictor,
        }

    def reset_stats(self) -> None:
        """Reset all tracked statistics."""
        self.stats = TreeBranchStats()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _tensor_hash(t: torch.Tensor, precision: int = 4) -> int:
    """Approximate hash for tensor comparison (rounds to `precision` decimals)."""
    rounded = (t.float().cpu() * (10 ** precision)).round().long()
    return hash(rounded.numpy().tobytes())


def _actions_match(a: Any, b: Any, tensor_atol: float = 1e-4) -> bool:
    """
    Check if two actions are equivalent.

    Handles tensors (approximate equality), ints, strings, and dicts.
    """
    if type(a) != type(b):
        return False

    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return False
        return torch.allclose(a.float(), b.float(), atol=tensor_atol)

    elif isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_actions_match(a[k], b[k], tensor_atol) for k in a)

    else:
        return a == b
