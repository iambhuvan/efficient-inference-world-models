"""
WorldServe optimizations package.

Provides pipeline-level, step-caching, KV-cache, sparse attention,
and speculative decoding optimizations for video world models.
"""

from typing import Optional


def get_pipeline_optimizers() -> dict:
    """Return a dictionary of available pipeline optimization classes."""
    optimizers = {}

    try:
        from worldserve.optimizations.system_level.pipeline.flash_attention import FlashAttention3Replacer
        optimizers["flash_attention"] = FlashAttention3Replacer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.pipeline.compile_optimizer import CompileOptimizer
        optimizers["compile"] = CompileOptimizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.pipeline.fp8_quantizer import DiTFP8Quantizer
        optimizers["fp8"] = DiTFP8Quantizer
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.pipeline.cuda_streams import CUDAStreamPipeline
        optimizers["cuda_streams"] = CUDAStreamPipeline
    except ImportError:
        pass

    try:
        from worldserve.optimizations.system_level.pipeline.fused_kernels import FusedAdaLN, FusedTimeEmbedding
        optimizers["fused_adaln"] = FusedAdaLN
        optimizers["fused_time_embed"] = FusedTimeEmbedding
    except ImportError:
        pass

    return optimizers


class OptimizationStack:
    """
    Container that holds all enabled optimization modules and provides
    hooks for the generation loop (pre_step, post_step, diffusers_callback).
    """

    def __init__(self, modules: dict[str, object], config: dict):
        self.modules = modules
        self.config = config
        self._stats: dict[str, object] = {}

    def pre_step(self, step_idx: int, total_steps: int, latents) -> bool:
        """
        Called before each denoising step.
        Returns True if the step should be SKIPPED (cache hit).
        """
        skip = False
        for name, mod in self.modules.items():
            if hasattr(mod, "pre_step"):
                result = mod.pre_step(step_idx, total_steps, latents)
                if result:
                    skip = True
        return skip

    def post_step(self, step_idx: int, total_steps: int, latents):
        """Called after each denoising step to update caches / stats."""
        for name, mod in self.modules.items():
            if hasattr(mod, "post_step"):
                mod.post_step(step_idx, total_steps, latents)

    def diffusers_callback(self, pipe, step_idx, timestep, callback_kwargs):
        """
        Callback compatible with diffusers `callback_on_step_end`.
        """
        latents = callback_kwargs.get("latents")
        if latents is not None:
            for name, mod in self.modules.items():
                if hasattr(mod, "on_step_end"):
                    callback_kwargs = mod.on_step_end(
                        step_idx, timestep, callback_kwargs
                    )
        return callback_kwargs

    def get_stats(self) -> dict:
        """Aggregate stats from all optimisation modules."""
        stats = {}
        for name, mod in self.modules.items():
            if hasattr(mod, "get_stats"):
                stats[name] = mod.get_stats()
        return stats

    def apply_to_model(self, model) -> None:
        """
        Apply pending structural optimizations that require a model reference.

        Call this after model load and BEFORE torch.compile.  Handles:
          - radial_attention_pending: calls RadialAttentionReplacer.wrap_model(model)
            and promotes the entry to modules["radial_attention"].
          - dydit_pending: instantiates DyDiTPlusPlus(model, ...) from stored config
            dict, calls wrap_model(model), and promotes to modules["dydit"].

        Both modifications are in-place on the model.  After this call no
        "_pending" keys remain in self.modules.
        """
        # --- RadialAttention ---
        if "radial_attention_pending" in self.modules:
            replacer = self.modules.pop("radial_attention_pending")
            try:
                replacer.wrap_model(model)
                # wrap_model returns nn.Module; read count from the replacer's own tracking list
                n_replaced = len(getattr(replacer, "_replaced_modules", []))
                self.modules["radial_attention"] = replacer
                print(f"[opt] RadialAttention applied: replaced {n_replaced} attention module(s)")
            except Exception as e:
                print(f"[opt] RadialAttention apply_to_model failed: {e} — skipping")

        # --- DyDiT++ ---
        if "dydit_pending" in self.modules:
            dydit_cfg = self.modules.pop("dydit_pending")
            try:
                from worldserve.optimizations.model_level.dynamic_compute.dydit_plus_plus import DyDiTPlusPlus
                dydit = DyDiTPlusPlus(
                    model=model,
                    embed_dim=dydit_cfg.get("embed_dim", 256),
                    min_scale=dydit_cfg.get("min_scale", 0.5),
                    max_scale=dydit_cfg.get("max_scale", 1.0),
                    entropy_threshold=dydit_cfg.get("entropy_threshold", 0.5),
                    skip_ratio_max=dydit_cfg.get("skip_ratio_max", 0.4),
                    apply_lora=dydit_cfg.get("apply_lora", True),
                )
                dydit.wrap_model(model)
                self.modules["dydit"] = dydit
                print("[opt] DyDiT++ applied: dynamic width + spatial token skipping active")
            except Exception as e:
                print(f"[opt] DyDiT++ apply_to_model failed: {e} — skipping")


def build_optimization_stack(config: dict) -> OptimizationStack:
    """
    Instantiate all enabled optimizations according to *config* and
    return an OptimizationStack ready to be passed to generate().
    """
    modules: dict[str, object] = {}

    # -----------------------------------------------------------------------
    # Conflict resolution: detect model-level / system-level interactions
    # -----------------------------------------------------------------------

    # 1. num_steps propagation: if a distilled model changes step count,
    #    update step-caching configs to match.
    effective_num_steps = config.get("num_steps", None)
    if effective_num_steps is not None:
        # Push num_steps into step_caching sub-configs
        if "step_caching" in config:
            config["step_caching"].setdefault("num_steps", effective_num_steps)
        if effective_num_steps == 1:
            # rCM / single-step model: step caching between steps is useless
            print("[opt] num_steps=1 detected — disabling all step/feature caching "
                  "(SeaCache, FlowCache, ERTACache, MagCache, TaylorSeer have no steps to cache between)")
            config.get("step_caching", {})["enabled"] = False
            config.get("feature_caching", {})["enabled"] = False

    # 2. Feature caching + torch.compile: no conflict.
    #    MagCache, TaylorSeer, and ERTACache all use @torch._dynamo.disable on their
    #    cached_forward and post_hook_fn closures.  The routing logic runs eagerly
    #    while original_fwd (the compiled kernel) is still dispatched through Triton.
    #    This is the same pattern SeaCache uses and is verified correct.

    # 3. Radial Attention + Flash Attention coexistence:
    #    Both replace attention .forward. Flag this so flash_attention skips
    #    RadialAttentionModule instances (handled in flash_attention.py).
    if config.get("radial_attention", {}).get("enabled") and config.get("use_flash_attention", True):
        print("[opt] Radial Attention + Flash Attention active: FA3 will skip "
              "RadialAttentionModule instances (radial sparse mask takes precedence)")

    # 3. DC-DiT + SVG2: update tokens_per_frame in sparse_attention config
    #    when DC-DiT compresses sequence length.
    if config.get("dc_dit", {}).get("enabled"):
        dc_ratio = config["dc_dit"].get("compression_ratio", 4)
        orig_tpf = config.get("sparse_attention", {}).get("tokens_per_frame", 15360)
        compressed_tpf = max(1, orig_tpf // dc_ratio)
        if "sparse_attention" in config:
            config["sparse_attention"]["tokens_per_frame"] = compressed_tpf
            print(f"[opt] DC-DiT active: SVG2 tokens_per_frame updated "
                  f"{orig_tpf} → {compressed_tpf} (compression_ratio={dc_ratio})")

    # 4. DyDiT++ + torch.compile: dynamic shapes required
    if config.get("dydit", {}).get("enabled"):
        print("[opt] DyDiT++ active: torch.compile will use dynamic=True "
              "(call CompileOptimizer.register_dynamic_module(model) before compile)")

    # 6. Radial Attention + compile: mask is static, compatible with CUDA Graphs.
    #    Stored as "radial_attention_pending"; call opt_stack.apply_to_model(model)
    #    after model load and BEFORE torch.compile to install the replacer.
    if config.get("radial_attention", {}).get("enabled", False):
        try:
            from worldserve.optimizations.model_level.attention.radial_attention import RadialAttentionReplacer
            ra_cfg = config["radial_attention"]
            modules["radial_attention_pending"] = RadialAttentionReplacer(
                tokens_per_frame=ra_cfg.get("tokens_per_frame", 15360),
                num_frames=ra_cfg.get("num_frames", 6),
                base_spatial_window=ra_cfg.get("base_spatial_window", 128),
                min_window=ra_cfg.get("min_spatial_window", 8),
            )
            print("[opt] RadialAttention queued; call opt_stack.apply_to_model(model) before compile")
        except ImportError:
            print("[opt] RadialAttentionReplacer not available, skipping")
        except Exception as e:
            print(f"[opt] RadialAttentionReplacer init failed: {e}, skipping")

    # 7. DyDiT++ — requires model ref for instantiation.  Config stored as
    #    "dydit_pending"; call opt_stack.apply_to_model(model) after model load
    #    and BEFORE torch.compile to instantiate DyDiTPlusPlus and wrap the model.
    if config.get("dydit", {}).get("enabled", False):
        try:
            from worldserve.optimizations.model_level.dynamic_compute.dydit_plus_plus import DyDiTPlusPlus  # noqa: F401
            modules["dydit_pending"] = config["dydit"]
            print("[opt] DyDiT++ queued; call opt_stack.apply_to_model(model) before compile")
        except ImportError:
            print("[opt] DyDiTPlusPlus not available, skipping")
        except Exception as e:
            print(f"[opt] DyDiTPlusPlus init failed: {e}, skipping")

    # 5. Speculative decoding with rCM 1-step: not beneficial
    if effective_num_steps == 1 and config.get("speculative", {}).get("enabled"):
        print("[opt] Speculative decoding disabled: num_steps=1 (rCM) means "
              "draft+verify costs 2x a single step — no benefit")
        config["speculative"]["enabled"] = False

    # Step caching
    if config.get("step_caching", {}).get("enabled", False):
        method = config["step_caching"].get("method", "seacache")
        try:
            if method in ("seacache", "hybrid"):
                from worldserve.optimizations.system_level.step_caching.seacache import SeaCacheOptimizer
                sc_cfg = config["step_caching"].get("seacache", {})
                modules["seacache"] = SeaCacheOptimizer(
                    num_layers=sc_cfg.get("num_layers", 30),
                    num_steps=config["step_caching"].get("num_steps", 3),
                    frequency_threshold=sc_cfg.get("similarity_threshold", 0.5),
                    cache_ratio=sc_cfg.get("max_skip_ratio", 0.6),
                    spatial_hw=sc_cfg.get("spatial_hw", None),
                )
        except ImportError:
            print("[opt] SeaCache module not available, skipping")
        except Exception as e:
            print(f"[opt] SeaCache init failed: {e}, skipping")
        try:
            if method in ("flowcache", "hybrid"):
                from worldserve.optimizations.system_level.step_caching.flowcache import FlowCacheOptimizer
                fc_cfg = config["step_caching"].get("flowcache", {})
                modules["flowcache"] = FlowCacheOptimizer(
                    redundancy_threshold=fc_cfg.get("error_threshold", 0.85),
                )
        except ImportError:
            print("[opt] FlowCache module not available, skipping")
        except Exception as e:
            print(f"[opt] FlowCache init failed: {e}, skipping")

    # KV-cache compression
    if config.get("kv_cache", {}).get("enabled", False):
        try:
            from worldserve.optimizations.system_level.kv_cache.manager import KVCacheManager
            modules["kv_cache"] = KVCacheManager(config["kv_cache"])
        except ImportError:
            print("[opt] KV-cache module not available, skipping")
        except KeyError as e:
            print(f"[opt] KV-cache config key missing: {e}, skipping")

    # Sparse attention (structural model modification — apply(model) called in modal_app.py)
    if config.get("sparse_attention", {}).get("enabled", False):
        try:
            from worldserve.optimizations.system_level.sparse_attention import SparseAttentionOptimizer
            modules["sparse_attention"] = SparseAttentionOptimizer(config["sparse_attention"])
            print(f"[opt] Sparse attention queued (method={config['sparse_attention'].get('method', 'svg2')}); "
                  "call opt_stack.modules['sparse_attention'].apply(model) after model load")
        except Exception as e:
            print(f"[opt] Sparse attention init failed: {e}, skipping")

    # Speculative decoding (needs model ref — call .build(model) in modal_app.py after model load)
    if config.get("speculative", {}).get("enabled", False):
        try:
            from worldserve.optimizations.system_level.speculative import SpeculativeDecoder
            modules["speculative"] = SpeculativeDecoder(config["speculative"])
            print("[opt] Speculative decoder queued; "
                  "call opt_stack.modules['speculative'].build(model) after model load")
        except Exception as e:
            print(f"[opt] Speculative decoding init failed: {e}, skipping")

    # -----------------------------------------------------------------------
    # Sprint 1 — Model-level, zero-retraining
    # -----------------------------------------------------------------------

    # ODE sampler (Oasis): DPM-Solver++ replaces Euler integrator
    if config.get("sampler", {}).get("method") in ("dpm_solver_pp", "dpm_solver"):
        try:
            from worldserve.optimizations.model_level.samplers import DPMSolverPPSampler
            order = config["sampler"].get("order", 2)
            modules["sampler"] = DPMSolverPPSampler(order=order)
        except ImportError:
            print("[opt] DPMSolverPPSampler not available, skipping")
        except Exception as e:
            print(f"[opt] DPMSolverPPSampler init failed: {e}, skipping")

    # Guidance: APG (drop-in CFG replacement, both models)
    if config.get("guidance", {}).get("method") == "apg":
        try:
            from worldserve.optimizations.model_level.guidance import APGGuidance
            g_cfg = config["guidance"]
            modules["guidance"] = APGGuidance(
                guidance_scale=g_cfg.get("guidance_scale", 7.5),
                alpha_parallel=g_cfg.get("alpha_parallel", 0.5),
                momentum_eta=g_cfg.get("momentum_eta", 0.9),
                momentum_beta=g_cfg.get("momentum_beta", 0.1),
            )
        except ImportError:
            print("[opt] APGGuidance not available, skipping")
        except Exception as e:
            print(f"[opt] APGGuidance init failed: {e}, skipping")

    # Guidance: NAG (few-step regime — distilled Oasis or MG2 3-step)
    if config.get("guidance", {}).get("method") == "nag":
        try:
            from worldserve.optimizations.model_level.guidance import NAGHook
            modules["guidance_nag_pending"] = config["guidance"]  # applied after model ref available
        except ImportError:
            print("[opt] NAGHook not available, skipping")

    # Feature caching (MG2 primary, stacks with SeaCache)
    fc_cfg = config.get("feature_caching", {})
    if fc_cfg.get("enabled", False):
        fc_method = fc_cfg.get("method", "mag")
        try:
            if fc_method == "erta":
                from worldserve.optimizations.model_level.feature_caching import ERTACache
                modules["feature_cache"] = ERTACache(
                    num_layers=fc_cfg.get("num_layers", 30),
                    num_steps=fc_cfg.get("num_steps", 3),
                    threshold=fc_cfg.get("threshold", 0.1),
                    correction_strength=fc_cfg.get("correction_strength", 0.3),
                )
            elif fc_method == "mag":
                from worldserve.optimizations.model_level.feature_caching import MagCache
                modules["feature_cache"] = MagCache(
                    num_layers=fc_cfg.get("num_layers", 30),
                    num_steps=fc_cfg.get("num_steps", 3),
                    cache_threshold_percentile=fc_cfg.get("threshold_percentile", 80),
                    min_threshold=fc_cfg.get("min_threshold", 0.02),
                    max_threshold=fc_cfg.get("max_threshold", 0.15),
                )
            elif fc_method == "taylor":
                from worldserve.optimizations.model_level.feature_caching import TaylorSeer
                modules["feature_cache"] = TaylorSeer(
                    num_layers=fc_cfg.get("num_layers", 30),
                    num_steps=fc_cfg.get("num_steps", 3),
                    order=fc_cfg.get("order", 1),
                    prediction_threshold=fc_cfg.get("prediction_threshold", 0.15),
                )
        except ImportError:
            print(f"[opt] feature_caching/{fc_method} not available, skipping")
        except Exception as e:
            print(f"[opt] feature_caching/{fc_method} init failed: {e}, skipping")

    # Pipeline-level: FA3, FP8, compile are applied directly to the model,
    # not through the stack. They are handled in modal_app.py / generate().

    print(f"[opt] Optimization stack built with {len(modules)} module(s): {list(modules.keys())}")
    return OptimizationStack(modules, config)


def reconfigure_for_distilled_model(
    stack: OptimizationStack,
    num_steps: int,
    model_type: str = "oasis",
) -> None:
    """
    Update a running OptimizationStack after loading a distilled model checkpoint.

    When a PCM-distilled Oasis (20→4 steps) or rCM MG2 (3→1 step) replaces
    the baseline model, the stack's step-caching schedules must be updated.

    Args:
        stack: The existing OptimizationStack to update.
        num_steps: New denoising step count (e.g., 4 for PCM Oasis, 1 for rCM MG2).
        model_type: "oasis" or "mg2" — controls which modules are reconfigured.
    """
    # Update SeaCache num_steps
    if "seacache" in stack.modules:
        sc = stack.modules["seacache"]
        if hasattr(sc, "reconfigure"):
            sc.reconfigure(num_steps)

    # Disable step-level caching entirely at 1 step
    if num_steps == 1:
        for key in ("seacache", "flowcache", "feature_cache"):
            if key in stack.modules:
                del stack.modules[key]
                print(f"[opt] reconfigure: removed {key} (num_steps=1, no caching benefit)")

    print(f"[opt] reconfigure_for_distilled_model: {model_type}, num_steps={num_steps}")
