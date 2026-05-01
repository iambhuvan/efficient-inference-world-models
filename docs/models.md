# WorldServe — Final H100-Fusable Model List

---

## H100 Benchmark Parameters & Time Estimates (Modal Scripts)

These are the **7 models actively benchmarked** in `modal/run_all.py`.
One warmup run + one timed run per script. Model weights cached in Modal volume after first download.

### What drives inference time

| Factor | Effect on time | Example |
|--------|---------------|---------|
| DDIM steps | **Linear** — 2× steps = 2× time | 10 steps → 20 steps = 2× |
| Token count | **Super-linear** — attention is O(N²), MLP is O(N) | 2× tokens ≈ 3-5× slower |
| Model params | **Linear multiplier** on all compute | 14B vs 5B ≈ 2.8× slower per token |
| Resolution | Determines token count via patchification (spatial/8 then /patch_size) | 720p vs 480p ≈ 2.25× more tokens |
| Frames | Multiplied into token count after temporal compression | 129f vs 49f ≈ 2.6× more tokens |

### Per-model benchmark config (from modal scripts)

| Model | Params | DDIM Steps | Frames | Resolution | Attn Tokens | Est. 1 generation | Est. total GPU time (load + 2 runs) | Risk |
|-------|--------|-----------|--------|------------|-------------|-------------------|--------------------------------------|------|
| **Oasis-500M** | 500M | 20 | 16 | 256×256 | ~1K | ~10-20s | **~3-5 min** | Medium — API uncertain |
| **FLUX.1-schnell** | 12B | 20 | 1 (image) | 1024×1024 | ~16K | ~30-60s | **~4-8 min** | Low — public, ungated |
| **LTX-Video** | ~2B | 20 | 121 | 480×704 | ~14K | ~60-120s | **~5-10 min** | Low — public |
| **CogVideoX-5b** | 5B | 20 | 49 | 480×720 | ~17.5K | **~108s** ✅ confirmed | **~7-10 min** | Low — working |
| **Wan2.1-T2V-14B** | 14B | 20 | 81 | 480×720 | ~27K | ~5-15 min | **~15-35 min** | Low — public |
| **Cosmos-7B-V2W** | 7B | 20 | 121 | 480×720 | ~56K | ~10-25 min | **~25-55 min** | High — NVIDIA gated |
| **HunyuanVideo** | 13B | 20 | 129 | 720×1280 | ~115K | ~20-45 min | **~45-95 min** | Medium — community model |

### Optimised run estimates (after kernel injection)

| Model | Baseline est. | STA optimised | SageAttn2 optimised | TeaCache/PrediT optimised |
|-------|--------------|---------------|---------------------|--------------------------|
| Oasis-500M | ~15s | — | — | ~8-10s (TeaCache+PrediT+INT4 stacked) |
| FLUX.1-schnell | ~45s | — | ~30-35s (1.3x) | ~20-30s (PrediT 1.5-2x) |
| LTX-Video | ~90s | ~50-65s (1.4-1.8x) | ~60-70s (1.3-1.5x) | — |
| CogVideoX-5b | ~108s | ~65-80s (1.4-1.7x) | ~75-90s (1.2-1.4x) | — |
| Wan2.1-14B | ~10 min | — | — | ~5-7 min (TeaCache 1.4-2x) |
| Cosmos-7B | ~18 min | ~8-12 min (1.5-2.5x) | ~12-14 min (1.3-1.5x) | ~9-13 min (TeaCache/PrediT) |
| HunyuanVideo | ~30 min | ~12-20 min (1.5-2.5x) | — | — |

### Priority recommendation (limited H100 budget)

```
Tier A — Run these first (~30 min total for all baselines + optimised)
  1. CogVideoX-5b  → baseline DONE, just need STA + SageAttn2 (2 runs ~4 min)
  2. FLUX.1-schnell → baseline + SageAttn2 + PrediT     (3 runs ~12 min)
  3. LTX-Video     → baseline + STA + SageAttn2         (3 runs ~15 min)
  4. Oasis-500M    → baseline + all-kernels             (2 runs ~8 min)

Tier B — Good coverage if budget allows (~60-90 min more)
  5. Wan2.1-14B    → baseline + TeaCache               (2 runs ~35 min)
  6. Cosmos-7B     → baseline + STA                    (2 runs ~50 min) — access may fail

Skip — too slow or too risky for limited access
  7. HunyuanVideo  → 45-95 min total, only STA available — do last
```

---

**Selection criteria (all three must pass):**
1. Custom CUDA attention kernel can be injected via `set_attn_processor()`, `dispatch_attention_fn`, or direct Python monkeypatch
2. Sequence length ≥ 512 tokens at inference (H100 attention speedup threshold); weight-quant models need ≥ 256-dim weight matrices
3. Model weights publicly available on HuggingFace (or gated with free access)

**Verified 2026-04-28 using 11 parallel research agents across HuggingFace Hub, GitHub source code, arXiv papers, and H100 benchmarking literature.**

---

## Category 1 — Video DiTs (Primary Kernel Target)

Long temporal sequences make every per-step kernel saving compound across 20–50 denoising steps.
Injection method: `pipeline.transformer.set_attn_processor(YourProcessor())` or `DIFFUSERS_ATTN_BACKEND=_sage_qk_int8_pv_fp8_cuda_sm90`.

| Model ID | Downloads | License | Params | Seq Tokens | Attn% | Attention Class | Injection |
|----------|-----------|---------|--------|-----------|-------|----------------|-----------|
| ✅ `Lightricks/LTX-Video` | 5.5M | other | ~2B | ~14,400 (720p/121f) | ~45% | `LTXVideoAttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| `Lightricks/LTX-Video-0.9.7-distilled` | 58.7K | other | ~2B | ~14,400 | ~45% | Same as base | `set_attention_backend()` |
| ✅ `zai-org/CogVideoX-5b` | 1.4M | other | 5B | ~17,550 (720p/49f) | 51% | `CogVideoXAttnProcessor2_0` → direct `F.sdpa` | `set_attn_processor()` |
| `zai-org/CogVideoX-2b` | 965K | Apache-2.0 | 2B | ~17,550 | 51% | Same as 5b | `set_attn_processor()` |
| `zai-org/CogVideoX1.5-5B` | 183K | other | 5B | >17,550 (1360×768) | ~55% | CogVideoX extended | `set_attn_processor()` |
| ✅ `tencent/HunyuanVideo` | 49.7K | other | ~13B | ~115,200 (720p/129f) | >80% | `HunyuanVideoAttnProcessor2_0` + `dispatch_attention_fn` | `set_attention_backend()` |
| `tencent/HunyuanVideo-I2V` | 18.7K | other | ~13B | ~115,200 | >80% | Same as T2V | `set_attention_backend()` |
| `Wan-AI/Wan2.1-I2V-14B-480P` | 1.2M | Apache-2.0 | 14B | ~20,160 (480p/81f) | ~50% | `WanAttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| `Wan-AI/Wan2.1-I2V-14B-720P` | 993K | Apache-2.0 | 14B | ~27,000 (720p/81f) | ~50% | `WanAttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| ✅ `Wan-AI/Wan2.1-T2V-14B` | 848K | Apache-2.0 | 14B | ~27,000 | ~50% | `WanAttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| `Wan-AI/Wan2.2-T2V-A14B` | 78.1K | Apache-2.0 | ~14B | ~27,000 | ~50% | `WanAttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| `Wan-AI/Wan2.1-T2V-1.3B` | 201K | Apache-2.0 | 1.3B | ~20,000 | ~50% | Same as 14B | `set_attention_backend()` |
| `Wan-AI/Wan2.1-VACE-14B` | 118K | Apache-2.0 | 14B | ~27,000 | ~50% | `vace` DiT + dispatch | `set_attention_backend()` |
| `genmo/mochi-1-preview` | 374K | Apache-2.0 | ~10B | ~18,000 (480p/163f) | ~45% | `MochiAttnProcessor2_0` → direct `F.sdpa` | per-module: `module.processor = YourProc()` |
| `Skywork/SkyReels-V2-DF-14B-720P` | 40.2K | other | 14B | ~28,000 (720p/121f) | ~50% | `SkyReelsV2AttnProcessor` + `dispatch_attention_fn` | `set_attention_backend()` |
| `Skywork/SkyReels-V2-I2V-14B-720P` | 53.6K | other | 14B | ~28,000 | ~50% | Same as DF | `set_attention_backend()` |
| `Skywork/SkyReels-V1-Hunyuan-I2V` | 71.5K | other | ~13B | ~115,200 | >80% | HunyuanVideo-derived | `set_attention_backend()` |
| `rhymes-ai/Allegro` | — | Apache-2.0 | 2.8B | ~79,200 (720p/88f) | ~70% | `AllegroAttnProcessor2_0` → direct `F.sdpa` | `set_attn_processor()` |
| `hpcai-tech/OpenSora-STDiT-v3` | 1.4M | Apache-2.0 | 1.21B | ~60-80K (720p/10s) | ~65% | `Attention` in opensora.layers → SDPA/flash | subclass `Attention.forward`, disable `sp_size>1` |
| `LanguageBind/Open-Sora-Plan-v1.3.0` | — | MIT | — | ~13-40K (480p) | ~50% | `AttnProcessor2_0` via `torch.backends.cuda.sdp_kernel` | replace `AttnProcessor2_0`; target full-attn layers only |
| `alibaba-pai/EasyAnimateV5-12b-zh` | 414 | Apache-2.0 | 12B | ~12-20K (HWA) | ~40% | `EasyAnimateAttnProcessor2_0` → direct `F.sdpa` | `set_attn_processor()`; inject in global-attn blocks |
| `alibaba-pai/CogVideoX-Fun-V1.1-5b-InP` | 78.6K | other | 5B | ~17,550 | 51% | CogVideoX-based | `set_attn_processor()` |

**Kernel targets for video DiTs:**

| Kernel | LTX-Video | CogVideoX-5b | HunyuanVideo | Wan2.1-14B | Mochi | Allegro | OpenSora-STDiT |
|--------|-----------|-------------|-------------|-----------|-------|---------|----------------|
| K1 PrediT AB-2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| K2 Radial Attention | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| K3 SageAttention2 FP8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| K4 STA sparse | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| K5 TeaCache | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| K6 LiteAttn skip-list | ✓ | ✓ | ✓ | — | — | — | ✓ |
| K7 TempCache KV dedup | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| K9 ClusCa | — | ✓ | ✓ | ✓ | — | ✓ | — |
| K13 INT4 KV | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| K14 QVG 2-bit temporal | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |

---

## Category 2 — Image DiTs

20–50 denoising steps × 4,096–16,384 tokens. Injection via `set_attn_processor()`.

| Model ID | Downloads | License | Params | Seq Tokens | Architecture | Injection Class | H100 Benefit |
|----------|-----------|---------|--------|-----------|-------------|----------------|--------------|
| ✅ `black-forest-labs/FLUX.1-dev` | 29.8M | BFL (nc) | ~12B | 16,384 (1024px) | FLUX MMDiT dual-stream | `FluxAttnProcessor` + `dispatch_attention_fn` | HIGH (2-3×) |
| `black-forest-labs/FLUX.1-schnell` | 20.7M | Apache-2.0 | ~12B | 16,384 | FLUX MMDiT 4-step | Same as dev | HIGH |
| `black-forest-labs/FLUX.1-Kontext-dev` | 2.6M | BFL (nc) | ~12B | 16,384 | FLUX MMDiT (Jan 2026) | Same | HIGH |
| `ostris/OpenFLUX.1` | 368K | Apache-2.0 | ~12B | 16,384 | FLUX MMDiT (open) | Same | HIGH |
| `stabilityai/stable-diffusion-3.5-large` | 2.3M | Stability AI | ~8B | 4,096 (1024px) | SD3.5 MMDiT-X | `JointAttnProcessor2_0` → direct `F.sdpa` | MEDIUM-HIGH |
| `stabilityai/stable-diffusion-3.5-medium` | 3.6M | Stability AI | ~2.5B | 4,096 | SD3.5 MMDiT-X | `JointAttnProcessor2_0` → direct `F.sdpa` | MEDIUM-HIGH |
| `stabilityai/stable-diffusion-3-medium-diffusers` | 3.9M | Stability AI | ~2B | 4,096 | SD3 MMDiT | `JointAttnProcessor2_0` | MEDIUM |
| `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | 470K | OpenRAIL++ | 611M | 4,096 | PixArt-Sigma cross-attn DiT | `AttnProcessor2_0` → direct `F.sdpa` | MEDIUM |
| `PixArt-alpha/PixArt-XL-2-1024-MS` | 1.8M | OpenRAIL++ | 611M | 4,096 | PixArt-alpha cross-attn DiT | `AttnProcessor2_0` | MEDIUM |
| `HiDream-ai/HiDream-I1-Full` | 1.2M | MIT | — | 4,096 | HiDream DiT+LLM (2025) | `HiDreamAttnProcessor` → direct `F.sdpa` | MEDIUM-HIGH |
| `HiDream-ai/HiDream-I1-Fast` | 648K | MIT | — | 4,096 | HiDream distilled | Same | MEDIUM-HIGH |
| `zai-org/CogView4-6B` | 1.7M | Apache-2.0 | 6B | 4,096 | CogView4 (GLM-4 backbone) | `CogView4AttnProcessor` → direct `F.sdpa` | MEDIUM-HIGH |
| `fal/AuraFlow-v0.2` | 97.8K | Apache-2.0 | 6.8B | 4,096 | AuraFlow MMDiT (SD3-style) | `AuraFlowAttnProcessor2_0` → direct `F.sdpa` | MEDIUM-HIGH |
| `Alpha-VLLM/Lumina-Image-2.0` | 106K | Apache-2.0 | — | 4,096 | Lumina Next-DiT flow-match | `LuminaAttnProcessor2_0` → direct `F.sdpa` | MEDIUM |
| `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | 35.7K | other | — | 4,096 | HunyuanDiT dual T5+CLIP | `HunyuanDiTAttnProcessor2_0` | MEDIUM |

**Kernel targets for image DiTs:** K1 (PrediT, FLUX), K3 (SageAttn2), K5 (TeaCache), K7 (TempCache), K8 (SVDQuant W4A4 → FLUX via `nunchaku`), K10 (INT4 weight), K11 (FP8 activation), K12 (TaQ-DiT → SD3.5).

**SVDQuant checkpoints available:** `mit-han-lab/svdq-int4-flux.1-dev`, `mit-han-lab/svdq-fp4-flux.1-dev` (FP4 H100-specific via `pip install nunchaku`).

---

## Category 3 — World Models

Action-conditioned frame generation. Primary targets for K15 (NFD+ action speculation), K16 (frame delta compression), B1 (Ca2-VDM conditioning KV cache).

| Model ID | Downloads | License | Params | Seq Tokens | Framework | Injection | Verdict |
|----------|-----------|---------|--------|-----------|-----------|-----------|---------|
| ✅ `Etched/oasis-500m` | 7.7K | MIT | 500M | 144 spatial / 32 temporal (axial) | Pure PyTorch, direct `F.sdpa` | Subclass `SpatialAxialAttention`, `TemporalAxialAttention` | **IMPLEMENTED** — baseline + STA + SageAttn2 + TeaCache + PrediT + INT4 + all-fused |
| `MatrixTeam/TheMatrix` | — | — | 17B | ~61,200 (720p/65f) | HunyuanVideo-derived, flash_attn_varlen | Patch `attention()` in `attenion.py`; use `mode="torch"` branch | **CONDITIONAL** — 80 GB VRAM needed; apex + FA3 install; disable TeaCache during bench |
| ✅ `nvidia/Cosmos-1.0-Diffusion-7B-Video2World` | 853K | NVIDIA (gated) | 7B | ~56,320 (720p/121f) | diffusers port exists | `set_attention_backend()` via diffusers Cosmos port | **IMPLEMENTED** — baseline + STA + SageAttn2 + TeaCache + PrediT |
| `nvidia/Cosmos-1.0-Autoregressive-4B` | — | NVIDIA (gated) | 4B | ~16,384 (discrete tokens) | NeMo/Megatron + TransformerEngine | Patch `transformer_engine_extensions.fused_attn_fwd` via pybind11 monkeypatch; TE varlen `[total_tokens, 3, H, D]` layout | **KEEP** — injectable via pybind11 symbol replacement without disabling TE baseline |
| `nvidia/Cosmos-1.0-Autoregressive-12B` | — | NVIDIA (gated) | 12B | ~16,384 | NeMo/Megatron + TransformerEngine | Same pybind11 patch as 4B | **KEEP** — 12B scale; stronger benchmark for INT8/FP8 kernel impact |
| `nvidia/Cosmos-Predict2.5-2B` | 619K | NVIDIA (gated) | ~2B | ~56,320 | Cosmos custom + FA3 | Patch `cosmos_predict2/module/attention.py`; **disable NATTEN** | **CONDITIONAL** — must disable NATTEN to measure speedup; competing against FA3 baseline |

**Injection note for Cosmos Autoregressive:** TE dispatches through a pybind11-exposed Python function `transformer_engine_extensions.fused_attn_fwd`. Since pybind11 module attributes are mutable Python objects, this can be replaced without disabling TE or using `mode="torch"`. The baseline therefore runs at full TE/cuDNN speed — a fair comparison. Kernel must handle TE's varlen QKV layout `[total_tokens, 3, n_heads, head_dim]` with `cu_seqlens` offsets. Alternative: `LD_PRELOAD` to intercept `nvte_fused_attn_fwd` at the shared-library symbol level (no Python required).

**Note on removed world models:**
- `eloialonso/diamond` — REMOVED: UNet, Atari 84×84, <441 tokens, conv-dominated, H100 speedup < 1%
- `1x-technologies/worldmodel_unet_v0` — REMOVED: 2D image UNet (no temporal), deprecated v0, no action conditioning
- `maitrix-org/Pandora` — REMOVED: weights not publicly available (data license issue)

---

## Category 4 — Large Multimodal Models (ViT Encoder Injection)

High-download production models. Inject custom attention into the vision encoder (`qwen2_5_vl`, `phi3_v`, `internvl_chat` etc.). Input: images or video frames. Kernel targets: K3 (SageAttn2 FP8), K10 (INT4 weight), K11 (FP8 activation).

| Model ID | Downloads | License | Params | Vision Encoder | Encoder Tokens | Injection |
|----------|-----------|---------|--------|---------------|----------------|-----------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | 60.7M | Apache-2.0 | 8.3B | qwen2_5_vl native ViT | dynamic (up to 4096) | `attn_implementation="sdpa"` + `ALL_ATTENTION_FUNCTIONS["sdpa"]` patch |
| `microsoft/Phi-3.5-vision-instruct` | 13.5M | MIT | 4.2B | phi3_v CLIP-style ViT | ~1024 (336px) | Register hook on `Phi3VImageEncoder` attention layers |
| `microsoft/Florence-2-large` | 20.4M | MIT | 777M | Florence2 ViT unified | 4096 (1024px input) | `attn_implementation="sdpa"` + hook |
| `microsoft/phi-4-multimodal-instruct` | 6.4M | MIT | 5.6B | phi4mm ViT+audio | ~1024 | Same as Phi-3.5 |
| `OpenGVLab/InternVL2_5-8B` | 1.3M | MIT | 8.1B | InternViT-300M-448px | 1024 (448px) | `set_attn_processor()` on InternViT attention |

---

## Category 5 — High-Resolution ViTs (Quantization / Weight-Loading Benchmarks)

At standard 224px, ViT attention is <1% of compute — useless for attention kernels. These models are kept **only for K10/K11 (INT4/FP8 weight quantization) benchmarks** where the bottleneck is weight memory bandwidth, not attention.

| Model ID | Downloads | License | Params | Use Case | H100 Kernel |
|----------|-----------|---------|--------|---------|-------------|
| `BAAI/EVA-CLIP-18B` | 23.3K | Apache-2.0 | 18B | INT4 weight quant baseline (largest ViT) | K10 INT4-wo |
| `OpenGVLab/InternViT-6B-448px-V2_5` | 23.7K | MIT | 5.5B | INT4 weight quant | K10 INT4-wo |
| `facebook/sam-vit-huge` | 9.0M | Apache-2.0 | 641M | 4 global attn layers at 4096 tokens | K3 (global layers only) |

**Why standard ViTs are excluded:** ViT-B/16 at 224px → 197 tokens (attention is 0.6% of compute). ViT-L/14 at 224px → 257 tokens. Below the 512-token H100 attention speedup threshold. Kept out entirely.

---

## Injection API Reference

### Universal injection for all diffusers dispatch-based models (FLUX, Wan, LTX, HunyuanVideo, SkyReels)

```python
import os
os.environ["DIFFUSERS_ATTN_BACKEND"] = "_sage_qk_int8_pv_fp8_cuda_sm90"  # H100 SageAttn2

# Or per-inference:
from diffusers import attention_backend
with attention_backend("_flash_3_hub"):       # FlashAttention-3
    output = pipeline(prompt, ...)

# Or per-layer:
from diffusers.models.attention import AttentionModuleMixin
for m in pipeline.transformer.modules():
    if isinstance(m, AttentionModuleMixin):
        m.set_attention_backend("_sage_qk_int8_pv_fp8_cuda_sm90")
```

### Custom AttnProcessor (for CogVideoX, SD3, PixArt, AuraFlow — old-style direct F.sdpa)

```python
class CustomKernelAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states or hidden_states)
        v = attn.to_v(encoder_hidden_states or hidden_states)
        q = attn.head_to_batch_dim(q)   # (B, heads, S, head_dim)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)
        out = your_custom_cuda_kernel(q, k, v, mask=attention_mask)
        out = attn.batch_to_head_dim(out)
        return attn.to_out[0](out)

pipeline.transformer.set_attn_processor(CustomKernelAttnProcessor())
```

### TeaCache (step caching) via diffusers TaylorSeer (native analog)

```python
from diffusers import TaylorSeerCacheConfig
pipeline.transformer.enable_cache(
    TaylorSeerCacheConfig(cache_interval=5, max_order=1, disable_cache_before_step=3)
)
# Supported: FLUX, CogVideoX, Wan, LTX, HunyuanVideo, Mochi
```

### SVDQuant / INT4 for FLUX (nunchaku)

```python
from nunchaku import NunchakuFluxTransformer2dModel
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-dev", torch_dtype=torch.bfloat16
)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16
)
```

---

## Priority for H100 Implementation

| Priority | Model | Tokens | Why |
|----------|-------|--------|-----|
| P0 | `tencent/HunyuanVideo` | 115,200 | >80% attention-bound; highest absolute speedup |
| P0 | `black-forest-labs/FLUX.1-dev` | 16,384 | 29.8M downloads; K1 PrediT benchmark model; SVDQuant target |
| P0 | `Lightricks/LTX-Video` | ~14,400 | 5.5M downloads; Apache-2.0; cleanest diffusers DiT |
| P0 | `zai-org/CogVideoX-5b` | 17,550 | 51% attention; paper-confirmed token count; Apache-2.0 |
| P0 | `Wan-AI/Wan2.1-I2V-14B-480P` | ~20,160 | 1.2M downloads; Apache-2.0; 14B |
| P0 | `Etched/oasis-500m` | 144/32 axial | Only open game world model; K15/K16/B1 exclusive |
| P1 | `rhymes-ai/Allegro` | 79,200 | Paper-confirmed 79.2K tokens; Apache-2.0 |
| P1 | `hpcai-tech/OpenSora-STDiT-v3` | 60-80K | 1.4M downloads; Apache-2.0; explicit STDiT3 class |
| P1 | `stabilityai/stable-diffusion-3.5-large` | 4,096 | 8B params; K8 SVDQuant + K12 TaQ-DiT target |
| P1 | `Wan-AI/Wan2.1-T2V-14B` | ~27,000 | Apache-2.0; 14B; dispatch_attention_fn |
| P2 | `zai-org/CogView4-6B` | 4,096 | Apache-2.0; 1.7M downloads; 2025 model |
| P2 | `HiDream-ai/HiDream-I1-Full` | 4,096 | MIT; 1.2M downloads; 2025 DiT+LLM hybrid |
| P2 | `MatrixTeam/TheMatrix` | ~61,200 | 17B game world model; strongest compute-bound test |
| P2 | `Qwen/Qwen2.5-VL-7B-Instruct` | dynamic | 60.7M downloads; Apache-2.0; LMM ViT injection |
| P3 | `nvidia/Cosmos-1.0-Diffusion-7B-Video2World` | ~56,320 | Gated; use diffusers port |
| P3 | `genmo/mochi-1-preview` | ~18,000 | Apache-2.0; asymmetric DiT arch variant |
| P3 | `LanguageBind/Open-Sora-Plan-v1.3.0` | ~13-40K | MIT; Skiparse sparse attention baseline |

---

## Models Removed and Reasons

| Model | Reason |
|-------|--------|
| `eloialonso/diamond` | UNet, Atari 84×84, <441 tokens, conv-dominated, H100 speedup <1% |
| `1x-technologies/worldmodel_unet_v0` | 2D image UNet, no temporal axis, no action conditioning, archived v0 |
| `maitrix-org/Pandora` | Weights hidden due to data license; not downloadable |
| ~~`nvidia/Cosmos-1.0-Autoregressive-*`~~ | Restored to Category 3 — injectable via `transformer_engine_extensions.fused_attn_fwd` pybind11 monkeypatch |
| `Efficient-Large-Model/Sana_*` | Linear attention (ReLU kernel trick), incompatible with FlashAttention-style CUDA kernels |
| `facebook/DiT-XL-2-256` | Only 256 tokens at native 256px resolution; negligible H100 speedup |
| `openai/clip-vit-large-patch14` | 257 tokens at 224px; attention <1% of compute |
| `google/vit-base-patch16-224` | 197 tokens; attention negligible |
| `google/vit-large-patch16-224` | Same |
| `google/vit-huge-patch14-224-in21k` | Same |
| `facebook/deit-base-distilled-patch16-224` | 197 tokens |
| `microsoft/beit-large-patch16-224-pt22k-ft22k` | Small ViT at 224px |
| `facebook/vit-mae-huge` | Small ViT at 224px |
| `google/siglip-so400m-patch14-384` | 729 tokens, below 1K meaningful threshold |
| `BAAI/EVA-CLIP-18B` / `EVA-CLIP-8B` | Classification ViT; kept only for K10 weight-quant in Cat 5 |
| `laion/CLIP-ViT-bigG-14` | 257 tokens; used as frozen encoder, not injection target |
| `timm/eva02_enormous_patch14*` | 256 tokens at 224px; no generation task |
| `apple/DFN5B-CLIP-ViT-H-14` | open_clip (not transformers), complex injection, frozen encoder |
| `facebook/dinov2-giant` | 257/1370 tokens; weight-quant baseline only, not generation |
| `MCG-NJU/videomae-huge-finetuned-kinetics` | Video classification, not generation, no denoising loop |
| `OpenGVLab/InternVideo2-Chat-8B` | Gated; video understanding LLM, not generation |
| `stabilityai/stable-video-diffusion-*` | UNet temporal — not DiT; attention kernel gain minimal |
| `ByteDance/AnimateDiff-Lightning` | UNet motion adapter; 63.2M downloads but architecture incompatible |
| `wangfuyun/AnimateLCM` | UNet motion adapter |
| `ali-vilab/text-to-video-ms-1.7b` | UNet-3D; 256×256/16fr; seq too short |
| `cerspense/zeroscope_v2_576w` | UNet-3D; small resolution |
| `ali-vilab/i2vgen-xl` | UNet hierarchical; not DiT |
| `showlab/show-1-base` | Hybrid pixel+latent; not pure DiT |
| `TIGER-Lab/ConsistI2V` | Research paper model; not primary DiT target |
| `TencentARC/MotionCtrl` | Motion controller adapter, not standalone model |
| `kandinsky-community/kandinsky-3` | UNet+transformer hybrid; 2023 architecture |
| `warp-ai/wuerstchen` | 2023; not a modern DiT |
| `Shitao/OmniGen-v1` | Phi3 backbone for generation; niche architecture |
| `Kwai-Kolors/Kolors` | SDXL UNet-based |

---

## Access Notes

**Gated (free, requires HF account + license accept):**
`black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-Kontext-dev`, `stabilityai/sd3.5-*`, `Etched/oasis-500m`, `tencent/HunyuanVideo`, `nvidia/Cosmos-*`, `meta-llama/Llama-3.2-Vision`

**Fully open (Apache-2.0 / MIT):**
`Wan-AI/Wan2.x-*`, `zai-org/CogVideoX-2b`, `genmo/mochi-1-preview`, `hpcai-tech/Open-Sora*`, `rhymes-ai/Allegro`, `LanguageBind/Open-Sora-Plan*`, `HiDream-ai/*`, `fal/AuraFlow*`, `Alpha-VLLM/Lumina-Image-2.0`, `ostris/OpenFLUX.1`, `PixArt-alpha/*`, `Qwen/Qwen2.5-VL-7B`, `microsoft/Phi-3.5-vision`, `microsoft/Florence-2-large`, `OpenGVLab/InternVL2_5-8B`

**Download:**
```bash
# Open model
huggingface-cli download Lightricks/LTX-Video --local-dir ./models/ltx-video

# Gated
HF_TOKEN=<token> huggingface-cli download tencent/HunyuanVideo --local-dir ./models/hunyuanvideo

# Cosmos (accept NVIDIA license first at nvidia.com/cosmos)
HF_TOKEN=<token> huggingface-cli download nvidia/Cosmos-Predict2.5-2B --local-dir ./models/cosmos-predict2-2b
```
