# World Models — Full Research Survey
## WorldServe / CMU 15-849 ML Systems Spring 2026

Comprehensive catalog of every world model found during our deep research pass (11 parallel agents across HuggingFace Hub, arXiv, GitHub source code, and H100 benchmarking literature, April 28 2026).

**Definition used:** A world model takes past observations (video frames) + optional actions → predicts future frames. Distinguished from pure video generation by having an explicit action/control signal or autoregressive frame loop.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented — baseline + individual kernel scripts + Modal H100 |
| 🟡 | Keep — injectable, not yet implemented |
| 🔶 | Conditional — injectable with caveats |
| ❌ | Removed — not injectable or weights unavailable |
| 🔒 | Closed — paper exists, no public weights |

---

## Part 1 — Models on HuggingFace (Verified During Research)

---

### ✅ Oasis-500M
**HuggingFace:** `Etched/oasis-500m`
**Paper:** "Oasis: A Universe in a Transformer" (Etched / Decart AI, Oct 2024)
**License:** MIT | **Params:** 500M | **Tokens:** 144 spatial + 32 temporal (axial attention)

#### What it is
The first open real-time interactive world model for Minecraft. Given a sequence of past frames and a keyboard/mouse action, it generates the next frame autoregressively — creating a fully playable Minecraft simulation entirely inside a neural network.

#### Architecture
- **Tokenizer:** Continuous VAE encodes 360×640 frames to 16×9 latent grid (144 spatial tokens per frame)
- **Backbone:** DiT-S/2 variant with **axial attention** — two separate attention passes per block:
  - `SpatialAxialAttention`: attends across the 144 spatial tokens of a single frame
  - `TemporalAxialAttention`: attends across the 32-frame history for each spatial position
- **Conditioning:** Discrete Minecraft action (18 possible actions: WASD, jump, attack, camera deltas) embedded and added to timestep embedding
- **Denoising:** 10-step DDIM over continuous latent space (not autoregressive token prediction — continuous diffusion per frame)
- **Why axial:** Full 3D attention over 144×32=4,608 tokens is expensive. Axial factorizes it into 144 spatial + 32 temporal, cutting complexity from O((144·32)²) to O(144² + 32²).

#### Injection points
Both `SpatialAxialAttention` and `TemporalAxialAttention` call `F.scaled_dot_product_attention` directly — replaceable by subclassing or monkeypatching. No diffusers dispatch layer.

#### Implemented kernels
STA (1D), SageAttn2, TeaCache, PrediT AB-2, INT4 weight quant, all-fused variant.

---

### ✅ Cosmos-1.0-Diffusion-7B-Video2World
**HuggingFace:** `nvidia/Cosmos-1.0-Diffusion-7B-Video2World`
**Paper:** "Cosmos World Foundation Model Platform for Physical AI" (NVIDIA, Jan 2025)
**License:** NVIDIA (gated, free) | **Params:** 7B | **Tokens:** ~56,320 (720p/121f)

#### What it is
NVIDIA's Video-to-World model: given a short conditioning video clip + text prompt, generates a physically consistent continuation. Designed for robotics and autonomous driving simulation ("Physical AI"). Not game-specific — generalizes across real-world scenes.

#### Architecture
- **Tokenizer:** Cosmos Continuous Video Tokenizer — 8× spatial + 4× temporal compression → `(T//4, H//8, W//8)` latent grid
- **Backbone:** Full 3D attention DiT, 7B parameters. All transformer blocks use 3D self-attention over the full `T×H×W` latent volume
- **Conditioning:** Text (T5-XXL encoder) cross-attention + video conditioning via concatenation in the latent space
- **Training:** Flow matching (continuous normalizing flow, not DDPM)
- **Diffusion type:** Continuous latent diffusion, 50-step default

#### Why a world model (not just T2V)
It takes an existing video as the conditioning signal (not just text), predicting what physically happens next — making it a forward model over physical state rather than a pure generative model.

#### Injection
Diffusers has a full port — uses `set_attention_backend()` / `DIFFUSERS_ATTN_BACKEND` env var identical to FLUX and HunyuanVideo. Clean injection, no source patching needed.

#### Implemented kernels
STA (3D, T=31 H=60 W=80 window=8), SageAttn2, TeaCache, PrediT AB-2.

---

### 🟡 Cosmos-1.0-Autoregressive-4B / 12B
**HuggingFace:** `nvidia/Cosmos-1.0-Autoregressive-4B`, `nvidia/Cosmos-1.0-Autoregressive-12B`
**Paper:** Same Cosmos platform paper (NVIDIA, Jan 2025)
**License:** NVIDIA (gated, free) | **Params:** 4B / 12B | **Tokens:** ~16,384 discrete

#### What it is
The autoregressive sibling to the diffusion model. Instead of continuous denoising, it tokenizes each video frame into discrete visual tokens (via Cosmos Discrete Tokenizer, VQ-based) and predicts them autoregressively — like a language model generating image patches.

#### Architecture
- **Tokenizer:** Cosmos Discrete Video Tokenizer — VQ-VAE with 64K codebook, spatial 8× compression. Each frame → 16×9 = 144 discrete integer tokens at 720p
- **Backbone:** Transformer decoder (GPT-style autoregressive), 4B or 12B parameters, causal self-attention
- **Framework:** NeMo + Megatron (NVIDIA's large-scale training stack) + **TransformerEngine** for all compute
- **Conditioning:** Text tokens prepended to the context window; action tokens injected at frame boundaries
- **Generation:** Left-to-right token prediction, one token per forward pass (or speculative decoding)

#### The TransformerEngine injection challenge
TE replaces all attention with `te.DotProductAttention` — a C++/CUDA op. However, TE's Python wrapper dispatches through `transformer_engine_extensions.fused_attn_fwd`, a pybind11-exposed function whose module attribute is mutable. Replacing it injects your kernel while keeping the cuDNN baseline intact.

#### Why kept (not removed)
At 4B–12B scale with autoregressive token generation, the attention fraction is very high (each token attends to all previous tokens — no step-batching amortization). SageAttn2 and STA on the attention heads would show strong speedup without quality loss.

---

### 🔶 Cosmos-Predict2.5-2B
**HuggingFace:** `nvidia/Cosmos-Predict2.5-2B`
**License:** NVIDIA (gated) | **Params:** ~2B | **Tokens:** ~56,320

#### What it is
Cosmos's next-gen prediction model — a 2.5B continuous diffusion DiT for video world prediction, successor to Cosmos-1.0.

#### Architecture
- Similar DiT structure to Cosmos-1.0-Diffusion but smaller (2B vs 7B)
- Key difference: uses **NATTEN (Neighborhood Attention)** for local sparse attention — already an optimized kernel
- Also uses FlashAttention-3 as the default backend

#### Why conditional
Must disable NATTEN to inject our STA kernel — then we're measuring speedup against vanilla SDPA (our kernel wins trivially). For a fair benchmark, we'd need to outperform NATTEN+FA3, which is a much higher bar. Kept in docs but deferred.

---

### 🔶 TheMatrix
**HuggingFace:** `MatrixTeam/TheMatrix`
**Paper:** "TheMatrix: Infinite-Horizon World Simulation with Global Coherence" (2025)
**License:** — | **Params:** 17B | **Tokens:** ~61,200 (720p/65f)

#### What it is
A 17B game world model specifically designed for open-world game simulation at cinematic quality. Generates consistent long-horizon video (>10 seconds) with action conditioning. Architecture derived from HunyuanVideo.

#### Architecture
- **Backbone:** HunyuanVideo-derived 3D full-attention DiT, 17B parameters
- **Attention:** `flash_attn_varlen_func` with variable-length sequences — packed format
- **Conditioning:** Game action tokens (position, orientation, interaction type) injected via cross-attention layers at each transformer block
- **Temporal coherence:** Uses a sliding window memory bank that preserves key-value pairs from previous segments, enabling coherent generation beyond the context window
- **Scale:** 17B parameters — largest world model with public weights

#### Injection approach
The Python `attention()` function in `attenion.py` (typo in source) has a `mode` parameter. When `mode="torch"` is set, it calls standard PyTorch attention instead of flash_attn_varlen — making it Python-injectable. Downside: this disables the varlen optimization, so baseline speed drops.

#### Status
CONDITIONAL — needs 80GB VRAM (tight for H100), requires apex + FA3 at install, and the `mode="torch"` fallback creates an unfair baseline.

---

### ❌ DIAMOND
**HuggingFace:** `eloialonso/diamond`
**Paper:** "Diffusion for World Modeling: Visual Details Matter in Atari" (DIAMOND, ICLR 2025)
**License:** Apache-2.0 | **Params:** ~70M | **Resolution:** 84×84 (Atari)

#### What it is
Diffusion-based world model for Atari games. Trains a UNet to denoise individual 84×84 game frames conditioned on past frames + action.

#### Architecture
- **Backbone:** Residual UNet (not a DiT) — convolutional encoder-decoder with skip connections
- **Attention:** 4 global self-attention layers at the bottleneck (8×8 feature map = 64 tokens)
- **Diffusion:** DDPM, 10-step generation per frame

#### Why removed
64 tokens at the bottleneck → attention is <1% of compute. Entire model is convolution-dominated. H100 tensor core speedup on 64-token attention rounds to zero. No DiT structure to inject into.

---

### ❌ Pandora
**HuggingFace:** `maitrix-org/Pandora`
**Paper:** "Pandora: Towards General World Model with Natural Language Actions and Video States" (2024)

#### What it is
A general world model that takes natural language action descriptions (not discrete actions) + video states → generates the next video segment. Attempts to unify game, robotics, and real-world simulation under one model.

#### Architecture
- **Backbone:** DiT-based continuous diffusion model
- **Action conditioning:** LLM-encoded natural language action descriptions cross-attended at each transformer block
- **Video state:** Past 4-8 frames encoded by a separate video encoder
- **Scale:** ~7B parameters

#### Why removed
Weights are hidden due to data licensing issues — the training data includes proprietary game footage. Model card exists on HuggingFace but no downloadable checkpoint. Unusable for benchmarking.

---

### ❌ 1x-Technologies WorldModel v0
**HuggingFace:** `1x-technologies/worldmodel_unet_v0`

#### What it is
A robot manipulation world model from 1X Technologies (humanoid robotics company). Predicts next visual observations from robot actions.

#### Architecture
- **Backbone:** 2D UNet (no temporal modeling)
- **Input:** Single RGB frame + action vector
- **Output:** Single predicted next frame

#### Why removed
2D UNet with no temporal axis, processes only one frame at a time (no sequence), deprecated v0 with no active development. Not compatible with any of our DiT/attention kernels.

---

## Part 2 — Models Found During Research: No Public Weights

These appeared in arXiv paper searches during our research phase. All are relevant world models but have no downloadable checkpoints on HuggingFace.

---

### 🔒 Genie (Google DeepMind, 2024)
**Paper:** "Genie: Generative Interactive Environments" (arXiv 2402.15391, Feb 2024)
**Authors:** Tim Bruce, Nelson Batchelor, et al. — Google DeepMind
**Params:** 11B | **Status:** No public weights

#### What it is
The model you were thinking of. Genie learns controllable world models from unlabeled internet video — no action labels required. It discovers latent action representations automatically from video alone, then uses them for interactive control.

#### Architecture (three-component system)
1. **Video Tokenizer:** VQ-VAE that encodes 16-frame clips at 160×90 → sequence of discrete spatial tokens (16×9 = 144 per frame, 16 frames = 2,304 tokens total)
2. **Latent Action Model (LAM):** Learns to infer a latent action code from consecutive frame pairs `(f_t, f_{t+1})` — the action is whatever caused the transition, discovered without labels
3. **Dynamics Model:** Spatial transformer that predicts the next frame token sequence given past frames + the latent action code. This is the world model proper — a causal transformer over flattened frame tokens

#### Why not available
Trained on 200,000+ hours of internet gameplay video using internal Google infrastructure. Weights kept proprietary. The paper explicitly states "we do not release model weights."

#### Relevance to our project
The LAM architecture (discovering action representations from video) is directly applicable to Oasis-style world models. The dynamics model is a causal spatial transformer — identical injection target to Oasis-500M's axial attention.

---

### 🔒 Genie 2 (Google DeepMind, 2024)
**Paper:** "Genie 2: A Large-Scale Foundation World Model" (blog post + technical report, Dec 2024)
**Authors:** Google DeepMind
**Params:** Not disclosed | **Status:** No public weights

#### What it is
The successor to Genie, extended to 3D consistent world generation. Generates physically consistent 3D environments from a single image prompt, with persistent object permanence (objects stay where placed even after camera moves away).

#### Architecture
- **Foundation:** Built on top of a large video generation model (believed to be based on Veo internals)
- **3D consistency:** Uses an implicit 3D scene representation (NeRF-like) learned jointly with the video model
- **Action space:** Mouse + keyboard (richer than Genie 1's latent actions)
- **Scale:** Significantly larger than Genie 1 (11B+)

#### Why not available
Fully internal at Google DeepMind. No code, no weights, no API access.

---

### 🔒 GameNGen (Google, 2024)
**Paper:** "Diffusion Models Are Real-Time Game Engines" (arXiv 2408.14837, Aug 2024)
**Authors:** Dani Valevski, Yaniv Leviathan, et al. — Google
**Params:** ~2B (based on SD 1.4) | **Status:** No public weights

#### What it is
Runs DOOM at 20fps entirely inside a neural network. The world model is a fine-tuned Stable Diffusion 1.4 UNet — standard convolution + cross-attention. A 2-phase training: first an RL agent plays DOOM and logs `(frames, actions)`, then the diffusion model is trained to predict next frames given the action-conditioned history.

#### Architecture
- **Backbone:** SD 1.4 UNet fine-tuned for video prediction (NOT a DiT — still convolutional)
- **Context:** Past 3 frames + current action concatenated to the noisy latent
- **Resolution:** 320×240 game frames → 40×30 latent (4× VAE compression)
- **Tokens:** 1,200 spatial tokens — just above our 512 threshold, but UNet architecture still limits kernel impact

#### Why not pursued
UNet architecture (not DiT) limits attention kernel applicability. No public weights. The paper's claim of "20fps" is with aggressive quality tradeoffs.

---

### 🔒 UniSim (OpenAI, 2023)
**Paper:** "Learning Interactive Real-World Simulators" (arXiv 2310.06114, Oct 2023)
**Authors:** Mengjiao Yang, Yilun Du, et al. — OpenAI / MIT
**Status:** No public weights, no HuggingFace repo

#### What it is
A unified world simulator that handles diverse environments: robotics manipulation, autonomous driving, and interactive scenes. Takes text or action descriptions → generates realistic video of the environment responding to those actions.

#### Architecture
- **Backbone:** Video diffusion model (transformer-based, architecture not fully disclosed in paper)
- **Conditioning:** Interleaved text + action tokens processed by a joint encoder
- **Scale:** Large — exact parameter count undisclosed
- **Training data:** Mix of robotics demonstrations, driving footage, and web video

#### Why not pursued
OpenAI internal project, no public release. Paper describes capabilities but provides no reproducible architecture details.

---

### 🔒 IRASim (Shanghai AI Lab / Tsinghua, 2024)
**Paper:** "IRASim: Learning Interactive Real-Robot Action Simulators" (arXiv 2406.14540, June 2024)
**Status:** No public weights

#### What it is
A world model specifically for robot manipulation — predicts future video frames given a robot arm trajectory (end-effector path) as the action signal.

#### Architecture
- **Backbone:** 3D DiT (Video DiT) — full spatiotemporal attention
- **Action conditioning:** Robot trajectory encoded as a sequence of 6-DOF end-effector poses, cross-attended at each transformer block
- **Resolution:** 256×256 workspace camera view
- **Tokens:** ~4,096 per frame window

#### Why relevant
The 3D DiT backbone is identical in structure to CogVideoX — same injection path. If weights become available, all our CogVideoX kernels transfer directly.

---

### 🔒 DreamerV3 (Google DeepMind, 2023)
**Paper:** "Mastering Diverse Domains through World Models" (arXiv 2301.04104, Jan 2023)
**Authors:** Danijar Hafner et al.
**Code:** github.com/danijar/dreamerv3 (JAX) | **Status:** Code open, no HF checkpoint

#### What it is
The state-of-the-art model-based RL world model. Learns compact world representations in a latent space using a Recurrent State Space Model (RSSM), then trains a policy entirely inside the imagined world without environment interaction.

#### Architecture
- **Backbone:** RSSM (Recurrent State Space Model) — NOT a transformer/DiT
  - Recurrent component: GRU over discrete latent states
  - Representation model: CNN encoder + categorical bottleneck
  - Decoder: CNN decoder for image reconstruction
- **No attention:** Entirely convolutional + recurrent — no transformer, no attention layers
- **Scale:** Small — designed to run on single GPU

#### Why not relevant
No transformer attention to inject into. RSSM architecture is fundamentally incompatible with our kernel suite (all kernels target attention or linear layers in DiTs).

---

### 🔒 GameGen-O (Comyco, 2024)
**Paper:** "GameGen-O: Open-World Video Game Generation with Interactive Control" (arXiv 2406.09750, June 2024)
**HuggingFace:** `Comyco/GameGen-O` (limited availability, not fully public)
**Params:** ~9B | **Status:** Partially available

#### What it is
Open-world game video generation model. Generates high-quality game videos (Minecraft, GTA-style environments) with keyboard + mouse action conditioning. Aims for cinematically realistic game world simulation.

#### Architecture
- **Backbone:** 3D Video DiT — full spatiotemporal attention over video tokens
- **Tokenizer:** Continuous video VAE (similar to CogVideoX's 3D VAE)
- **Action conditioning:** Keyboard + mouse delta encoded as action embeddings, injected via AdaLN conditioning at each transformer block
- **Training:** OGameData dataset — curated gameplay footage from multiple open-world games
- **Scale:** ~9B parameters, competitive with CogVideoX-5b

#### Why not implemented
Checkpoint availability is inconsistent — model card exists but weights require contacting authors. If weights become freely available, injection is identical to CogVideoX (same DiT family, same `set_attn_processor()` path).

---

## Part 3 — Architecture Comparison

| Model | Backbone | Attention Type | Tokens | Action Signal | Open Weights |
|-------|----------|---------------|--------|---------------|-------------|
| Oasis-500M | DiT-S/2 + Axial | Spatial (144) + Temporal (32) separated | 176 total | Discrete (18 Minecraft actions) | ✅ MIT |
| Cosmos-Diffusion-7B | 3D Full DiT | Full 3D spatiotemporal | ~56,320 | Text + video conditioning | ✅ NVIDIA gated |
| Cosmos-AR-4B/12B | GPT-style causal | Causal 1D (discrete tokens) | ~16,384 | Text + action tokens | ✅ NVIDIA gated |
| TheMatrix | HunyuanVideo DiT | 3D Full (varlen) | ~61,200 | Game action tokens | ✅ (17B, tight VRAM) |
| DIAMOND | UNet | Bottleneck only (64 tokens) | 64 | Discrete Atari actions | ✅ Apache-2.0 |
| Genie 1 | Spatial Transformer | Causal spatial | 2,304 | Latent (discovered) | 🔒 Google internal |
| Genie 2 | Video DiT (est.) | Full 3D | Large | Mouse + keyboard | 🔒 Google internal |
| GameNGen | UNet (SD 1.4) | Cross-attention only | 1,200 | Discrete DOOM actions | 🔒 Google internal |
| UniSim | Video DiT (est.) | Full 3D | Large | Text + action | 🔒 OpenAI internal |
| DreamerV3 | RSSM (GRU + CNN) | None | — | RL action space | Code only (JAX) |
| GameGen-O | 3D Video DiT | Full 3D | Large | Keyboard + mouse | Partial |
| IRASim | 3D Video DiT | Full 3D | ~4,096 | 6-DOF trajectory | 🔒 No checkpoint |
| Pandora | DiT | Full 3D | Large | Natural language | ❌ License issue |

---

## Part 4 — Key Architectural Patterns

### Pattern 1: Axial Attention (Oasis)
Factorize 3D attention into separate spatial and temporal passes. Reduces O((T·H·W)²) to O(T²·HW + (HW)²·T). The two attention modules are separate `nn.Module`s — both injectable independently.

### Pattern 2: Full 3D DiT (Cosmos, TheMatrix, GameGen-O)
Single attention module attends over the full T×H×W token volume. Highest attention fraction (>60% of compute at 56K+ tokens). Largest absolute speedup from STA and SageAttn2.

### Pattern 3: Causal Autoregressive (Cosmos-AR, Genie)
GPT-style causal attention over a flat sequence of discrete visual tokens. No spatial structure exploitable by STA (tokens are raster-ordered, not spatially aligned after tokenization). SageAttn2 and INT8/FP8 quantization still apply.

### Pattern 4: UNet Bottleneck Attention (DIAMOND, GameNGen)
Attention only at the spatial bottleneck of a convolutional UNet (64–1,200 tokens). Attention fraction negligible. Not a viable kernel injection target.

### Pattern 5: Diffusion + Action Cross-Attention (IRASim, Pandora)
Standard diffusion DiT with an additional cross-attention module per block attending to the encoded action sequence. Both the self-attention and cross-attention are injectable.

---

## Part 5 — Why Genie Specifically Matters for This Project

Genie's **Latent Action Model** (discovering what action caused `f_t → f_{t+1}` without labels) is the most novel contribution in world model research. Its dynamics model is architecturally identical to Oasis's temporal attention — a causal spatial transformer over flattened video tokens.

If Genie weights were public, it would be the ideal world model benchmark because:
1. Largest open-domain world model (11B) — highest token count
2. Spatial transformer backbone directly injectable with STA and SageAttn2
3. No game-specific bias — generalizes across any video domain

Since weights are not available, **Oasis-500M is the best open proxy** — same axial spatial transformer pattern, MIT license, and explicit DDIM loop for clean benchmarking.
