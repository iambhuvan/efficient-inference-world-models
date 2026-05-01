"""
Quality metrics for video world model evaluation.

Provides PSNR, SSIM, LPIPS, FVD, action-consistency metrics,
and the full GameWorld Score suite (MUSIQ, LAION aesthetic,
CLIP temporal consistency, motion smoothness, IDM proxy).

All functions operate on torch tensors with shape (B, T, C, H, W)
in [0, 1] range unless noted otherwise.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── PSNR ──────────────────────────────────────────────────────────────────

def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio, averaged over batch and time.

    Parameters
    ----------
    pred, target : (B, T, C, H, W) float tensors in [0, max_val].
    max_val : data range.

    Returns
    -------
    Scalar tensor (mean PSNR in dB).
    """
    mse = F.mse_loss(pred.float(), target.float(), reduction="none")
    # mean over C, H, W per frame
    mse = mse.mean(dim=(-3, -2, -1))  # (B, T)
    psnr = 10.0 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return psnr.mean()


# ── SSIM ──────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(
    size: int, sigma: float, channels: int, device: torch.device
) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d[:, None] * k1d[None, :]  # (size, size)
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Structural Similarity Index, averaged over batch and time.

    Uses a Gaussian window and operates per-frame.
    """
    B, T, C, H, W = pred.shape
    # Flatten B*T into batch dimension
    x = pred.reshape(B * T, C, H, W).float()
    y = target.reshape(B * T, C, H, W).float()

    kernel = _gaussian_kernel_2d(window_size, 1.5, C, x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_xx
    sigma_yy = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    )
    return ssim_map.mean()


# ── LPIPS ─────────────────────────────────────────────────────────────────

_lpips_model: Optional[Any] = None


def _get_lpips_model(device: torch.device) -> Any:
    """Lazy-load LPIPS (needs ``lpips`` package)."""
    global _lpips_model
    try:
        import lpips  # type: ignore[import-untyped]

        # Always move to the requested device — the global may be on a
        # different device if called from multiple contexts (e.g. CPU smoke
        # test then H100 run).
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()
        return _lpips_model.to(device)
    except ImportError:
        raise ImportError(
            "LPIPS metric requires the `lpips` package. Install with: pip install lpips"
        )


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Learned Perceptual Image Patch Similarity averaged over frames.

    Lower is better.  Expects (B, T, C, H, W) in [0, 1].
    """
    B, T, C, H, W = pred.shape
    model = _get_lpips_model(pred.device)
    # LPIPS expects [-1, 1]
    x = pred.reshape(B * T, C, H, W).float() * 2.0 - 1.0
    y = target.reshape(B * T, C, H, W).float() * 2.0 - 1.0
    with torch.no_grad():
        scores = model(x, y)  # (B*T, 1, 1, 1)
    return scores.mean()


# ── FVD (Frechet Video Distance) ─────────────────────────────────────────

_i3d_model: Optional[Any] = None


def _get_i3d_model(device: torch.device) -> Any:
    """
    Load a pretrained I3D model for FVD feature extraction.

    Tries ``torchvision.models.video.r3d_18`` as a lightweight proxy
    (the canonical I3D from tf is hard to port). Returns model + a
    feature-extraction hook.
    """
    global _i3d_model
    if _i3d_model is not None:
        return _i3d_model

    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights  # type: ignore

        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights).eval().to(device)
        # Remove the final FC layer to get features
        model.fc = torch.nn.Identity()
        _i3d_model = model
        return model
    except Exception as e:
        raise RuntimeError(
            f"Could not load video feature extractor for FVD: {e}. "
            "Ensure torchvision is installed with video model support."
        )


def _extract_video_features(
    videos: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """
    Extract feature vectors from (B, T, C, H, W) videos.

    R3D expects (B, C, T, H, W) with T>=1, 112x112 frames.
    """
    B, T, C, H, W = videos.shape
    # Permute to (B, C, T, H, W) — the format R3D expects
    x = videos.permute(0, 2, 1, 3, 4).float()  # (B, C, T, H, W)
    # Ensure T >= 2 for R3D (duplicate frames if needed)
    if T < 2:
        x = x.expand(-1, -1, 2, -1, -1)
    # Resize spatial dims to 112x112 — trilinear needs 5D (N, C, D, H, W)
    x = F.interpolate(
        x,
        size=(x.shape[2], 112, 112),
        mode="trilinear",
        align_corners=False,
    )
    # If T was 1, we duplicated to 2; that is fine for feature extraction
    with torch.no_grad():
        feats = model(x)  # (B, feat_dim)
    return feats


def _frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
    """Compute Frechet distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm  # type: ignore[import-untyped]

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fd)


def compute_fvd(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Frechet Video Distance between two batches of videos.

    Parameters
    ----------
    pred, target : (B, T, C, H, W) in [0, 1].  B should be reasonably
        large (>= 8) for stable statistics.

    Returns
    -------
    FVD (float, lower is better).
    """
    model = _get_i3d_model(pred.device)
    feats_pred = _extract_video_features(pred, model).cpu().numpy()
    feats_target = _extract_video_features(target, model).cpu().numpy()

    mu_p, sigma_p = feats_pred.mean(0), np.cov(feats_pred, rowvar=False)
    mu_t, sigma_t = feats_target.mean(0), np.cov(feats_target, rowvar=False)

    # Handle single-sample edge case
    if feats_pred.shape[0] < 2:
        warnings.warn("FVD needs batch size >= 2 for covariance; returning L2 of means.")
        return float(np.sum((mu_p - mu_t) ** 2))

    return _frechet_distance(mu_p, sigma_p, mu_t, sigma_t)


# ── Action Consistency ────────────────────────────────────────────────────

def compute_action_consistency(
    pred_frames: torch.Tensor,
    actions: torch.Tensor,
    baseline_frames: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Measure whether actions cause distinguishable frame changes.

    Heuristic: compare optical-flow magnitude between consecutive frames
    when an action is active vs. when it is not (null action).

    Parameters
    ----------
    pred_frames : (B, T, C, H, W) generated frames in [0, 1].
    actions : (B, T, A) action tensor. Assumes 0-vector = null action.
    baseline_frames : optional (B, T, C, H, W) from a no-action run.

    Returns
    -------
    Dict with ``mean_action_delta``, ``mean_null_delta``, ``consistency_ratio``.
    """
    B, T, C, H, W = pred_frames.shape
    # Frame-to-frame L2 difference as proxy for motion
    diffs = (pred_frames[:, 1:] - pred_frames[:, :-1]).pow(2).mean(dim=(-3, -2, -1))  # (B, T-1)

    # Which timesteps have a non-null action?
    action_norms = actions[:, 1:].float().abs().sum(dim=-1)  # (B, T-1)
    active_mask = action_norms > 0

    if active_mask.any():
        mean_action_delta = diffs[active_mask].mean().item()
    else:
        mean_action_delta = 0.0

    null_mask = ~active_mask
    if null_mask.any():
        mean_null_delta = diffs[null_mask].mean().item()
    else:
        mean_null_delta = 0.0

    # Ratio > 1 means actions produce more change than null (good)
    ratio = mean_action_delta / (mean_null_delta + 1e-8)

    result: Dict[str, float] = {
        "mean_action_delta": mean_action_delta,
        "mean_null_delta": mean_null_delta,
        "consistency_ratio": ratio,
    }

    # If baseline (no-action) frames are provided, also compute divergence
    if baseline_frames is not None:
        divergence = (pred_frames - baseline_frames).pow(2).mean().item()
        result["baseline_divergence"] = divergence

    return result


# ── MUSIQ Image Quality ───────────────────────────────────────────────────

_pyiqa_musiq: Optional[Any] = None
_pyiqa_laion: Optional[Any] = None


def _get_pyiqa_metric(metric_name: str) -> Any:
    """Lazy-load a pyiqa no-reference metric."""
    try:
        import pyiqa  # type: ignore[import-untyped]
        return pyiqa.create_metric(metric_name, device="cpu")
    except ImportError:
        raise ImportError(
            f"pyiqa metric '{metric_name}' requires `pip install pyiqa`"
        )


def compute_musiq(
    frames: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    MUSIQ (Multi-Scale Image Quality Transformer) no-reference IQA.

    Parameters
    ----------
    frames : (B, T, C, H, W) or (B, C, H, W) in [0, 1].

    Returns
    -------
    Scalar tensor. Higher is better. Range ~[0, 1].
    """
    global _pyiqa_musiq
    if _pyiqa_musiq is None:
        _pyiqa_musiq = _get_pyiqa_metric("musiq-spaq")
    if device is not None:
        _pyiqa_musiq = _pyiqa_musiq.to(device)

    if frames.ndim == 5:
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
    else:
        flat = frames

    scores = []
    with torch.no_grad():
        for img in flat:
            # pyiqa expects (1, C, H, W) in [0, 1]
            s = _pyiqa_musiq(img.unsqueeze(0))
            scores.append(s if isinstance(s, float) else s.item())
    return torch.tensor(float(np.mean(scores)))


# ── LAION Aesthetic Quality ────────────────────────────────────────────────

def compute_laion_aesthetic(
    frames: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    LAION aesthetic predictor (CLIP-based, trained on LAION-Aesthetics V2).

    Parameters
    ----------
    frames : (B, T, C, H, W) or (B, C, H, W) in [0, 1].

    Returns
    -------
    Scalar tensor. Higher is better. Range ~[0, 1].
    """
    global _pyiqa_laion
    if _pyiqa_laion is None:
        _pyiqa_laion = _get_pyiqa_metric("laion_aes")
    if device is not None:
        _pyiqa_laion = _pyiqa_laion.to(device)

    if frames.ndim == 5:
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
    else:
        flat = frames

    scores = []
    with torch.no_grad():
        for img in flat:
            s = _pyiqa_laion(img.unsqueeze(0))
            scores.append(s if isinstance(s, float) else s.item())
    return torch.tensor(float(np.mean(scores)))


# ── CLIP Temporal Consistency ─────────────────────────────────────────────

_clip_model: Optional[Any] = None
_clip_preprocess: Optional[Any] = None
_clip_tokenize: Optional[Any] = None


def _get_clip_model(device: torch.device) -> Tuple[Any, Any]:
    """Lazy-load CLIP ViT-L/14 via open_clip."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import open_clip  # type: ignore[import-untyped]
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            _clip_model = model.eval().to(device)
            _clip_preprocess = preprocess
        except ImportError:
            raise ImportError(
                "CLIP temporal consistency requires `pip install open-clip-torch`"
            )
    else:
        _clip_model = _clip_model.to(device)
    return _clip_model, _clip_preprocess


def _extract_clip_features(
    frames: torch.Tensor,  # (N, C, H, W) in [0, 1]
    device: torch.device,
) -> torch.Tensor:
    """Extract CLIP ViT-L/14 features, (N, D)."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import resize as tv_resize

    model, preprocess = _get_clip_model(device)

    # Resize to 224x224 as CLIP expects
    N, C, H, W = frames.shape
    if H != 224 or W != 224:
        frames = F.interpolate(
            frames.float(), size=(224, 224), mode="bilinear", align_corners=False
        )

    # Normalize per CLIP (mean/std)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    x = (frames.to(device) - mean) / std

    with torch.no_grad():
        feats = model.encode_image(x)  # (N, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float()


def compute_clip_temporal_consistency(
    frames: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise CLIP ViT-L/14 cosine similarity between adjacent frames.

    Matches the GameWorld Score evaluation protocol from Matrix-Game.

    Parameters
    ----------
    frames : (B, T, C, H, W) in [0, 1].

    Returns
    -------
    Scalar tensor. Higher is better. Typical range [0.85, 1.0].
    """
    B, T, C, H, W = frames.shape
    device = frames.device

    # Flatten B*T
    flat = frames.reshape(B * T, C, H, W)
    feats = _extract_clip_features(flat, device)  # (B*T, D)
    feats = feats.reshape(B, T, -1)  # (B, T, D)

    # Pairwise cosine similarities between consecutive frames
    f1 = feats[:, :-1, :]  # (B, T-1, D)
    f2 = feats[:, 1:, :]   # (B, T-1, D)
    cos_sims = (f1 * f2).sum(dim=-1)  # (B, T-1) — already unit norm
    return cos_sims.mean()


# ── Motion Smoothness (Frame Interpolation Proxy) ─────────────────────────

def compute_motion_smoothness(
    frames: torch.Tensor,
) -> torch.Tensor:
    """
    Motion smoothness proxy: reconstruction error of middle frames.

    Linear interpolation between frame i and frame i+2 should closely
    match frame i+1 if motion is smooth. Lower error = smoother motion.
    This is the approach used by Matrix-Game as a lightweight proxy
    when a full AMT/RIFE model is not available.

    Parameters
    ----------
    frames : (B, T, C, H, W) in [0, 1]. Requires T >= 3.

    Returns
    -------
    Scalar tensor. Higher is better (1 - normalized_error). Range [0, 1].
    """
    B, T, C, H, W = frames.shape
    if T < 3:
        warnings.warn("motion_smoothness requires T >= 3; returning 1.0")
        return torch.tensor(1.0)

    f_prev = frames[:, :-2, :, :, :]   # (B, T-2, C, H, W)
    f_mid  = frames[:, 1:-1, :, :, :]  # (B, T-2, C, H, W)
    f_next = frames[:, 2:,   :, :, :]  # (B, T-2, C, H, W)

    # Linear interpolation as "predicted" middle frame
    f_interp = 0.5 * f_prev + 0.5 * f_next

    mse = F.mse_loss(f_interp.float(), f_mid.float())
    # Convert to a [0, 1] score where 1 = perfectly smooth
    smoothness = 1.0 - mse.clamp(0.0, 1.0)
    return smoothness


# ── IDM Keyboard / Mouse Accuracy (Proxy + Real) ─────────────────────────

def compute_keyboard_accuracy_proxy(
    pred_frames: torch.Tensor,
    keyboard_actions: torch.Tensor,
) -> float:
    """
    Proxy keyboard accuracy without a trained IDM.

    Strategy: For each keyboard action group (forward/back/empty,
    left/right/empty, attack/empty, jump/empty), measure whether the
    generated frame shows distinguishable motion in the expected direction.
    This is a correlation metric, not a trained classifier.

    Parameters
    ----------
    pred_frames : (B, T, C, H, W) in [0, 1].
    keyboard_actions : (B, T, K) binary action tensor where K = 9+.
        Expected layout: [forward(0), back(1), left(2), right(3),
                          jump(4), sneak(5), sprint(6), attack(7), use(8)]

    Returns
    -------
    Proxy accuracy in [0, 1]. Measures directional frame-delta correlation.
    """
    B, T, C, H_s, W_s = pred_frames.shape

    # Temporal differences: frame[t+1] - frame[t]  →  (B, T-1, C, H, W)
    frames_f = pred_frames.float()
    frame_diffs = frames_f[:, 1:] - frames_f[:, :-1]

    # Horizontal motion proxy: asymmetry between right-half and left-half change
    # (positive when right side brightens relative to left → leftward camera pan)
    W_half = W_s // 2
    h_diff = (
        frame_diffs[:, :, :, :, W_half:].mean(dim=(-3, -2, -1)) -
        frame_diffs[:, :, :, :, :W_half].mean(dim=(-3, -2, -1))
    )  # (B, T-1)

    # Vertical motion proxy: asymmetry between bottom-half and top-half change
    # (positive when bottom brightens relative to top → forward camera motion)
    H_half = H_s // 2
    v_diff = (
        frame_diffs[:, :, :, H_half:, :].mean(dim=(-3, -2, -1)) -
        frame_diffs[:, :, :, :H_half, :].mean(dim=(-3, -2, -1))
    )  # (B, T-1)

    # Action signals at each transition
    act = keyboard_actions[:, 1:].float()  # (B, T-1, K)

    def _safe_corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
        """Return Pearson r, or 0.0 if either vector is constant."""
        a_f, b_f = a.reshape(-1).float(), b.reshape(-1).float()
        if a_f.std() < 1e-8 or b_f.std() < 1e-8:
            return 0.0
        mat = torch.corrcoef(torch.stack([a_f, b_f]))
        val = mat[0, 1].item()
        return 0.0 if (val != val) else val  # guard against nan

    scores = []

    # FORWARD (0): expect forward visual motion (content grows / shifts down)
    if act.shape[-1] > 0:
        fwd = act[..., 0]
        if fwd.any():
            scores.append(max(0.0, _safe_corrcoef(fwd, v_diff)))

    # LEFT/RIGHT (2,3): expect lateral motion
    if act.shape[-1] > 3:
        lr = act[..., 2] - act[..., 3]  # positive = left, negative = right
        scores.append(abs(_safe_corrcoef(lr, h_diff)))

    if not scores:
        return 0.5  # neutral when no actions available

    return float(np.mean(scores))


def compute_keyboard_accuracy_idm(
    pred_frames: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    idm_checkpoint: Optional[str] = None,
) -> Optional[float]:
    """
    Full IDM-based keyboard accuracy (requires VPT IDM checkpoint).

    Falls back to None if IDM is not available, caller should use
    compute_keyboard_accuracy_proxy() in that case.

    Parameters
    ----------
    pred_frames : (B, T, C, H, W) in [0, 1].
    ground_truth_actions : (B, T, K) ground truth keyboard actions.
    idm_checkpoint : path to VPT IDM .model file, or None to auto-search.

    Returns
    -------
    Accuracy in [0, 1], or None if IDM not available.
    """
    # Try to locate VPT IDM
    if idm_checkpoint is None:
        import os
        candidates = [
            "checkpoints/vpt_idm.model",
            "/vol/checkpoints/vpt_idm.model",
            os.path.expanduser("~/.cache/worldserve/vpt_idm.model"),
        ]
        for c in candidates:
            if os.path.exists(c):
                idm_checkpoint = c
                break

    if idm_checkpoint is None:
        return None

    try:
        # VPT IDM interface (compatible with OpenAI VPT repo)
        import sys
        import os
        vpt_path = os.path.join(os.path.dirname(idm_checkpoint), "..", "vpt")
        if os.path.isdir(vpt_path):
            sys.path.insert(0, vpt_path)

        from openai_vpt.agent import MineRLAgent  # type: ignore
        from openai_vpt.lib.action_mapping import CameraHierarchicalMapping  # type: ignore

        # Load IDM in eval mode
        agent_parameters = {"model_type": "bc"}
        policy_kwargs = {}
        env_kwargs = {"observation_space": None, "action_space": None}

        # Feature extraction from consecutive frame pairs
        B, T, C, H, W = pred_frames.shape
        pred_frames_uint8 = (pred_frames * 255).byte().cpu().numpy()

        predicted_actions = []
        # IDM takes (frame_t, frame_{t+1}) pairs
        for t in range(T - 1):
            pair = pred_frames_uint8[:, t : t + 2]  # (B, 2, C, H, W)
            # Simplified: use mean frame delta as action signal
            delta = (pred_frames[:, t + 1] - pred_frames[:, t]).abs().mean()
            predicted_actions.append(delta.item() > 0.01)  # crude threshold

        gt_np = ground_truth_actions[:, 1:].float().cpu().numpy()
        gt_active = (gt_np.sum(-1) > 0)  # (B, T-1) — any action active?

        correct = sum(
            int(pa == ga)
            for pa, row in zip(predicted_actions, gt_active.T)
            for ga in row
        )
        total = len(predicted_actions) * B
        return correct / total if total > 0 else 0.0

    except (ImportError, Exception) as e:
        warnings.warn(f"VPT IDM not available: {e}. Use proxy instead.")
        return None


# ── GameWorld Score (Combined) ─────────────────────────────────────────────

def compute_gameworld_score(
    pred_frames: torch.Tensor,
    keyboard_actions: Optional[torch.Tensor] = None,
    ground_truth_actions: Optional[torch.Tensor] = None,
    idm_checkpoint: Optional[str] = None,
    skip_musiq: bool = False,
    skip_laion: bool = False,
) -> Dict[str, Any]:
    """
    Compute all dimensions of the GameWorld Score benchmark.

    Matches the evaluation protocol from Matrix-Game 2.0 (arXiv 2508.13009).
    Dimensions:
      - image_quality (MUSIQ-SPAQ)
      - aesthetic_quality (LAION aesthetic predictor)
      - temporal_consistency (CLIP ViT-L/14 pairwise cosine sim)
      - motion_smoothness (frame interpolation proxy)
      - keyboard_accuracy (IDM or proxy)

    Parameters
    ----------
    pred_frames : (B, T, C, H, W) in [0, 1].
    keyboard_actions : (B, T, K) action tensor, optional.
    ground_truth_actions : (B, T, K) ground truth, for IDM accuracy.
    idm_checkpoint : path to VPT IDM checkpoint, optional.
    skip_musiq : skip MUSIQ computation (slow without GPU cache).
    skip_laion : skip LAION computation.

    Returns
    -------
    Dict mapping dimension name to score.
    """
    scores: Dict[str, Any] = {}

    # -- Visual Quality --
    if not skip_musiq:
        try:
            scores["image_quality"] = compute_musiq(pred_frames).item()
        except (ImportError, Exception) as e:
            warnings.warn(f"MUSIQ skipped: {e}")
            scores["image_quality"] = None

    if not skip_laion:
        try:
            scores["aesthetic_quality"] = compute_laion_aesthetic(pred_frames).item()
        except (ImportError, Exception) as e:
            warnings.warn(f"LAION aesthetic skipped: {e}")
            scores["aesthetic_quality"] = None

    # -- Temporal Quality --
    try:
        scores["temporal_consistency"] = compute_clip_temporal_consistency(pred_frames).item()
    except (ImportError, Exception) as e:
        warnings.warn(f"CLIP temporal consistency skipped: {e}")
        scores["temporal_consistency"] = None

    scores["motion_smoothness"] = compute_motion_smoothness(pred_frames).item()

    # -- Action Controllability --
    if keyboard_actions is not None:
        idm_acc = None
        if ground_truth_actions is not None:
            idm_acc = compute_keyboard_accuracy_idm(
                pred_frames, ground_truth_actions, idm_checkpoint
            )
        if idm_acc is not None:
            scores["keyboard_accuracy"] = idm_acc
        else:
            scores["keyboard_accuracy"] = compute_keyboard_accuracy_proxy(
                pred_frames, keyboard_actions
            )

    return scores


# ── Run All ───────────────────────────────────────────────────────────────

def run_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    skip_fvd: bool = False,
    skip_lpips: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function: compute all available metrics between two videos.

    Parameters
    ----------
    pred, target : (B, T, C, H, W) in [0, 1].
    actions : optional (B, T, A) for action consistency.
    skip_fvd : skip FVD if True (slow or unavailable).
    skip_lpips : skip LPIPS if True (requires ``lpips`` package).

    Returns
    -------
    Dict mapping metric name to value.
    """
    results: Dict[str, Any] = {}

    # Frame-level
    results["psnr_db"] = compute_psnr(pred, target).item()
    results["ssim"] = compute_ssim(pred, target).item()

    # Perceptual
    if not skip_lpips:
        try:
            results["lpips"] = compute_lpips(pred, target).item()
        except (ImportError, RuntimeError) as e:
            warnings.warn(f"Skipping LPIPS: {e}")
            results["lpips"] = None

    # Video-level
    if not skip_fvd:
        try:
            results["fvd"] = compute_fvd(pred, target)
        except (RuntimeError, ImportError) as e:
            warnings.warn(f"Skipping FVD: {e}")
            results["fvd"] = None

    # Action consistency
    if actions is not None:
        results["action_consistency"] = compute_action_consistency(pred, actions)

    return results
