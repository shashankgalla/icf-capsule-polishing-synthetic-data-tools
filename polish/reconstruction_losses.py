"""
Reconstruction losses for PhysDiff-Polish — SOTA training improvements.

Two complementary improvements over the standard diffusion MSE loss:

1.  Min-SNR-γ weighting  (Hang et al., NeurIPS 2023)
    https://arxiv.org/abs/2303.09556
    ─────────────────────────────────────────────
    Standard diffusion loss gives equal weight to ALL timesteps, including
    very noisy ones (large t, small SNR) which contribute little signal but
    lots of gradient noise.  Min-SNR-γ clips the per-sample weight at γ:

        SNR(t)  = α_t / (1 − α_t)
        w_v(t)  = min(SNR(t), γ) / (SNR(t) + 1)   [v-prediction formula]

    Effect: the model focuses on high-SNR (low-noise) timesteps that
    matter most for image quality, without discarding the low-SNR range
    entirely.  γ=5 is the SOTA default.

2.  Differentiable latent SSIM loss
    ──────────────────────────────────
    Structural Similarity (SSIM) directly penalises luminance, contrast,
    and structural mismatches between the predicted x₀ latents and the
    ground-truth latents.  Minimising (1 − SSIM) in latent space directly
    improves the SSIM metric at evaluation time.

These two modules are combined in train.py with a latent L1 loss to give
the model three orthogonal reconstruction objectives from epoch 1:

    L_total = w(t) · L_diffusion(v_pred, v_target)   # MinSNR-weighted
            + λ_recon · L1(x0_pred, x0_gt)            # pixel-level PSNR
            + λ_ssim  · SSIM_loss(x0_pred, x0_gt)     # structural quality
            + [physics losses after warmup]
"""

import torch
import torch.nn.functional as F


# ── Gaussian kernel (cached per device) ─────────────────────────────────────

_kernel_cache: dict = {}


def _get_gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Return a [1, 1, ws, ws] Gaussian kernel, cached per device."""
    key = (window_size, sigma, str(device))
    if key not in _kernel_cache:
        x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        g = g / g.sum()
        k = g.outer(g).unsqueeze(0).unsqueeze(0)   # [1, 1, ws, ws]
        _kernel_cache[key] = k.to(device)
    return _kernel_cache[key]


# ── SSIM ────────────────────────────────────────────────────────────────────

def ssim_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> torch.Tensor:
    """
    Differentiable (1 − SSIM) loss on latent tensors.

    Handles [B, T, C, H, W] or [B, C, H, W].
    Values are per-channel min-max normalised before SSIM so the constants
    (C1, C2) are always appropriate regardless of latent scale.

    Returns:
        scalar ∈ [0, 2]  (0 = identical, ≈1 = maximal structural mismatch)
    """
    if pred.dim() == 5:
        B, T, C, H, W = pred.shape
        pred   = pred.reshape(B * T, C, H, W)
        target = target.reshape(B * T, C, H, W)

    N, C, H, W = pred.shape
    # Treat each (sample, channel) pair independently → flatten to [N*C, 1, H, W]
    pred_f   = pred.reshape(N * C, 1, H, W)
    target_f = target.reshape(N * C, 1, H, W)

    # Per-channel normalisation to [0, 1] for stable C1, C2 constants
    with torch.no_grad():
        lo = torch.minimum(
            target_f.amin(dim=[2, 3], keepdim=True),
            pred_f.amin(dim=[2, 3], keepdim=True),
        )
        hi = torch.maximum(
            target_f.amax(dim=[2, 3], keepdim=True),
            pred_f.amax(dim=[2, 3], keepdim=True),
        )
        scale = (hi - lo).clamp(min=1e-6)

    pred_n   = (pred_f   - lo) / scale   # ∈ [0, 1]
    target_n = (target_f - lo) / scale

    # Standard SSIM constants for L = 1
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    kernel = _get_gaussian_kernel(window_size, sigma=1.5, device=pred.device)
    pad    = window_size // 2

    mu1 = F.conv2d(pred_n,   kernel, padding=pad)
    mu2 = F.conv2d(target_n, kernel, padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(pred_n.pow(2),         kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target_n.pow(2),       kernel, padding=pad) - mu2_sq
    sigma12   = F.conv2d(pred_n * target_n,     kernel, padding=pad) - mu12

    numerator   = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8

    return 1.0 - (numerator / denominator).mean()


# ── Min-SNR-γ weighting ──────────────────────────────────────────────────────

def minsnr_weight(
    timesteps:      torch.Tensor,
    alphas_cumprod: torch.Tensor,
    gamma:          float = 5.0,
) -> torch.Tensor:
    """
    Per-sample Min-SNR-γ loss weight for v-prediction training.

    Args:
        timesteps:      [B]  integer indices into the noise schedule
        alphas_cumprod: [T]  scheduler's cumulative alpha array (can be on CPU)
        gamma:          SNR clipping threshold (default 5.0 per the paper)

    Returns:
        weights: [B] float32 weights ∈ (0, ~1], detach before using as multiplier
    """
    alpha = alphas_cumprod.to(timesteps.device)[timesteps].float()  # [B]
    snr   = alpha / (1.0 - alpha + 1e-8)                            # [B]
    # v-prediction formula: w = min(SNR, γ) / (SNR + 1)
    w = torch.minimum(snr, snr.new_full([], gamma)) / (snr + 1.0)
    return w  # [B]
