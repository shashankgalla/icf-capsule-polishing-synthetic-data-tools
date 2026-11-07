"""
Physics-informed loss functions for PhysDiff-Polish.

Three differentiable losses encoding polishing physics:
1. L_kinematic: Material removal rate proportional to shell velocity (Preston equation)
                Uses exact concentric V-groove kinematics from Lee et al. (2006)
2. L_monotonic: Surface roughness should decrease during polishing
3. L_dynamics: Higher RPM → faster visual dynamics

Reference:
    Lee, R.-T., Hwang, Y.-C., & Chiou, Y.-C. (2006).
    "Lapping of ultra-precision ball surfaces. Part I.
     Concentric V-groove lapping system."
    Int. J. Machine Tools & Manufacture, 46, 1146-1156.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianRoughness(nn.Module):
    """
    Compute surface roughness proxy via Laplacian variance.
    Higher variance = rougher surface texture.
    Differentiable through the generated frames.
    """

    def __init__(self):
        super().__init__()
        # Laplacian kernel (3x3)
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        self.register_buffer("kernel", kernel)

    def forward(self, frames):
        """
        Compute Laplacian variance for each frame.

        Args:
            frames: [B, T, C, H, W] or [T, C, H, W] - video frames in [0, 1]

        Returns:
            roughness: [B, T] or [T] - Laplacian variance per frame
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True

        B, T, C, H, W = frames.shape

        # Convert to grayscale: simple mean across channels
        gray = frames.mean(dim=2)  # [B, T, H, W]

        # Reshape for conv2d: [B*T, 1, H, W]
        gray_flat = gray.reshape(B * T, 1, H, W)

        # Apply Laplacian
        lap = F.conv2d(gray_flat, self.kernel, padding=1)  # [B*T, 1, H, W]

        # Variance of Laplacian per frame
        lap = lap.reshape(B * T, -1)
        roughness = lap.var(dim=1).reshape(B, T)

        if squeeze_batch:
            roughness = roughness.squeeze(0)

        return roughness


class ConcentricVGrooveKinematics(nn.Module):
    """
    Concentric V-groove lapping kinematics from Lee et al. (2006), using the
    no-slip-at-A/B, slip-only-at-C model:

        V_A = V_B = 0
        tan β  = R_b · sin θ / R_g                         (β fixed by geometry)
        ω_c    = ω_p · (1 + S_C · sin θ) / (1 + sin θ)     (ball circulation)
        ω_b    = R_g · ω_p · (1 − S_C) /
                 (R_b · cos β · (1 + sin θ))               (ball spin)
        V_C    = S_C · R_g · ω_p                            (sliding at head C)
        V_eff  = sqrt(ω_c² + ω_b² + V_C²)                  (RMS for dynamics loss)
    """

    def __init__(
        self,
        R_g: float = 12.5e-3,       # groove radius to upper-lap contact [m]
        R_b: float = 5.0e-3,        # ball radius [m]
        theta_deg: float = 60.0,     # V-groove half-angle [degrees]
        S_C: float = 0.95,           # default raw head-contact slip
    ):
        super().__init__()
        self.R_g = R_g
        self.R_b = R_b
        self.theta = math.radians(theta_deg)
        self.S_C = S_C

        # Pre-computed geometry (does NOT depend on slip).
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        self.R_i = R_g - R_b * self.cos_theta
        self.R_o = R_g + R_b * self.cos_theta
        self.beta_fixed = math.atan(R_b * self.sin_theta / R_g)
        self.cos_beta_fixed = math.cos(self.beta_fixed)

    def _resolve_sc(self, sc, reference):
        """Broadcast sc scalar/None to match the batch of `reference`."""
        if sc is None:
            return torch.full_like(reference, float(self.S_C))
        if torch.is_tensor(sc):
            return sc.to(reference)
        return torch.full_like(reference, float(sc))

    def forward(self, omega_p, sc=None):
        """
        Args:
            omega_p: [B] tensor — lower-lap angular speed (rad/s)
            sc:      [B] tensor or None — raw head-contact slip S_C

        Returns:
            V_eff:   [B]
            omega_c: [B]
            omega_b: [B]
        """
        sc_t = self._resolve_sc(sc, omega_p)

        R_g, R_b = self.R_g, self.R_b
        sin_theta = self.sin_theta

        omega_c = omega_p * (1.0 + sc_t * sin_theta) / (1.0 + sin_theta)
        omega_b = (
            R_g * omega_p * (1.0 - sc_t)
            / (R_b * self.cos_beta_fixed * (1.0 + sin_theta) + 1e-12)
        )
        V_C = sc_t * R_g * omega_p

        V_eff = torch.sqrt(omega_c ** 2 + omega_b ** 2 + V_C ** 2 + 1e-12)
        return V_eff, omega_c, omega_b

    def compute_v_eff(self, rpm_normalized, sc=None,
                      rpm_min=10.0, rpm_max=40.0):
        """
        Convenience: compute V_eff from normalized RPM + optional slip.
        """
        rpm = rpm_normalized * (rpm_max - rpm_min) + rpm_min
        omega_p = rpm * (2.0 * math.pi / 60.0)
        V_eff, _, _ = self.forward(omega_p, sc=sc)
        return V_eff

    def compute_ratio(self, rpm_normalized, sc=None,
                      rpm_min=10.0, rpm_max=40.0):
        """
        Kinematic velocity ratio ω_c / ω_b (diagnostic).

        Note: the L_ratio loss (LatentVelocityRatioLoss) uses
        R_T = ω_c / ω_p, not this quantity.
        """
        rpm = rpm_normalized * (rpm_max - rpm_min) + rpm_min
        omega_p = rpm * (2.0 * math.pi / 60.0)
        _, omega_c, omega_b = self.forward(omega_p, sc=sc)
        return omega_c / (omega_b + 1e-8)


# NOTE: PrestonKinematicLoss removed (v2). Preston equation (MRR = Kp·P·V)
# assumed a direct proportionality between V_eff and roughness change rate
# that we cannot validate without calibrated material removal data.
# The physics-informed dynamics loss (correlation with V_eff) and monotonic
# roughness loss remain as the physics priors.


class MonotonicRoughnessLoss(nn.Module):
    """
    Polishing should monotonically decrease surface roughness.
    Penalize any increase in roughness between consecutive frames.

    L_monotonic = sum(max(0, roughness(f_{i+1}) - roughness(f_i)))
    """

    def __init__(self):
        super().__init__()
        self.roughness_fn = LaplacianRoughness()

    def forward(self, frames):
        """
        Args:
            frames: [B, T, C, H, W] generated video frames

        Returns:
            loss: scalar
        """
        roughness = self.roughness_fn(frames)  # [B, T]

        # Roughness increases (violations)
        d_roughness = roughness[:, 1:] - roughness[:, :-1]  # [B, T-1]
        violations = F.relu(d_roughness)  # only penalize increases

        return violations.mean()


class DynamicsConsistencyLoss(nn.Module):
    """
    Higher effective sliding speed (V_eff) should produce faster visual
    dynamics (more frame-to-frame change). V_eff is computed from the
    full Lee et al. (2006) kinematics including slip effects.

    L_dynamics = -correlation(mean_frame_diff, V_eff) across the batch
    """

    def forward(self, frames, v_eff):
        """
        Args:
            frames: [B, T, C, H, W] generated video frames
            v_eff:  [B] effective sliding speed from kinematics
                    (or normalized RPM if slip is not used)

        Returns:
            loss: scalar (negative correlation → minimize to maximize correlation)
        """
        # Frame differences
        diffs = (frames[:, 1:] - frames[:, :-1]).abs()  # [B, T-1, C, H, W]
        mean_diff = diffs.mean(dim=(1, 2, 3, 4))  # [B]

        # Need batch size > 1 for correlation
        if frames.shape[0] < 2:
            return torch.tensor(0.0, device=frames.device)

        # Pearson correlation with V_eff (captures both RPM and slip effects)
        x = v_eff - v_eff.mean()
        y = mean_diff - mean_diff.mean()

        corr_num = (x * y).sum()
        corr_den = (x.norm() * y.norm()).clamp(min=1e-8)
        correlation = corr_num / corr_den

        # Minimize negative correlation = maximize positive correlation
        return -correlation


class PhysicsLossBundle(nn.Module):
    """
    Combined physics loss with configurable weights.
    Uses V_eff from Lee et al. kinematics (with slip) for dynamics correlation.
    """

    def __init__(
        self,
        lambda_monotonic=0.05,
        lambda_dynamics=0.05,
        R_g=12.5e-3,
        R_b=5.0e-3,
        theta_deg=60.0,
        S_C=0.95,
    ):
        super().__init__()
        self.monotonic_loss = MonotonicRoughnessLoss()
        self.dynamics_loss = DynamicsConsistencyLoss()
        self.kinematics = ConcentricVGrooveKinematics(R_g, R_b, theta_deg, S_C)

        self.lambda_monotonic = lambda_monotonic
        self.lambda_dynamics = lambda_dynamics

    def forward(self, frames, rpm_normalized, sc=None):
        """
        Args:
            frames: [B, T, C, H, W] generated video frames in [0, 1]
            rpm_normalized: [B] normalized RPM values in [0, 1]
            sc: [B] tensor or None — raw head-contact slip S_C

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss values for logging
        """
        # Compute V_eff from full kinematics (with slip)
        v_eff = self.kinematics.compute_v_eff(rpm_normalized, sc=sc)

        l_mono = self.monotonic_loss(frames)
        l_dyn = self.dynamics_loss(frames, v_eff)

        total = (
            self.lambda_monotonic * l_mono
            + self.lambda_dynamics * l_dyn
        )

        loss_dict = {
            "physics/monotonic": l_mono.item(),
            "physics/dynamics": l_dyn.item(),
            "physics/v_eff_mean": v_eff.mean().item(),
            "physics/total": total.item(),
        }

        return total, loss_dict


# ============================================================
# Latent-space physics losses (no VAE decode needed, ~100x faster)
# ============================================================

class LatentRoughnessProxy(nn.Module):
    """
    Roughness proxy computed directly on VAE latent tensors.
    Uses spatial variance across latent channels as a proxy for
    surface texture complexity (higher variance = rougher).
    """

    def forward(self, latents):
        """
        Args:
            latents: [B, T, C_lat, H_lat, W_lat] latent tensors

        Returns:
            roughness: [B, T] roughness proxy per frame
        """
        B, T, C, H, W = latents.shape
        flat = latents.reshape(B * T, C, H * W)  # [B*T, C, H*W]
        # Spatial variance per channel, then mean across channels
        roughness = flat.var(dim=2).mean(dim=1)  # [B*T]
        return roughness.reshape(B, T)


# NOTE: LatentPrestonKinematicLoss removed (v2). See PrestonKinematicLoss note above.


class LatentMonotonicRoughnessLoss(nn.Module):
    """Monotonic roughness decrease loss in latent space."""

    def __init__(self):
        super().__init__()
        self.roughness_fn = LatentRoughnessProxy()

    def forward(self, latents):
        """
        Args:
            latents: [B, T, C_lat, H_lat, W_lat]
        """
        roughness = self.roughness_fn(latents)  # [B, T]
        d_roughness = roughness[:, 1:] - roughness[:, :-1]
        return F.relu(d_roughness).mean()


class LatentTemporalCoherenceLoss(nn.Module):
    """
    Temporal coherence loss in latent space.
    Penalizes abrupt frame-to-frame jumps to reduce flicker.
    """

    def forward(self, latents):
        # First-order temporal smoothness in latent space.
        return (latents[:, 1:] - latents[:, :-1]).abs().mean()


class LatentDynamicsConsistencyLoss(nn.Module):
    """
    Dynamics consistency loss in latent space.
    Higher V_eff (from kinematics with slip) should produce larger
    frame-to-frame latent differences.
    """

    def forward(self, latents, v_eff):
        """
        Args:
            latents: [B, T, C_lat, H_lat, W_lat]
            v_eff:   [B] effective sliding speed from kinematics
        """
        diffs = (latents[:, 1:] - latents[:, :-1]).abs()  # [B, T-1, C, H, W]
        mean_diff = diffs.mean(dim=(1, 2, 3, 4))  # [B]

        if latents.shape[0] < 2:
            return torch.tensor(0.0, device=latents.device)

        x = v_eff - v_eff.mean()
        y = mean_diff - mean_diff.mean()
        corr_num = (x * y).sum()
        corr_den = (x.norm() * y.norm()).clamp(min=1e-8)
        return -(corr_num / corr_den)


class LatentVelocityRatioLoss(nn.Module):
    """
    Enforce the closed-form velocity ratio
        R_T = ω_c / ω_p = (1 + S_C · sin θ) / (1 + sin θ)
    from Lee et al. (2006) using a batch-wise Pearson correlation between
    an FFT AC/(AC+DC) latent proxy and the ground-truth R_T(S_C).
    """

    def __init__(self, theta_deg: float = 60.0):
        super().__init__()
        self.theta = math.radians(theta_deg)

    def forward(self, latents, sc):
        # latents: [B, T, C, H, W]; sc: [B] or None
        if sc is None or latents.shape[0] < 2:
            return torch.tensor(0.0, device=latents.device)

        temporal_signal = latents.mean(dim=(2, 3, 4))            # [B, T]
        fft_amp = torch.fft.rfft(temporal_signal, dim=1).abs()   # [B, T//2+1]
        dc = fft_amp[:, 0]
        ac = fft_amp[:, 1:].pow(2).sum(dim=1).sqrt()
        R_T_proxy = ac / (ac + dc + 1e-8)                         # [B]

        sin_theta = math.sin(self.theta)
        sc_t = sc.to(latents).float()
        R_T_gt = (1.0 + sc_t * sin_theta) / (1.0 + sin_theta)     # [B]

        x = R_T_proxy - R_T_proxy.mean()
        y = R_T_gt - R_T_gt.mean()
        denom = (x.norm() * y.norm()).clamp(min=1e-8)
        return -(x * y).sum() / denom


class LatentPhysicsLossBundle(nn.Module):
    """
    Combined latent-space physics losses.
    Operates on latent tensors (no VAE decode). Uses V_eff from the full
    Lee et al. (2006) kinematics with slip for dynamics correlation.
    """

    def __init__(
        self,
        lambda_monotonic=0.0,
        lambda_temporal=0.08,
        lambda_dynamics=0.05,
        lambda_ratio=0.0,
        R_g=12.5e-3,
        R_b=5.0e-3,
        theta_deg=60.0,
        S_C=0.95,
        # Kept for backward compat — ignored
        lambda_kinematic=None,
    ):
        super().__init__()
        self.temporal_loss = LatentTemporalCoherenceLoss()
        self.dynamics_loss = LatentDynamicsConsistencyLoss()
        self.ratio_loss = LatentVelocityRatioLoss(theta_deg=theta_deg)
        self.kinematics = ConcentricVGrooveKinematics(R_g, R_b, theta_deg, S_C)
        self.theta_deg = theta_deg
        self._sin_theta = math.sin(math.radians(theta_deg))

        # Keep lambda_monotonic for backward compatibility; it is intentionally unused.
        self.lambda_monotonic = lambda_monotonic
        self.lambda_temporal = lambda_temporal
        self.lambda_dynamics = lambda_dynamics
        self.lambda_ratio = lambda_ratio

    def forward(self, latents, rpm_normalized, sc=None):
        """
        Args:
            latents: [B, T, C_lat, H_lat, W_lat] predicted clean latents
            rpm_normalized: [B] normalized RPM values in [0, 1]
            sc: [B] tensor or None — raw head-contact slip S_C

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss values for logging
        """
        # Compute V_eff from kinematics (RPM + slip)
        v_eff = self.kinematics.compute_v_eff(rpm_normalized, sc=sc)

        l_temp = self.temporal_loss(latents)
        l_dyn = self.dynamics_loss(latents, v_eff)
        l_ratio = self.ratio_loss(latents, sc)

        total = (
            self.lambda_temporal * l_temp
            + self.lambda_dynamics * l_dyn
            + self.lambda_ratio * l_ratio
        )

        if sc is not None and torch.is_tensor(sc):
            r_t_gt_mean = (
                (1.0 + sc.float().mean() * self._sin_theta) / (1.0 + self._sin_theta)
            ).item()
        else:
            r_t_gt_mean = 0.0

        loss_dict = {
            "physics/temporal": l_temp.item(),
            "physics/dynamics": l_dyn.item(),
            "physics/ratio": l_ratio.item() if torch.is_tensor(l_ratio) else float(l_ratio),
            "physics/v_eff_mean": v_eff.mean().item(),
            "physics/R_T_gt_mean": r_t_gt_mean,
            "physics/total": total.item(),
        }

        return total, loss_dict


if __name__ == "__main__":
    # Quick test
    B, T, C, H, W = 4, 25, 3, 256, 256
    frames = torch.randn(B, T, C, H, W).sigmoid()  # [0, 1]
    rpm = torch.tensor([0.0, 0.3, 0.6, 1.0])  # normalized RPM

    # --- Test kinematic model (default slip S_C=0.95) ---
    kin = ConcentricVGrooveKinematics()
    rpms_phys = torch.tensor([20.0, 30.0, 50.0, 70.0])
    omega_p = rpms_phys * (2.0 * math.pi / 60.0)
    v_eff, wc, wb = kin(omega_p)
    print("Concentric V-groove kinematics (Lee et al. 2006, S_C=0.95):")
    for r, v in zip(rpms_phys.tolist(), v_eff.tolist()):
        print(f"  RPM={r:.0f}  →  V_eff={v:.6f} m/s")

    # --- Test with per-sample S_C tensor ---
    sc = torch.tensor([0.50, 0.75, 0.90, 0.99])
    v_eff_slip, _, _ = kin(omega_p, sc=sc)
    print("\nWith per-sample S_C:")
    for r, s, v in zip(rpms_phys.tolist(), sc.tolist(), v_eff_slip.tolist()):
        print(f"  RPM={r:.0f}, S_C={s:.2f}  →  V_eff={v:.6f} m/s")

    # --- compute_ratio monotonicity sanity check (ω_c/ω_b should be monotonic in S_C) ---
    print("\n--- compute_ratio monotonicity (ω_c/ω_b, ω_c/ω_p) ---")
    rpm_ref = torch.full((5,), 0.6)
    sc_grid = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    ratio_wc_wb = kin.compute_ratio(rpm_ref, sc=sc_grid)
    # Closed-form R_T = ω_c / ω_p
    sin_th = math.sin(math.radians(60.0))
    R_T_gt = (1.0 + sc_grid * sin_th) / (1.0 + sin_th)
    for s, r, rt in zip(sc_grid.tolist(), ratio_wc_wb.tolist(), R_T_gt.tolist()):
        print(f"  S_C={s:.2f}  ω_c/ω_b={r:.6f}  R_T={rt:.6f}")
    diffs = ratio_wc_wb[1:] - ratio_wc_wb[:-1]
    assert (diffs > 0).all(), "compute_ratio (ω_c/ω_b) must be strictly increasing in S_C"
    diffs_rt = R_T_gt[1:] - R_T_gt[:-1]
    assert (diffs_rt > 0).all(), "R_T must be strictly increasing in S_C"
    print("  ✓ monotonic in S_C")

    # --- Test full physics bundle (no Preston) ---
    bundle = PhysicsLossBundle()
    total_loss, loss_dict = bundle(frames, rpm, sc=torch.full_like(rpm, 0.95))

    print(f"\nTotal physics loss: {total_loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")

    # Test gradient flow
    frames.requires_grad_(True)
    total_loss, _ = bundle(frames, rpm, sc=torch.full_like(rpm, 0.95))
    total_loss.backward()
    print(f"\nGradient flows: {frames.grad is not None}")
    print(f"Gradient magnitude: {frames.grad.abs().mean().item():.8f}")

    # --- Test latent-space physics bundle ---
    print("\n--- Latent-space physics losses ---")
    B_l, T_l, C_l, H_l, W_l = 4, 14, 4, 64, 16  # latent dims for 512x128
    latents = torch.randn(B_l, T_l, C_l, H_l, W_l)
    latent_bundle = LatentPhysicsLossBundle(lambda_ratio=0.1)
    sc_lat = torch.tensor([0.80, 0.88, 0.95, 0.99])
    lat_loss, lat_dict = latent_bundle(latents, rpm, sc=sc_lat)
    print(f"Latent physics loss: {lat_loss.item():.6f}")
    for k, v in lat_dict.items():
        print(f"  {k}: {v:.6f}")

    # --- Standalone LatentVelocityRatioLoss ---
    print("\n--- LatentVelocityRatioLoss standalone ---")
    ratio_loss_fn = LatentVelocityRatioLoss()
    r_loss = ratio_loss_fn(latents, sc_lat)
    print(f"  L_ratio: {r_loss.item():.6f}")

    latents.requires_grad_(True)
    lat_loss, _ = latent_bundle(latents, rpm, sc=sc_lat)
    lat_loss.backward()
    print(f"\nLatent gradient flows: {latents.grad is not None}")
    print(f"Latent gradient magnitude: {latents.grad.abs().mean().item():.8f}")
