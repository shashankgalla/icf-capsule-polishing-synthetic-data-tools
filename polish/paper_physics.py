"""Explicit implementation of the declared manuscript equations, not a historical run reproduction."""
import math
import torch
from torch import nn


class PaperPhysicsLossBundle(nn.Module):
    def __init__(self, lambda_temporal=.08, lambda_dynamics=.05, lambda_ratio=.1,
                 R_g=.3168, R_b=.01, theta_deg=45., rpm_min=21., rpm_max=108.):
        super().__init__()
        if R_g <= 0 or R_b <= 0 or not 0 < theta_deg < 90:
            raise ValueError('Invalid synthetic geometry')
        self.R_g, self.R_b = R_g, R_b
        self.sin_theta = math.sin(math.radians(theta_deg))
        self.cos_beta = R_g / math.sqrt(R_g ** 2 + R_b ** 2 * self.sin_theta ** 2)
        self.rpm_min, self.rpm_max = rpm_min, rpm_max
        self.weights = lambda_temporal, lambda_dynamics, lambda_ratio

    def targets(self, rpm_norm, slip):
        omega = (self.rpm_min + rpm_norm * (self.rpm_max - self.rpm_min)) * (2 * math.pi / 60)
        rt = (1 + slip * self.sin_theta) / (1 + self.sin_theta)
        wc = omega * rt
        wb = self.R_g * omega * (1 - slip) / (self.R_b * self.cos_beta * (1 + self.sin_theta))
        vc = slip * self.R_g * omega
        veff = torch.sqrt((self.R_g * wc) ** 2 + (self.R_b * wb) ** 2 + vc ** 2)
        return veff, rt, wc, wb

    @staticmethod
    def stratified_ratio(dynamics, ratio, rpm):
        terms = []
        for speed in torch.unique(rpm):
            mask = rpm == speed
            d, r = dynamics[mask], ratio[mask]
            if len(d) < 2 or r.var(unbiased=False) <= 1e-12:
                continue
            terms.append(((d / d.mean().detach().clamp_min(1e-6) - r / r.mean().clamp_min(1e-6)) ** 2).mean())
        return torch.stack(terms).mean() if terms else dynamics.sum() * 0

    def forward(self, latents, rpm_normalized, sc=None):
        if sc is None:
            raise ValueError('Synthetic slip labels are required')
        if latents.shape[1] < 2:
            raise ValueError('At least two frames are required for motion losses')
        d = (latents[:, 1:] - latents[:, :-1]).abs().mean((1, 2, 3, 4))
        veff, rt, _, _ = self.targets(rpm_normalized, sc)
        x, y = veff - veff.mean(), d - d.mean()
        denom = x.norm() * y.norm()
        corr = -(x * y).sum() / denom.clamp_min(1e-8) if len(d) > 1 else d.sum() * 0
        ratio = self.stratified_ratio(d, rt, rpm_normalized)
        temporal = d.mean()
        total = sum(w * loss for w, loss in zip(self.weights, [temporal, corr, ratio]))
        return total, {'physics/temporal': temporal.item(), 'physics/dynamics': corr.item(),
                       'physics/ratio': ratio.item(), 'physics/v_eff_mean': veff.mean().item(),
                       'physics/R_T_gt_mean': rt.mean().item(), 'physics/total': total.item()}
