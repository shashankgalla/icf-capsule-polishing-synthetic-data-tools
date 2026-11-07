"""
Training loop for PhysDiff-Polish.

Fine-tunes SVD-xt with LoRA and physics conditioning in one process.
Implements two-phase training:
  Phase 1 (warmup): diffusion loss only
  Phase 2: diffusion loss + physics losses (ramped in)

W&B tracking (--use_wandb):
  - Per-step: loss curves (total, diffusion, physics components), LR
  - Per-epoch: avg metrics table, sample video at 3 RPMs
"""

import argparse
import hashlib
import math
import os
import time
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, DDIMScheduler
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

from .dataset import PolishingVideoDataset
from .model import (
    create_model,
    encode_image_for_svd,
    prepare_encoder_hidden_states,
    get_added_time_ids,
    DisentangledPhysicsEncoder,
    PhysicsCLIPContrastiveLoss,
    RPMFiLM,
    build_conditioning_hidden_states,
    orthogonality_loss,
)
from .physics_losses import PhysicsLossBundle, LatentPhysicsLossBundle
from .reconstruction_losses import ssim_loss, minsnr_weight
from .paper_physics import PaperPhysicsLossBundle
from .data_integrity import audit
from .checkpoints import save_training_state, restore_training_state


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysDiff-Polish")
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    parser.add_argument("--method_profile", choices=["legacy_supplied", "paper_declared"], required=True, help="Explicit scientific version; paper_declared changes legacy results")
    parser.add_argument("--allow_historical_protocol", action="store_true", help="Explicitly accept shared simulations for historical temporal evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Disable TF32 and benchmark selection for reproducibility tests")
    parser.add_argument("--max_steps", type=int, default=None, help="Integration test limit; not a paper benchmark")
    parser.add_argument("--R_g", type=float, default=.3168)
    parser.add_argument("--R_b", type=float, default=.01)
    parser.add_argument("--theta_deg", type=float, default=45.)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_physics_mlp", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Physics losses
    parser.add_argument("--lambda_kinematic", type=float, default=0.0,
                        help="DEPRECATED: Preston kinematic loss removed. Kept for backward compat.")
    parser.add_argument("--lambda_monotonic", type=float, default=0.0,
                        help="DEPRECATED: roughness proxy removed; kept for compatibility")
    parser.add_argument("--lambda_temporal", type=float, default=0.08,
                        help="Weight for latent temporal coherence loss (reduces flicker)")
    parser.add_argument("--lambda_dynamics", type=float, default=0.05)
    parser.add_argument("--lambda_ratio", type=float, default=0.0,
                        help="Weight for latent velocity-ratio loss (Lee et al. R_T proxy)")
    parser.add_argument("--physics_warmup_epoch", type=int, default=10,
                        help="Start introducing physics losses after this epoch")
    parser.add_argument("--physics_ramp_epochs", type=int, default=5,
                        help="Ramp physics losses linearly over this many epochs")
    parser.add_argument("--lambda_disentangle", type=float, default=0.01,
                        help="Weight for orthogonality loss on physics tokens")
    parser.add_argument("--lambda_contrastive", type=float, default=0.01,
                        help="Weight for physics-CLIP contrastive loss")

    # Reconstruction & generation quality (SOTA improvements)
    parser.add_argument("--lambda_recon", type=float, default=1.0,
                        help="Weight for latent L1 reconstruction loss (directly improves PSNR)")
    parser.add_argument("--lambda_ssim", type=float, default=0.5,
                        help="Weight for latent SSIM reconstruction loss (directly improves SSIM)")
    parser.add_argument("--minsnr_gamma", type=float, default=5.0,
                        help="Min-SNR-gamma for diffusion loss weighting (Hang et al. 2023)")
    parser.add_argument("--noise_aug", type=float, default=0.02,
                        help="Std of noise added to conditioning image during training")
    parser.add_argument("--cfg_dropout", type=float, default=0.1,
                        help="Fraction of batches to null-condition (enables CFG at inference)")

    # Data (rectangular: width > height for polishing strip videos)
    parser.add_argument("--crop_width", type=int, default=736)
    parser.add_argument("--crop_height", type=int, default=184)
    parser.add_argument("--output_width", type=int, default=0,
                        help="Output width; <=0 means auto infer from resize_factor")
    parser.add_argument("--output_height", type=int, default=0,
                        help="Output height; <=0 means auto infer from resize_factor")
    parser.add_argument("--resize_factor", type=float, default=0.25,
                        help="Resize full frames by this factor (0.25 = one quarter)")
    parser.add_argument("--allow_random_crop", action="store_true",
                        help="Enable random crop augmentation (disabled by default)")
    # Legacy (ignored if width/height are set)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--output_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=30,
                        help="Frames per clip (reduce for VRAM; SVD-xt default is 25)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--clips_per_sample", type=int, default=3)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (0.2 = 20%% for quick validation)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # Logging & sample generation
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_epoch", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="physdiff-polish")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if None)")
    parser.add_argument("--sample_every_epoch", type=int, default=1,
                        help="Generate sample videos every N epochs")
    parser.add_argument("--sample_rpms", type=str, default="10,20,30,40",
                        help="Comma-separated RPMs for sample generation")
    parser.add_argument("--sample_steps", type=int, default=25,
                        help="Diffusion steps for sample generation")
    parser.add_argument("--sample_image", type=str, default=None,
                        help="Path to conditioning image for samples (auto-picks if None)")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale for inference (1.0 = no CFG)")
    parser.add_argument("--eval_every_epoch", type=int, default=5,
                        help="Evaluate on validation set every N epochs")
    parser.add_argument("--num_val_samples", type=int, default=9,
                        help="Number of validation clips to evaluate per epoch")

    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="cross_attn",
        choices=["cross_attn", "film", "image_only"],
        help="Conditioning strategy: cross_attn (ours), film (RPM FiLM), image_only baseline",
    )

    # Physics encoder ablation knobs
    parser.add_argument(
        "--sc_encoding",
        type=str,
        default="fourier",
        choices=["fourier", "raw"],
        help="S_C input encoding: 'fourier' (default) or 'raw' (single Linear)",
    )
    parser.add_argument(
        "--physics_encoding",
        type=str,
        default="disentangled",
        choices=["disentangled", "concat"],
        help="'disentangled' (per-variable Fourier + interaction attn) or 'concat' (shared MLP, single token)",
    )
    parser.add_argument(
        "--num_frequencies_sc",
        type=int,
        default=16,
        help="Fourier frequencies for S_C encoder (only when --sc_encoding fourier)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (strict=False, legacy shell/fps keys dropped)",
    )
    parser.add_argument(
        "--freeze_lora",
        action="store_true",
        help="Freeze LoRA parameters (used in physics-only fine-tuning phase)",
    )

    return parser.parse_args()


def get_physics_loss_weight(epoch, warmup_epoch, ramp_epochs):
    """Compute physics loss weight multiplier with warmup ramp."""
    if epoch < warmup_epoch:
        return 0.0
    elif epoch < warmup_epoch + ramp_epochs:
        return (epoch - warmup_epoch) / ramp_epochs
    else:
        return 1.0


def cosine_lr_schedule(optimizer, step, total_steps, warmup_steps, base_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
    return lr


@torch.no_grad()
def generate_sample_videos(
    unet, vae, image_encoder, feature_extractor, physics_mlp,
    rpm_film,
    conditioning_mode,
    noise_scheduler_cfg, cond_image_path, rpms, num_frames,
    num_inference_steps, output_width, output_height, device, weight_dtype,
    output_dir, epoch, cfg_scale=3.0,
):
    """
    Generate sample videos at given RPMs using DDIM sampling.
    Returns dict of {rpm: np.ndarray [T, H, W, 3] uint8}.
    """
    import imageio
    from PIL import Image

    unet.eval()
    physics_mlp.eval()

    try:
        # Load & resize conditioning image to match training resolution
        cond_pil = Image.open(cond_image_path).convert("RGB")
        cond_np = np.array(cond_pil)  # [H, W, 3]
        cond_np = cv2.resize(cond_np, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        cond_np = cond_np.astype(np.float32) / 255.0
        cond_tensor = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        cond_tensor = cond_tensor.to(device)

        # CLIP encode
        image_embeddings = encode_image_for_svd(
            image_encoder, feature_extractor, cond_tensor, device, weight_dtype
        )

        # VAE encode conditioning image
        cond_for_vae = cond_tensor.to(dtype=weight_dtype)
        cond_latent = vae.encode(cond_for_vae).latent_dist.mode()
        cond_latent = cond_latent * vae.config.scaling_factor
        # [1, 4, H_lat, W_lat]

        # Use DDIMScheduler for inference — avoids EulerDiscreteScheduler
        # step_index tracking bugs in diffusers >=0.30
        infer_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler_cfg["num_train_timesteps"],
            beta_start=noise_scheduler_cfg["beta_start"],
            beta_end=noise_scheduler_cfg["beta_end"],
            beta_schedule=noise_scheduler_cfg["beta_schedule"],
            prediction_type="v_prediction",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        infer_scheduler.set_timesteps(num_inference_steps, device=device)

        results = {}
        for rpm in rpms:
            # Physics params: (rpm_norm, slip_raw). Default raw S_C at head contact.
            low, high = (21., 108.) if getattr(physics_mlp, "release_config", {}).get("method_profile") == "paper_declared" else (10., 40.)
            rpm_norm = (rpm - low) / (high - low)
            slip_raw = 0.95
            physics_params = torch.tensor(
                [[rpm_norm, slip_raw]],
                dtype=torch.float32, device=device,
            )
            encoder_hidden_states, _ = build_conditioning_hidden_states(
                image_embeddings=image_embeddings,
                physics_params=physics_params,
                conditioning_mode=conditioning_mode,
                physics_mlp=physics_mlp,
                rpm_film=rpm_film,
                weight_dtype=weight_dtype,
            )
            added_time_ids = get_added_time_ids(
                batch_size=1, dtype=weight_dtype, device=device
            )

            # Repeat cond_latent across frames: [1, T, 4, H, W]
            cond_lat_t = cond_latent.unsqueeze(1).expand(1, num_frames, -1, -1, -1).float()

            # Start from pure noise [1, T, 4, H_lat, W_lat]
            latent_shape = (1, num_frames, cond_latent.shape[1],
                            cond_latent.shape[2], cond_latent.shape[3])
            latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

            for t in infer_scheduler.timesteps:
                unet_input = torch.cat([latents, cond_lat_t], dim=2)
                ts = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)

                # Conditional prediction
                noise_pred_cond = unet(
                    unet_input.to(weight_dtype), ts,
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids,
                ).sample.float()

                # Classifier-Free Guidance: blend conditional and unconditional
                if cfg_scale > 1.0:
                    uncond_hidden = torch.zeros_like(encoder_hidden_states)
                    noise_pred_uncond = unet(
                        unet_input.to(weight_dtype), ts,
                        encoder_hidden_states=uncond_hidden,
                        added_time_ids=added_time_ids,
                    ).sample.float()
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = infer_scheduler.step(noise_pred, t, latents).prev_sample

            # Decode to pixels
            lat_flat = latents.reshape(-1, *latents.shape[2:])  # [T, 4, H, W]
            decoded = vae.decode(
                lat_flat.to(weight_dtype) / vae.config.scaling_factor,
                num_frames=num_frames,
            ).sample
            decoded = decoded.float().clamp(0, 1)  # [T, 3, H, W]
            frames_np = (decoded.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            results[rpm] = frames_np

            # Save as mp4
            out_path = output_dir / f"epoch{epoch:03d}_rpm{rpm}.mp4"
            writer = imageio.get_writer(str(out_path), fps=7, codec="libx264")
            for f in frames_np:
                writer.append_data(f)
            writer.close()
            print(f"  RPM={rpm} done -> {out_path.name}")

        return results

    finally:
        unet.train()
        physics_mlp.train()


@torch.no_grad()
def validate_epoch(
    unet, vae, image_encoder, feature_extractor, physics_mlp,
    rpm_film,
    conditioning_mode,
    noise_scheduler_cfg, val_dataset, num_samples, num_frames,
    num_inference_steps, output_width, output_height,
    cfg_scale, device, weight_dtype, output_dir, epoch,
):
    """
    Generate videos for a subset of validation clips, compute PSNR/SSIM,
    and save side-by-side frame comparisons.
    Returns dict of metrics.
    """
    import imageio
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim

    unet.eval()
    physics_mlp.eval()

    val_dir = Path(output_dir) / "val_eval" / f"epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Sample indices uniformly
    total = len(val_dataset)
    step_size = max(1, total // num_samples)
    indices = list(range(0, total, step_size))[:num_samples]

    # Set up DDIM scheduler for inference
    infer_scheduler = DDIMScheduler(
        num_train_timesteps=noise_scheduler_cfg["num_train_timesteps"],
        beta_start=noise_scheduler_cfg["beta_start"],
        beta_end=noise_scheduler_cfg["beta_end"],
        beta_schedule=noise_scheduler_cfg["beta_schedule"],
        prediction_type="v_prediction",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    infer_scheduler.set_timesteps(num_inference_steps, device=device)

    all_psnrs, all_ssims = [], []

    try:
        for i, idx in enumerate(indices):
            sample = val_dataset[idx]
            gt_video = sample["video"]          # [T, C, H, W] float [0,1]
            cond_image = sample["cond_image"]    # [C, H, W] float [0,1]
            physics_params = sample["physics_params"]  # [2] — (rpm_norm, slip_raw)

            cond_tensor = cond_image.unsqueeze(0).to(device)  # [1, C, H, W]

            # CLIP encode
            image_embeddings = encode_image_for_svd(
                image_encoder, feature_extractor, cond_tensor, device, weight_dtype
            )

            # VAE encode conditioning image
            cond_for_vae = cond_tensor.to(dtype=weight_dtype)
            cond_latent = vae.encode(cond_for_vae).latent_dist.mode()
            cond_latent = cond_latent * vae.config.scaling_factor

            pp = physics_params.unsqueeze(0).to(device).float()
            encoder_hidden_states, _ = build_conditioning_hidden_states(
                image_embeddings=image_embeddings,
                physics_params=pp,
                conditioning_mode=conditioning_mode,
                physics_mlp=physics_mlp,
                rpm_film=rpm_film,
                weight_dtype=weight_dtype,
            )
            added_time_ids = get_added_time_ids(batch_size=1, dtype=weight_dtype, device=device)

            # Repeat cond latent across frames
            cond_lat_t = cond_latent.unsqueeze(1).expand(1, num_frames, -1, -1, -1).float()

            # Start from noise
            latent_shape = (1, num_frames, cond_latent.shape[1],
                            cond_latent.shape[2], cond_latent.shape[3])
            latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

            # Denoise with CFG
            for t in infer_scheduler.timesteps:
                unet_input = torch.cat([latents, cond_lat_t], dim=2)
                ts = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)

                noise_pred_cond = unet(
                    unet_input.to(weight_dtype), ts,
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids,
                ).sample.float()

                if cfg_scale > 1.0:
                    uncond_hidden = torch.zeros_like(encoder_hidden_states)
                    noise_pred_uncond = unet(
                        unet_input.to(weight_dtype), ts,
                        encoder_hidden_states=uncond_hidden,
                        added_time_ids=added_time_ids,
                    ).sample.float()
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = infer_scheduler.step(noise_pred, t, latents).prev_sample

            # Decode to pixels
            lat_flat = latents.reshape(-1, *latents.shape[2:])
            decoded = vae.decode(
                lat_flat.to(weight_dtype) / vae.config.scaling_factor,
                num_frames=num_frames,
            ).sample
            gen_frames = decoded.float().clamp(0, 1)  # [T, C, H, W]

            # GT frames: [T, C, H, W] → uint8
            gt_np = (gt_video.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            gen_np = (gen_frames.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            # Compute per-frame PSNR/SSIM
            for t_idx in range(min(num_frames, len(gt_np), len(gen_np))):
                psnr_v = skimage_psnr(gt_np[t_idx], gen_np[t_idx], data_range=255)
                ssim_v = skimage_ssim(gt_np[t_idx], gen_np[t_idx], channel_axis=-1, data_range=255)
                all_psnrs.append(psnr_v)
                all_ssims.append(ssim_v)

            # Save side-by-side comparison frames (first, middle, last)
            rpm_val = (
                physics_params[0].item()
                * (val_dataset.rpm_max - val_dataset.rpm_min)
                + val_dataset.rpm_min
            )
            for t_idx in [0, num_frames // 2, num_frames - 1]:
                if t_idx < len(gt_np) and t_idx < len(gen_np):
                    side_by_side = np.concatenate([gt_np[t_idx], gen_np[t_idx]], axis=1)
                    fname = val_dir / f"clip{i:02d}_rpm{rpm_val:.0f}_frame{t_idx:02d}_GTvsGen.png"
                    Image.fromarray(side_by_side).save(str(fname))

            # Save generated video
            out_path = val_dir / f"clip{i:02d}_rpm{rpm_val:.0f}.mp4"
            writer = imageio.get_writer(str(out_path), fps=7, codec="libx264")
            for f in gen_np:
                writer.append_data(f)
            writer.close()

            print(f"  Val clip {i+1}/{len(indices)}: PSNR={np.mean(all_psnrs[-num_frames:]):.2f} SSIM={np.mean(all_ssims[-num_frames:]):.4f}")

        metrics = {
            "val/psnr_mean": float(np.mean(all_psnrs)) if all_psnrs else 0.0,
            "val/ssim_mean": float(np.mean(all_ssims)) if all_ssims else 0.0,
            "val/psnr_std": float(np.std(all_psnrs)) if all_psnrs else 0.0,
            "val/ssim_std": float(np.std(all_ssims)) if all_ssims else 0.0,
            "val/num_clips": len(indices),
        }
        print(f"\n  Val Epoch {epoch}: PSNR={metrics['val/psnr_mean']:.2f}±{metrics['val/psnr_std']:.2f}  SSIM={metrics['val/ssim_mean']:.4f}±{metrics['val/ssim_std']:.4f}")
        return metrics

    finally:
        unet.train()
        physics_mlp.train()


def train():
    args = parse_args()

    if args.lr <= 0 or args.lr_physics_mlp <= 0 or not 0 < args.data_fraction <= 1:
        raise ValueError("Learning rates and data_fraction must be positive")
    if args.sample_every_epoch < 1 or args.eval_every_epoch < 1 or args.save_every_epoch < 1:
        raise ValueError("Logging and checkpoint intervals must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("Epochs and batch counts must be positive")
    if args.use_wandb and wandb is None:
        raise ImportError("Install wandb to enable optional external logging")
    report = audit(args.metadata_csv, strict=not args.allow_historical_protocol)
    if report["status"] != "PASS":
        raise ValueError("Synthetic integrity preflight failed: " + json.dumps(report["issues"]))
    manifest_dir = Path(args.metadata_csv).resolve().parent
    args.split_fingerprint = {name: hashlib.sha256((manifest_dir / name).read_bytes()).hexdigest() for name in ["metadata.csv", "train.csv", "val.csv", "test.csv"]}
    output_path = Path(args.output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not args.resume_from:
        raise FileExistsError("Refusing to overwrite a nonempty experiment directory")
    if args.resume_from and args.max_steps:
        raise ValueError("max_steps integration checkpoints do not support exact mid-epoch resume")
    norm = {"rpm_column": "raw_rpm", "rpm_min": 21., "rpm_max": 108.} if args.method_profile == "paper_declared" else {"rpm_column": "rpm", "rpm_min": 10., "rpm_max": 40.}

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )

    if accelerator.num_processes != 1:
        raise ValueError("This release candidate verifies single-process training only; distributed state needs a separate audit")
    set_seed(args.seed)

    # Parse sample RPMs
    sample_rpms = [int(r) for r in args.sample_rpms.split(",")]

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)

        if args.use_wandb:
            run_name = args.wandb_run_name or f"physdiff_lr{args.lr}_rank{args.lora_rank}_{time.strftime('%m%d_%H%M')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                tags=["physdiff-polish", f"lora-r{args.lora_rank}", f"frames-{args.num_frames}"],
            )
            wandb.define_metric("global_step")
            wandb.define_metric("loss/*", step_metric="global_step")
            wandb.define_metric("lr", step_metric="global_step")
            wandb.define_metric("epoch/*", step_metric="epoch")

    if accelerator.is_main_process:
        Path(args.output_dir, "data_audit.json").write_text(json.dumps(report, indent=2))
        Path(args.output_dir, "run_config.json").write_text(json.dumps(vars(args), indent=2))

    # Load model components
    components = create_model(
        pretrained_model_name=args.pretrained_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=accelerator.device,
        dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
        sc_encoding=args.sc_encoding,
        num_frequencies_sc=args.num_frequencies_sc,
        physics_encoding=args.physics_encoding,
        method_profile=args.method_profile,
    )

    unet = components["unet"]
    vae = components["vae"]
    image_encoder = components["image_encoder"]
    feature_extractor = components["feature_extractor"]
    physics_mlp = components["physics_mlp"]
    physics_mlp.release_config = {"method_profile": args.method_profile}
    rpm_film = RPMFiLM(embed_dim=1024) if args.conditioning_mode == "film" else None

    physics_loss_fn = (PaperPhysicsLossBundle(lambda_temporal=args.lambda_temporal, lambda_dynamics=args.lambda_dynamics, lambda_ratio=args.lambda_ratio, R_g=args.R_g, R_b=args.R_b, theta_deg=args.theta_deg) if args.method_profile == "paper_declared" else LatentPhysicsLossBundle(lambda_temporal=args.lambda_temporal, lambda_monotonic=args.lambda_monotonic, lambda_dynamics=args.lambda_dynamics, lambda_ratio=args.lambda_ratio))

    # Optional: resume from a previous checkpoint (strict=False, legacy keys dropped)
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {resume_dir}")

        saved = json.loads((resume_dir / "run_config.json").read_text())
        for key in ["method_profile", "conditioning_mode", "sc_encoding", "num_frequencies_sc", "physics_encoding", "lora_rank", "lora_alpha", "batch_size", "grad_accum", "epochs", "lr", "lr_physics_mlp", "deterministic", "split_fingerprint", "seed", "num_frames", "output_width", "output_height", "data_fraction", "mixed_precision"]:
            if saved[key] != getattr(args, key):
                raise ValueError(f"Exact resume configuration mismatch: {key}")
        if not (resume_dir / "trainer_state.pt").is_file():
            raise FileNotFoundError("Missing optimizer and RNG state for exact resume")
        unet_lora_dir = resume_dir / "unet_lora"
        if not unet_lora_dir.is_dir():
            raise FileNotFoundError("Missing resume adapter")
        if unet_lora_dir.exists():
            try:
                from peft import PeftModel
                # Load adapter weights into the already-wrapped PEFT model, strict=False.
                unet.load_adapter(str(unet_lora_dir), adapter_name="default", is_trainable=True)
                if accelerator.is_main_process:
                    print(f"  loaded LoRA adapter from {unet_lora_dir}")
            except Exception as e:
                if accelerator.is_main_process:
                    raise RuntimeError("LoRA resume failed") from e

        mlp_ckpt = resume_dir / "physics_mlp.pt"
        if not mlp_ckpt.is_file():
            raise FileNotFoundError("Missing resume encoder")
        physics_mlp.load_state_dict(torch.load(mlp_ckpt, map_location="cpu", weights_only=True), strict=True)

        if rpm_film is not None:
            film_ckpt = resume_dir / "rpm_film.pt"
            if film_ckpt.exists():
                rpm_film.load_state_dict(torch.load(film_ckpt, map_location="cpu"), strict=False)

        loss_ckpt = resume_dir / "physics_loss.pt"
        if loss_ckpt.exists():
            # Physics loss bundle changed shape (S_A/S_B → S_C); tolerate mismatch.
            try:
                physics_loss_fn.load_state_dict(
                    torch.load(loss_ckpt, map_location="cpu", weights_only=True), strict=False
                )
            except Exception as e:
                if accelerator.is_main_process:
                    raise RuntimeError("Physics state resume failed") from e

    # Use DDPMScheduler for training (supports random integer timesteps).
    # The pipeline's EulerDiscreteScheduler is for inference only.
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="v_prediction",
    )

    # Enable gradient checkpointing to reduce VRAM
    if args.gradient_checkpointing:
        # Reach through PeftModel → LoraModel → base UNet
        base_unet = unet
        while hasattr(base_unet, 'model'):
            base_unet = base_unet.model
        base_unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")

    # Enable tf32 for faster float32 matmuls on Ampere+ GPUs (RTX 3090/4090)
    torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
    torch.backends.cudnn.allow_tf32 = not args.deterministic
    # Optimize cudnn for fixed input sizes
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    # Move frozen models
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    contrastive_loss_fn = PhysicsCLIPContrastiveLoss(temperature=0.1)

    # Dataset (rectangular output for polishing strips)
    full_dataset = PolishingVideoDataset(
        metadata_csv=args.metadata_csv,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        output_width=args.output_width,
        output_height=args.output_height,
        resize_factor=args.resize_factor,
        no_crop=not args.allow_random_crop,
        clips_per_sample=args.clips_per_sample,
        augment=True,
        num_frames=args.num_frames,
        split="train", **norm,
    )

    if len(full_dataset) < args.batch_size:
        raise ValueError("Training split has fewer samples than one complete batch")

    # Keep train/eval/generation resolution consistent with dataset-inferred size.
    args.output_width = full_dataset.output_width
    args.output_height = full_dataset.output_height

    # Subset dataset if data_fraction < 1.0
    if args.data_fraction < 1.0:
        from torch.utils.data import Subset
        total = len(full_dataset)
        subset_size = max(1, int(total * args.data_fraction))
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(total, generator=generator)[:subset_size].tolist()
        dataset = Subset(full_dataset, indices)
        if subset_size < args.batch_size:
            raise ValueError("Selected subset is smaller than one complete batch")
        if accelerator.is_main_process:
            print(f"Using {subset_size}/{total} samples ({args.data_fraction*100:.0f}% of dataset)")
    else:
        dataset = full_dataset

    # Validation dataset (no augmentation, no subset)
    val_dataset = PolishingVideoDataset(
        metadata_csv=args.metadata_csv,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        output_width=args.output_width,
        output_height=args.output_height,
        resize_factor=args.resize_factor,
        no_crop=True,
        clips_per_sample=1,
        augment=False,
        num_frames=args.num_frames,
        split="val", **norm,
    )
    if accelerator.is_main_process:
        print(f"Validation dataset: {len(val_dataset)} clips")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optionally freeze LoRA params (physics-only fine-tuning phase)
    if args.freeze_lora:
        for name, p in unet.named_parameters():
            if "lora_" in name.lower() or p.requires_grad:
                p.requires_grad_(False)
        if accelerator.is_main_process:
            print("Froze LoRA parameters (--freeze_lora)")

    # Optimizer: separate lr for LoRA params and physics MLP
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    mlp_params = list(physics_mlp.parameters()) if args.conditioning_mode == "cross_attn" else []
    film_params = list(rpm_film.parameters()) if rpm_film is not None else []
    physics_loss_params = list(physics_loss_fn.parameters())

    optim_groups = [
        {"params": lora_params, "lr": args.lr, "lr_scale": 1.0},
        {"params": physics_loss_params, "lr": args.lr_physics_mlp, "lr_scale": args.lr_physics_mlp / args.lr},
    ]
    if mlp_params:
        optim_groups.append({"params": mlp_params, "lr": args.lr_physics_mlp, "lr_scale": args.lr_physics_mlp / args.lr})
    if film_params:
        optim_groups.append({"params": film_params, "lr": args.lr_physics_mlp, "lr_scale": args.lr_physics_mlp / args.lr})

    optimizer = torch.optim.AdamW(optim_groups, weight_decay=args.weight_decay)

    # Prepare with accelerator
    if rpm_film is not None:
        unet, physics_mlp, rpm_film, physics_loss_fn, optimizer, dataloader = accelerator.prepare(
            unet, physics_mlp, rpm_film, physics_loss_fn, optimizer, dataloader
        )
    else:
        unet, physics_mlp, physics_loss_fn, optimizer, dataloader = accelerator.prepare(
            unet, physics_mlp, physics_loss_fn, optimizer, dataloader
        )

    sample_image_path = args.sample_image or str(Path(full_dataset.clips[0]["source_dir"]) / f"frame_{full_dataset.clips[0]['start_frame']:06d}.png")

    # Scheduler config dict for inference scheduler creation
    noise_scheduler_cfg = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
    }

    # Training info
    total_steps = math.ceil(len(dataloader) / args.grad_accum) * args.epochs
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Training PhysDiff-Polish")
        print(f"{'='*60}")
        print(f"Dataset size: {len(dataset)}")
        print(f"Training resolution: {args.output_width}x{args.output_height}")
        print(f"Batch size (effective): {args.batch_size * args.grad_accum * accelerator.num_processes}")
        print(f"Total steps: {total_steps}")
        print(f"Physics warmup at epoch: {args.physics_warmup_epoch}")
        print(f"Sample image: {sample_image_path}")
        print(f"Sample RPMs: {sample_rpms}")
        print(f"Conditioning mode: {args.conditioning_mode}")
        print(f"{'='*60}\n")

    def save_checkpoint(directory, epoch_number):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(directory / "unet_lora")
        torch.save(accelerator.unwrap_model(physics_mlp).state_dict(), directory / "physics_mlp.pt")
        torch.save(accelerator.unwrap_model(physics_loss_fn).state_dict(), directory / "physics_loss.pt")
        if rpm_film is not None:
            torch.save(accelerator.unwrap_model(rpm_film).state_dict(), directory / "rpm_film.pt")
        save_training_state(directory, optimizer, epoch_number, global_step, best_loss, args, scaler=accelerator.scaler)

    # Training loop. Epoch-boundary state is saved separately from frozen weights.
    global_step = 0
    best_loss = float("inf")
    start_epoch = 0
    if args.resume_from:
        resumed = restore_training_state(args.resume_from, optimizer, scaler=accelerator.scaler)
        global_step, best_loss, start_epoch = resumed["global_step"], resumed["best_loss"], resumed["next_epoch"]
    epoch_start_time = None

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        unet.train()
        physics_mlp.train()
        if rpm_film is not None:
            rpm_film.train()

        physics_weight = get_physics_loss_weight(
            epoch, args.physics_warmup_epoch, args.physics_ramp_epochs
        )

        epoch_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_ssim_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            leave=True,
        )
        for batch_idx, batch in pbar:
            with accelerator.accumulate(unet, physics_mlp):
                video = batch["video"]        # [B, T, C, H, W]
                cond_image = batch["cond_image"]  # [B, C, H, W]
                physics_params = batch["physics_params"]  # [B, 2] — (rpm_norm, slip_raw)

                B, T, C, H, W = video.shape

                # 1. Encode video frames to latent space
                # Reshape for VAE: [B*T, C, H, W]
                video_flat = video.reshape(B * T, C, H, W).to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(video_flat).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                # Reshape back: [B, T, latent_C, latent_H, latent_W]
                latent_C = latents.shape[1]
                latent_H = latents.shape[2]
                latent_W = latents.shape[3]
                latents = latents.reshape(B, T, latent_C, latent_H, latent_W)
                # latents is now [B, T, 4, H_lat, W_lat]

                # Encode conditioning image (first frame) with VAE for
                # channel-concatenation (SVD UNet has in_channels=8).
                cond_for_vae = cond_image.to(dtype=weight_dtype)
                # Noise augmentation on conditioning image (SVD training technique)
                # Encourages the model to learn motion rather than memorise frames.
                if args.noise_aug > 0.0:
                    aug_sigma = args.noise_aug * torch.rand(
                        B, 1, 1, 1, device=cond_for_vae.device, dtype=cond_for_vae.dtype
                    )
                    cond_for_vae = (
                        cond_for_vae + torch.randn_like(cond_for_vae) * aug_sigma
                    ).clamp(0.0, 1.0)
                with torch.no_grad():
                    cond_latent = vae.encode(cond_for_vae).latent_dist.mode()
                    cond_latent = cond_latent * vae.config.scaling_factor
                # cond_latent: [B, 4, H_lat, W_lat] → repeat across frames
                cond_latent = cond_latent.unsqueeze(1).expand_as(latents)
                # cond_latent: [B, T, 4, H_lat, W_lat]

                # Cast to float32 for noise scheduling math
                latents = latents.float()
                cond_latent = cond_latent.float()

                # 2. Encode conditioning image with CLIP
                with torch.no_grad():
                    image_embeddings = encode_image_for_svd(
                        image_encoder, feature_extractor,
                        cond_image, accelerator.device, weight_dtype
                    )

                # 3-4. Build conditioning hidden states
                encoder_hidden_states, indep_physics_tokens = build_conditioning_hidden_states(
                    image_embeddings=image_embeddings,
                    physics_params=physics_params,
                    conditioning_mode=args.conditioning_mode,
                    physics_mlp=physics_mlp,
                    rpm_film=rpm_film,
                    weight_dtype=weight_dtype,
                )

                # CFG dropout: randomly null-condition a fraction of batches.
                # This teaches the model to generate unconditionally, enabling
                # classifier-free guidance (CFG) at inference for quality boost.
                if args.cfg_dropout > 0.0:
                    drop_mask = (
                        torch.rand(B, device=accelerator.device) < args.cfg_dropout
                    ).view(B, 1, 1)
                    encoder_hidden_states = encoder_hidden_states * (~drop_mask)

                # 5. Add noise (diffusion forward process)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=latents.device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate conditioning image latent along channel dim
                # UNet expects [B, T, 8, H, W] (4 noisy + 4 cond)
                unet_input = torch.cat([noisy_latents, cond_latent], dim=2)

                # 6. Prepare added_time_ids
                added_time_ids = get_added_time_ids(
                    batch_size=B, dtype=weight_dtype, device=accelerator.device
                )

                # 7. UNet forward pass: predict noise
                noise_pred = unet(
                    unet_input.to(dtype=weight_dtype),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=added_time_ids,
                ).sample

                # 8. Shared α-schedule tensor (float32 throughout)
                # Shape [B] for MinSNR weighting; broadcastable view for latent math.
                alpha_prod_flat = noise_scheduler.alphas_cumprod[timesteps].to(
                    device=latents.device, dtype=torch.float32
                )  # [B]
                alpha_prod = alpha_prod_flat.view(
                    B, *([1] * (latents.dim() - 1))
                )  # [B, 1, 1, 1, 1] broadcastable

                # v-target: v = sqrt(α)·ε − sqrt(1−α)·x₀
                v_target = (
                    alpha_prod.sqrt() * noise.float()
                    - (1 - alpha_prod).sqrt() * latents.float()
                )

                # x₀ prediction from v-pred (always computed — needed by recon & physics)
                # Inversion: x₀ = sqrt(α)·xₜ − sqrt(1−α)·v
                x0_pred = (
                    alpha_prod.sqrt() * noisy_latents.float()
                    - (1 - alpha_prod).sqrt() * noise_pred.float()
                )

                # 8a. Min-SNR-γ weighted diffusion loss  (Hang et al., NeurIPS 2023)
                # Focuses learning on high-SNR (low-noise) timesteps that matter
                # most for image quality; γ=5 is the recommended default.
                minsnr_w = minsnr_weight(
                    timesteps, noise_scheduler.alphas_cumprod, args.minsnr_gamma
                ).detach()  # [B]  — stop-gradient: weight is a schedule, not a learned param
                # Per-sample MSE, weighted by MinSNR, then averaged over the batch
                diffusion_loss = (
                    minsnr_w
                    * F.mse_loss(noise_pred.float(), v_target, reduction="none")
                      .flatten(1).mean(1)
                ).mean()

                # 8b. Latent reconstruction losses — active from epoch 1
                # L1 directly minimises mean absolute pixel error → improves PSNR.
                # SSIM loss penalises luminance/contrast/structure mismatches → improves SSIM.
                recon_loss    = F.l1_loss(x0_pred, latents.float())
                ssim_loss_val = ssim_loss(x0_pred, latents.float())

                # 9. Physics losses (latent-space, enabled after warmup epoch)
                #    Uses V_eff from Lee et al. kinematics with slip
                total_physics_loss = torch.tensor(0.0, device=accelerator.device)
                physics_dict = {}

                if physics_weight > 0:
                    rpm_norm = physics_params[:, 0].float()
                    slip_raw = physics_params[:, 1].float()  # raw S_C
                    total_physics_loss, physics_dict = physics_loss_fn(
                        x0_pred, rpm_norm, sc=slip_raw,
                    )

                # 10. Disentanglement losses (on physics tokens)
                disentangle_loss = torch.tensor(0.0, device=accelerator.device)
                contrastive_loss_val = torch.tensor(0.0, device=accelerator.device)
                if physics_weight > 0 and indep_physics_tokens is not None:
                    disentangle_loss = orthogonality_loss(indep_physics_tokens.float())
                    contrastive_loss_val = contrastive_loss_fn(
                        indep_physics_tokens.float(), physics_params.float()
                    )
                    physics_dict["physics/orthogonality"] = disentangle_loss.item()
                    physics_dict["physics/contrastive"] = contrastive_loss_val.item()

                # 11. Total loss
                loss = (
                    diffusion_loss
                    + args.lambda_recon  * recon_loss
                    + args.lambda_ssim   * ssim_loss_val
                    + physics_weight * total_physics_loss
                    + physics_weight * args.lambda_disentangle * disentangle_loss
                    + physics_weight * args.lambda_contrastive * contrastive_loss_val
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(physics_mlp.parameters()),
                        max_norm=1.0,
                    )
                optimizer.step()
                optimizer.zero_grad()

                # Update LR
                cosine_lr_schedule(optimizer, global_step, total_steps, args.warmup_steps, args.lr)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")

            # Logging
            epoch_loss += loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_physics_loss += (physics_weight * total_physics_loss).item()
            epoch_recon_loss += recon_loss.item()
            epoch_ssim_loss += ssim_loss_val.item()
            num_batches += 1

            if accelerator.sync_gradients:
                global_step += 1

            # Update tqdm bar every step
            if accelerator.is_main_process:
                cur_lr = optimizer.param_groups[0]["lr"]
                phys_loss_val = (physics_weight * total_physics_loss).item()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "diff": f"{diffusion_loss.item():.4f}",
                    "rec": f"{recon_loss.item():.4f}",
                    "ssim": f"{ssim_loss_val.item():.4f}",
                    "phys": f"{phys_loss_val:.4f}",
                    "lr": f"{cur_lr:.1e}",
                })

            if batch_idx % args.log_every == 0 and accelerator.is_main_process:
                cur_lr = optimizer.param_groups[0]["lr"]
                phys_loss_val = (physics_weight * total_physics_loss).item()

                if args.use_wandb:
                    log_dict = {
                        "global_step": global_step,
                        "loss/total": loss.item(),
                        "loss/diffusion": diffusion_loss.item(),
                        "loss/recon_l1": recon_loss.item(),
                        "loss/recon_ssim": ssim_loss_val.item(),
                        "loss/physics_weighted": phys_loss_val,
                        "loss/physics_monotonic": physics_dict.get("physics/monotonic", 0.0),
                        "loss/physics_dynamics": physics_dict.get("physics/dynamics", 0.0),
                        "loss/physics_v_eff_mean": physics_dict.get("physics/v_eff_mean", 0.0),
                        "loss/orthogonality": physics_dict.get("physics/orthogonality", 0.0),
                        "loss/contrastive": physics_dict.get("physics/contrastive", 0.0),
                        "physics_weight": physics_weight,
                        "lr": cur_lr,
                    }
                    wandb.log(log_dict, step=global_step)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        if accelerator.is_main_process and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_diff = epoch_diffusion_loss / num_batches
            avg_phys = epoch_physics_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_ssim  = epoch_ssim_loss  / num_batches

            print(
                f"\n{'─'*50}\n"
                f"Epoch {epoch+1}/{args.epochs} Summary  ({epoch_time:.0f}s)\n"
                f"{'─'*50}\n"
                f"  Avg Loss:      {avg_loss:.4f}\n"
                f"  Avg Diffusion: {avg_diff:.4f}\n"
                f"  Avg Recon L1:  {avg_recon:.4f}\n"
                f"  Avg SSIM-loss: {avg_ssim:.4f}\n"
                f"  Avg Physics:   {avg_phys:.4f}\n"
                f"  Physics Weight:{physics_weight:.2f}\n"
            )

            if args.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/avg_loss": avg_loss,
                    "epoch/avg_diffusion": avg_diff,
                    "epoch/avg_recon_l1": avg_recon,
                    "epoch/avg_recon_ssim": avg_ssim,
                    "epoch/avg_physics": avg_phys,
                    "epoch/physics_weight": physics_weight,
                    "epoch/time_seconds": epoch_time,
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # ─── Sample video generation ──────────────────────────────
        if (
            accelerator.is_main_process
            and (epoch + 1) % args.sample_every_epoch == 0
            and sample_image_path is not None
        ):
            print(f"Generating sample videos at RPMs {sample_rpms}...")
            samples_dir = Path(args.output_dir) / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            try:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_mlp = accelerator.unwrap_model(physics_mlp)

                sample_videos = generate_sample_videos(
                    unet=unwrapped_unet,
                    vae=vae,
                    image_encoder=image_encoder,
                    feature_extractor=feature_extractor,
                    physics_mlp=unwrapped_mlp,
                    rpm_film=accelerator.unwrap_model(rpm_film) if rpm_film is not None else None,
                    conditioning_mode=args.conditioning_mode,
                    noise_scheduler_cfg=noise_scheduler_cfg,
                    cond_image_path=sample_image_path,
                    rpms=sample_rpms,
                    num_frames=args.num_frames,
                    num_inference_steps=args.sample_steps,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    output_dir=samples_dir,
                    epoch=epoch + 1,
                    cfg_scale=args.cfg_scale,
                )

                if args.use_wandb:
                    # Log videos to W&B using saved mp4 files (no moviepy needed)
                    video_logs = {}
                    for rpm in sample_rpms:
                        mp4_path = samples_dir / f"epoch{epoch+1:03d}_rpm{rpm}.mp4"
                        if mp4_path.exists():
                            vid = wandb.Video(
                                str(mp4_path),
                                fps=7,
                                caption=f"Epoch {epoch+1}, RPM={rpm}",
                            )
                            video_logs[f"samples/rpm_{rpm}"] = vid

                    # Also log a side-by-side grid image of last frames
                    if sample_videos:
                        last_frames = [sample_videos[r][-1] for r in sample_rpms if r in sample_videos]
                        if last_frames:
                            grid = np.concatenate(last_frames, axis=1)
                            video_logs["samples/last_frame_grid"] = wandb.Image(
                                grid, caption=f"Epoch {epoch+1} | RPMs: {sample_rpms}"
                            )
                    wandb.log(video_logs, step=global_step)

                print(f"Sample videos saved to {samples_dir}")

            except Exception as e:
                import traceback
                print(f"\n{'!'*60}")
                raise RuntimeError("Sample generation failed") from e
                print(f"{'!'*60}")
                traceback.print_exc()
                print(f"{'!'*60}\n")

        # ─── Validation evaluation ────────────────────────────────
        if (
            accelerator.is_main_process
            and (epoch + 1) % args.eval_every_epoch == 0
            and len(val_dataset) > 0
        ):
            print(f"Running validation evaluation ({args.num_val_samples} clips)...")
            try:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_mlp = accelerator.unwrap_model(physics_mlp)

                with torch.random.fork_rng(devices=[accelerator.device.index or 0] if accelerator.device.type == "cuda" else []):
                    torch.manual_seed(args.seed + 10000)
                    val_metrics = validate_epoch(
                    unet=unwrapped_unet,
                    vae=vae,
                    image_encoder=image_encoder,
                    feature_extractor=feature_extractor,
                    physics_mlp=unwrapped_mlp,
                    rpm_film=accelerator.unwrap_model(rpm_film) if rpm_film is not None else None,
                    conditioning_mode=args.conditioning_mode,
                    noise_scheduler_cfg=noise_scheduler_cfg,
                    val_dataset=val_dataset,
                    num_samples=args.num_val_samples,
                    num_frames=args.num_frames,
                    num_inference_steps=args.sample_steps,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    cfg_scale=args.cfg_scale,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    output_dir=args.output_dir,
                    epoch=epoch + 1,
                )

                if args.use_wandb and val_metrics:
                    wandb.log(val_metrics, step=global_step)

            except Exception as e:
                import traceback
                print(f"\n{'!'*60}")
                raise RuntimeError("Validation evaluation failed") from e
                print(f"{'!'*60}")
                traceback.print_exc()
                print(f"{'!'*60}\n")

        # ─── Save best checkpoint ─────────────────────────────────
        if accelerator.is_main_process and num_batches > 0:
            avg_loss_val = epoch_loss / num_batches
            if avg_loss_val < best_loss:
                best_loss = avg_loss_val
                save_dir = Path(args.output_dir) / "best"
                save_dir.mkdir(parents=True, exist_ok=True)

                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(save_dir / "unet_lora")

                unwrapped_mlp = accelerator.unwrap_model(physics_mlp)
                torch.save(unwrapped_mlp.state_dict(), save_dir / "physics_mlp.pt")
                if rpm_film is not None:
                    unwrapped_film = accelerator.unwrap_model(rpm_film)
                    torch.save(unwrapped_film.state_dict(), save_dir / "rpm_film.pt")

                unwrapped_loss = accelerator.unwrap_model(physics_loss_fn)
                torch.save(unwrapped_loss.state_dict(), save_dir / "physics_loss.pt")

                with open(save_dir / "conditioning_config.json", "w") as f:
                    json.dump({"conditioning_mode": args.conditioning_mode}, f, indent=2)

                save_training_state(save_dir, optimizer, epoch, global_step, best_loss, args, scaler=accelerator.scaler)

                print(f"Saved best checkpoint to {save_dir} (loss={best_loss:.4f})")

        if accelerator.is_main_process and (epoch + 1) % args.save_every_epoch == 0:
            save_checkpoint(Path(args.output_dir) / f"epoch_{epoch + 1:03d}", epoch)
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final save (last epoch)
    if accelerator.is_main_process:
        save_dir = Path(args.output_dir) / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(save_dir / "unet_lora")
        unwrapped_mlp = accelerator.unwrap_model(physics_mlp)
        torch.save(unwrapped_mlp.state_dict(), save_dir / "physics_mlp.pt")
        if rpm_film is not None:
            unwrapped_film = accelerator.unwrap_model(rpm_film)
            torch.save(unwrapped_film.state_dict(), save_dir / "rpm_film.pt")
        unwrapped_loss = accelerator.unwrap_model(physics_loss_fn)
        torch.save(unwrapped_loss.state_dict(), save_dir / "physics_loss.pt")
        with open(save_dir / "conditioning_config.json", "w") as f:
            json.dump({"conditioning_mode": args.conditioning_mode}, f, indent=2)
        save_training_state(save_dir, optimizer, epoch, global_step, best_loss, args, scaler=accelerator.scaler)
        print(f"\nTraining complete! Final model saved to {save_dir}")

        if args.use_wandb:
            wandb.run.summary["final_avg_loss"] = avg_loss
            wandb.run.summary["final_avg_diffusion"] = avg_diff
            wandb.run.summary["total_epochs"] = args.epochs
            wandb.run.summary["total_steps"] = global_step

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
