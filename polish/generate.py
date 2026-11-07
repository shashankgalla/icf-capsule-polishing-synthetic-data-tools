"""
Inference / video generation script for PhysDiff-Polish.

Generates polishing videos conditioned on RPM and shell count.
Supports RPM sweep, interpolation, and rectangular output.
"""

import argparse
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from diffusers import StableVideoDiffusionPipeline, DDIMScheduler
from peft import PeftModel

from .model import (
    DisentangledPhysicsEncoder,
    PhysicsEmbeddingMLP,
    RPMFiLM,
    build_conditioning_hidden_states,
    prepare_encoder_hidden_states,
    encode_image_for_svd,
    get_added_time_ids,
)

from .dataset import PolishingVideoDataset


def load_model(checkpoint_dir, pretrained_model, device="cuda", dtype=torch.float16):
    """Load trained model from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)

    # Load base pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model, torch_dtype=dtype, variant="fp16"
    )

    # Load LoRA weights
    lora_dir = checkpoint_dir / "unet_lora"
    if lora_dir.exists():
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(lora_dir))
        print(f"Loaded LoRA weights from {lora_dir}")

    cfg_path = checkpoint_dir / "conditioning_config.json"
    if not cfg_path.is_file() or not lora_dir.is_dir():
        raise FileNotFoundError("Checkpoint requires adapters and conditioning_config.json")
    config = json.loads(cfg_path.read_text())
    required = {"conditioning_mode", "sc_encoding", "num_frequencies_sc", "physics_encoding", "method_profile"}
    if not required <= config.keys():
        raise ValueError("Incomplete checkpoint configuration; legacy checkpoints require an explicit migration")
    conditioning_mode = config["conditioning_mode"]
    physics_mlp = DisentangledPhysicsEncoder(cross_attn_dim=1024,
        sc_encoding=config["sc_encoding"], num_frequencies_sc=config["num_frequencies_sc"], physics_encoding=config["physics_encoding"])
    mlp_path = checkpoint_dir / "physics_mlp.pt"
    if not mlp_path.is_file():
        raise FileNotFoundError("Missing physics encoder weights")
    physics_mlp.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True), strict=True)
    physics_mlp.release_config = config

    pipeline = pipeline.to(device)
    physics_mlp = physics_mlp.to(device, dtype=torch.float32)
    physics_mlp.eval()

    rpm_film = None
    film_path = checkpoint_dir / "rpm_film.pt"
    if conditioning_mode == "film" and film_path.exists():
        rpm_film = RPMFiLM(embed_dim=1024)
        rpm_film.load_state_dict(torch.load(str(film_path), map_location="cpu", weights_only=True))
        rpm_film = rpm_film.to(device, dtype=torch.float32)
        rpm_film.eval()
        print(f"Loaded RPMFiLM from {film_path}")

    # Attach mode info for evaluation scripts without changing return signature.
    pipeline.conditioning_mode = conditioning_mode
    pipeline.rpm_film = rpm_film

    pipeline.unet.eval()
    return pipeline, physics_mlp


def generate_video_manual(
    unet, vae, image_encoder, feature_extractor, physics_mlp,
    conditioning_mode,
    rpm_film,
    cond_image_path, rpm,
    num_frames=30, num_inference_steps=25,
    output_width=512, output_height=128,
    device="cuda", dtype=torch.float16,
    cfg_scale=3.0,
    slip_ratio=0.95,
    # Deprecated kwargs kept for backward-compat with old callers; ignored.
    shell_count=None,
):
    """
    Generate a video using manual denoising loop (not pipeline).
    This properly injects physics conditioning via cross-attention.
    """
    import imageio
    from PIL import Image

    unet.eval()
    physics_mlp.eval()

    # Load & resize conditioning image
    # Full-frame resizing matches the default synthetic training preprocessing.
    cond_pil = Image.open(cond_image_path).convert("RGB")
    cond_arr = np.array(cond_pil)
    cond_np = cv2.resize(cond_arr, (output_width, output_height),
                         interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    cond_tensor = torch.from_numpy(cond_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # CLIP encode
    image_embeddings = encode_image_for_svd(
        image_encoder, feature_extractor, cond_tensor, device, dtype
    )

    # VAE encode conditioning image
    cond_latent = vae.encode(cond_tensor.to(dtype)).latent_dist.mode()
    cond_latent = cond_latent * vae.config.scaling_factor

    # Physics conditioning: (rpm_norm, raw S_C). Slip is passed RAW (not normalized).
    config = getattr(physics_mlp, "release_config", {})
    low, high = (21., 108.) if config.get("method_profile") == "paper_declared" else (10., 40.)
    if not low <= rpm <= high or not 0 <= slip_ratio <= 1:
        raise ValueError("Requested process condition outside the declared range")
    rpm_norm = (rpm - low) / (high - low)
    slip_raw = float(slip_ratio)
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
        weight_dtype=dtype,
    )
    added_time_ids = get_added_time_ids(batch_size=1, dtype=dtype, device=device)

    # DDIM scheduler
    infer_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    infer_scheduler.set_timesteps(num_inference_steps, device=device)

    # Repeat cond_latent across frames
    cond_lat_t = cond_latent.unsqueeze(1).expand(1, num_frames, -1, -1, -1).float()

    # Start from noise
    latent_shape = (1, num_frames, cond_latent.shape[1],
                    cond_latent.shape[2], cond_latent.shape[3])
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        for t in infer_scheduler.timesteps:
            unet_input = torch.cat([latents, cond_lat_t], dim=2)
            ts = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)

            # Conditional prediction
            noise_pred_cond = unet(
                unet_input.to(dtype), ts,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
            ).sample.float()

            # Classifier-Free Guidance
            if cfg_scale != 1.0:
                uncond_hidden = torch.zeros_like(encoder_hidden_states)
                noise_pred_uncond = unet(
                    unet_input.to(dtype), ts,
                    encoder_hidden_states=uncond_hidden,
                    added_time_ids=added_time_ids,
                ).sample.float()
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            latents = infer_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        lat_flat = latents.reshape(-1, *latents.shape[2:])
        decoded = vae.decode(
            lat_flat.to(dtype) / vae.config.scaling_factor,
            num_frames=num_frames,
        ).sample
        decoded = decoded.float().clamp(0, 1)

    frames_np = (decoded.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return frames_np


def generate_interpolation(
    unet, vae, image_encoder, feature_extractor, physics_mlp,
    conditioning_mode, rpm_film,
    cond_image_path, param_name, start_val, end_val, num_steps=5,
    num_frames=30, num_inference_steps=25,
    output_width=512, output_height=128,
    device="cuda", dtype=torch.float16, output_dir="interpolation",
    cfg_scale=3.0,
):
    """
    Interpolate a single physics parameter while holding others fixed.
    Demonstrates disentanglement of the conditioning tokens.
    """
    import imageio
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = np.linspace(start_val, end_val, num_steps)

    for i, val in enumerate(values):
        if param_name == "rpm":
            rpm = val
            shell_count = 3.0
            slip = 0.05
        elif param_name == "slip":
            rpm = 30.0
            shell_count = 3.0
            slip = val
        else:
            rpm = 30.0
            shell_count = 3.0
            slip = 0.05

        print(f"  Interpolation step {i+1}/{num_steps}: {param_name}={val:.3f}")
        frames = generate_video_manual(
            unet, vae, image_encoder, feature_extractor, physics_mlp,
            conditioning_mode=conditioning_mode,
            rpm_film=rpm_film,
            cond_image_path=cond_image_path, rpm=rpm, shell_count=shell_count,
            num_frames=num_frames, num_inference_steps=num_inference_steps,
            output_width=output_width, output_height=output_height,
            device=device, dtype=dtype,
            cfg_scale=cfg_scale,
            slip_ratio=slip,
        )

        out_path = output_dir / f"interp_{param_name}_{val:.1f}.mp4"
        writer = imageio.get_writer(str(out_path), fps=7, codec="libx264")
        for f in frames:
            writer.append_data(f)
        writer.close()

    print(f"Interpolation videos saved to {output_dir}")


def save_video(frames_np, output_path, fps=7):
    """Save numpy frames [T, H, W, 3] as MP4."""
    import imageio
    output_path = str(output_path)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()
    print(f"Saved video to {output_path}")


def generate_rpm_sweep(
    unet, vae, image_encoder, feature_extractor, physics_mlp,
    conditioning_mode, rpm_film,
    cond_image_path, output_dir,
    rpms=[10, 20, 30, 40], **kwargs,
):
    """Generate videos at different RPMs for comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rpm in rpms:
        print(f"\nGenerating video at RPM={rpm}...")
        frames = generate_video_manual(
            unet, vae, image_encoder, feature_extractor, physics_mlp,
            conditioning_mode=conditioning_mode,
            rpm_film=rpm_film,
            cond_image_path=cond_image_path, rpm=rpm, **kwargs,
        )
        save_video(frames, output_dir / f"rpm_{rpm}.mp4")

    print(f"\nAll videos saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str,
                        default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to conditioning image")
    parser.add_argument("--rpm", type=float, default=30)
    parser.add_argument("--shell_count", type=float, default=3.0)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--output_width", type=int, default=512)
    parser.add_argument("--output_height", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--rpm_sweep", action="store_true",
                        help="Generate videos at all RPMs")
    parser.add_argument("--interpolate", action="store_true",
                        help="Generate RPM interpolation videos")
    parser.add_argument("--interp_start", type=float, default=10.0)
    parser.add_argument("--interp_end", type=float, default=40.0)
    parser.add_argument("--interp_steps", type=int, default=4)
    args = parser.parse_args()

    pipeline, physics_mlp = load_model(args.checkpoint, args.pretrained_model)

    unet = pipeline.unet
    vae = pipeline.vae
    image_encoder = pipeline.image_encoder
    feature_extractor = pipeline.feature_extractor
    conditioning_mode = getattr(pipeline, "conditioning_mode", "cross_attn")
    rpm_film = getattr(pipeline, "rpm_film", None)
    device = "cuda"
    dtype = torch.float16

    common_kwargs = dict(
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        output_width=args.output_width,
        output_height=args.output_height,
        device=device,
        dtype=dtype,
    )

    if args.interpolate:
        generate_interpolation(
            unet, vae, image_encoder, feature_extractor, physics_mlp,
            conditioning_mode, rpm_film,
            args.image, param_name="rpm",
            start_val=args.interp_start, end_val=args.interp_end,
            num_steps=args.interp_steps,
            output_dir=Path(args.output).parent / "interpolation",
            **common_kwargs,
        )
    elif args.rpm_sweep:
        generate_rpm_sweep(
            unet, vae, image_encoder, feature_extractor, physics_mlp,
            conditioning_mode, rpm_film,
            args.image,
            output_dir=Path(args.output).parent / "rpm_sweep",
            **common_kwargs,
        )
    else:
        frames = generate_video_manual(
            unet, vae, image_encoder, feature_extractor, physics_mlp,
            conditioning_mode=conditioning_mode,
            rpm_film=rpm_film,
            cond_image_path=args.image, rpm=args.rpm, shell_count=args.shell_count,
            **common_kwargs,
        )
        save_video(frames, args.output)
