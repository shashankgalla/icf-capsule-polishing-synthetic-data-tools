"""
PhysDiff-Polish model: SVD-xt + LoRA + Disentangled Physics Conditioning.

Wraps the Stable Video Diffusion pipeline with:
1. LoRA adapters on UNet attention layers
2. Disentangled physics encoder with Fourier features for cross-attention
3. Separate tokens for RPM and head-contact slip (S_C), plus an interaction token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from peft import LoraConfig, get_peft_model


class FourierFeatures(nn.Module):
    """
    Sinusoidal positional encoding for continuous scalar values.
    Maps a scalar to a vector of sin/cos features at learned frequencies.
    More interpretable than dense MLP — each frequency band captures a specific scale.
    """

    def __init__(self, num_frequencies=16, output_dim=64):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Learnable frequency bands (initialized as powers of 2)
        freqs = torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        self.register_buffer("init_freqs", 2.0 ** freqs * torch.pi)
        self.freq_scale = nn.Parameter(torch.ones(num_frequencies))
        # Project from 2*num_frequencies (sin+cos) to output_dim
        self.proj = nn.Linear(2 * num_frequencies, output_dim)

    def forward(self, x):
        """
        Args:
            x: [B] or [B, 1] scalar values in [0, 1]
        Returns:
            features: [B, output_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [B, 1]
        freqs = self.init_freqs * self.freq_scale  # [num_frequencies]
        angles = x * freqs.unsqueeze(0)  # [B, num_frequencies]
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, 2*num_freq]
        return self.proj(features)  # [B, output_dim]


class DisentangledPhysicsEncoder(nn.Module):
    """
    Produces disentangled cross-attention tokens for physics conditioning:
      Token 0: RPM — Fourier features encode rotation speed (rpm_norm in [0,1])
      Token 1: Slip — Fourier features encode RAW head-contact slip S_C
      Token 2: Interaction — cross-attention captures joint (RPM, S_C) effects

    Output shape: [B, 3, cross_attn_dim].
    Independent-token output (for disentanglement/contrastive losses): [B, 2, cross_attn_dim].
    """

    def __init__(
        self,
        cross_attn_dim: int = 1024,
        fourier_dim: int = 64,
        hidden_dim: int = 256,
        sc_encoding: str = "fourier",
        num_frequencies_sc: int = 16,
        physics_encoding: str = "disentangled",
    ):
        super().__init__()
        self.cross_attn_dim = cross_attn_dim
        self.sc_encoding = sc_encoding
        self.physics_encoding = physics_encoding

        if physics_encoding == "concat":
            # Single shared MLP that maps [rpm_norm, slip_raw] → one token.
            self.concat_mlp = nn.Sequential(
                nn.Linear(2, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, cross_attn_dim),
            )
            return

        # Disentangled path — independent tokens per scalar.
        # RPM encoder: Fourier features -> MLP -> cross_attn_dim
        self.rpm_fourier = FourierFeatures(num_frequencies=16, output_dim=fourier_dim)
        self.rpm_proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, cross_attn_dim),
        )

        # Slip encoder: either Fourier + MLP, or a bare Linear over the raw scalar.
        if sc_encoding == "fourier":
            self.slip_fourier = FourierFeatures(
                num_frequencies=num_frequencies_sc, output_dim=fourier_dim
            )
            self.slip_proj = nn.Sequential(
                nn.Linear(fourier_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, cross_attn_dim),
            )
        elif sc_encoding == "raw":
            self.slip_proj = nn.Linear(1, cross_attn_dim)
        else:
            raise ValueError(f"Unknown sc_encoding: {sc_encoding}")

        # Interaction token: cross-attention over the two independent tokens.
        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=cross_attn_dim, num_heads=4, batch_first=True,
        )
        self.interaction_norm = nn.LayerNorm(cross_attn_dim)

    def forward(self, physics_params):
        """
        Args:
            physics_params: [B, 2] — (rpm_norm, slip_raw)

        Returns:
            tokens:            [B, 3, cross_attn_dim]  (concat path returns [B, 1, D])
            individual_tokens: [B, 2, cross_attn_dim]  (concat path returns [B, 1, D])
        """
        rpm_norm = physics_params[:, 0]   # [B]
        slip_raw = physics_params[:, 1]   # [B] — RAW S_C

        if self.physics_encoding == "concat":
            tok = self.concat_mlp(physics_params).unsqueeze(1)  # [B, 1, D]
            return tok, tok

        rpm_tok = self.rpm_proj(self.rpm_fourier(rpm_norm))  # [B, D]

        if self.sc_encoding == "fourier":
            slip_tok = self.slip_proj(self.slip_fourier(slip_raw))  # [B, D]
        else:
            slip_tok = self.slip_proj(slip_raw.unsqueeze(-1))  # [B, D]

        indep_tokens = torch.stack([rpm_tok, slip_tok], dim=1)  # [B, 2, D]

        # Interaction token via cross-attention (query = mean of tokens)
        query = indep_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]
        interaction_tok, _ = self.interaction_attn(query, indep_tokens, indep_tokens)
        interaction_tok = self.interaction_norm(interaction_tok + query)  # [B, 1, D]

        all_tokens = torch.cat(
            [rpm_tok.unsqueeze(1), slip_tok.unsqueeze(1), interaction_tok], dim=1
        )  # [B, 3, D]

        return all_tokens, indep_tokens


class RPMFiLM(nn.Module):
    """
    Lightweight FiLM conditioner using RPM only.
    Produces per-channel gamma/beta to modulate image embedding tokens.
    """

    def __init__(self, embed_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * embed_dim),
        )

    def forward(self, image_embeddings, rpm_norm):
        """
        Args:
            image_embeddings: [B, N, D]
            rpm_norm: [B] normalized RPM in [0,1]
        Returns:
            modulated embeddings: [B, N, D]
        """
        if rpm_norm.dim() == 1:
            rpm_norm = rpm_norm.unsqueeze(1)
        film = self.net(rpm_norm.float())  # [B, 2D]
        gamma, beta = torch.chunk(film, 2, dim=1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return image_embeddings * (1.0 + gamma) + beta


def build_conditioning_hidden_states(
    image_embeddings,
    physics_params,
    conditioning_mode="cross_attn",
    physics_mlp=None,
    rpm_film=None,
    weight_dtype=torch.float16,
):
    """
    Build encoder hidden states for one of:
      - cross_attn: CLIP image token + physics tokens
      - film: FiLM-modulated CLIP image token (RPM only)
      - image_only: CLIP image token only
    Returns:
      encoder_hidden_states: [B, N, D]
      indep_physics_tokens: [B, M, D] or None
    """
    mode = conditioning_mode.lower()

    if mode == "image_only":
        return image_embeddings, None

    if mode == "film":
        if rpm_film is None:
            raise ValueError("conditioning_mode=film requires rpm_film module")
        rpm_norm = physics_params[:, 0]
        mod = rpm_film(image_embeddings.float(), rpm_norm).to(dtype=weight_dtype)
        return mod, None

    if mode == "cross_attn":
        if physics_mlp is None:
            raise ValueError("conditioning_mode=cross_attn requires physics_mlp")
        physics_embeddings, indep_tokens = physics_mlp(physics_params.float())
        physics_embeddings = physics_embeddings.to(dtype=weight_dtype)
        return prepare_encoder_hidden_states(image_embeddings, physics_embeddings), indep_tokens

    raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")


class PhysicsCLIPContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss ensuring each physics token faithfully
    encodes its corresponding parameter. Creates positive/negative pairs
    within a batch based on parameter similarity.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, tokens, physics_params):
        """
        Args:
            tokens: [B, 2, D] — the independent physics tokens (rpm, slip)
            physics_params: [B, 2] — (rpm_norm, slip_raw)

        Returns:
            loss: scalar contrastive loss on the RPM token.
        """
        B = tokens.shape[0]
        if B < 2 or tokens.shape[1] < 1:
            return torch.tensor(0.0, device=tokens.device)

        # RPM token (continuous): soft-target InfoNCE keyed by RPM proximity.
        rpm_embeds = F.normalize(tokens[:, 0], dim=1)  # [B, D]
        rpm_vals = physics_params[:, 0]                # [B]

        embed_sim = rpm_embeds @ rpm_embeds.T / self.temperature
        rpm_diff = (rpm_vals.unsqueeze(1) - rpm_vals.unsqueeze(0)).abs()
        target_sim = torch.exp(-rpm_diff * 5.0)
        target_sim = target_sim / target_sim.sum(dim=1, keepdim=True)

        log_pred = F.log_softmax(embed_sim, dim=1)
        return (-target_sim * log_pred).sum(dim=1).mean()


def orthogonality_loss(tokens):
    """
    Encourage the independent physics tokens to be orthogonal (disentangled).
    Handles 3 tokens (RPM, surface, FPS) or 4 tokens (+ slip).

    Args:
        tokens: [B, N, D] — independent physics tokens (N=3 or N=4)

    Returns:
        loss: scalar
    """
    N = tokens.shape[1]
    normed = F.normalize(tokens, dim=2)  # [B, N, D]
    sim = torch.bmm(normed, normed.transpose(1, 2))  # [B, N, N]
    eye = torch.eye(N, device=tokens.device).unsqueeze(0)
    return ((sim - eye) ** 2).mean()


# Keep legacy class for checkpoint compatibility
class PhysicsEmbeddingMLP(nn.Module):
    """Legacy MLP encoder (kept for loading old checkpoints)."""

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1024, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim * num_tokens),
        )
        self.output_dim = output_dim

    def forward(self, physics_params):
        out = self.mlp(physics_params)
        return out.reshape(-1, self.num_tokens, self.output_dim)


def create_model(
    pretrained_model_name="stabilityai/stable-video-diffusion-img2vid-xt",
    lora_rank=32,
    lora_alpha=32,
    physics_embed_dim=1024,
    physics_num_tokens=4,
    device="cuda",
    dtype=torch.float16,
    sc_encoding: str = "fourier",
    num_frequencies_sc: int = 16,
    physics_encoding: str = "disentangled",
    method_profile: str = "legacy_supplied",
):
    """
    Create the PhysDiff-Polish model.

    Returns:
        pipeline: SVD pipeline (for inference convenience)
        unet: UNet with LoRA adapters
        vae: frozen VAE
        physics_mlp: physics parameter embedding
        image_encoder: frozen CLIP image encoder
        feature_extractor: image preprocessor
    """
    print("Loading SVD-xt pipeline...")
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=dtype,
        variant="fp16",
    )

    unet = pipeline.unet
    vae = pipeline.vae
    image_encoder = pipeline.image_encoder
    feature_extractor = pipeline.feature_extractor
    scheduler = pipeline.scheduler

    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Apply LoRA to UNet attention layers
    print(f"Applying LoRA (rank={lora_rank})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=(r".*(?:attn1\.to_q|(?:attn1|attn2)\.(?:to_k|to_v|to_out\.0))$" if method_profile == "paper_declared" else ["to_q", "to_k", "to_v", "to_out.0"]),
        lora_dropout=0.05,   # reduced from 0.1 — small dataset benefits from less dropout
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Create disentangled physics encoder (replaces monolithic MLP)
    physics_mlp = DisentangledPhysicsEncoder(
        cross_attn_dim=physics_embed_dim,
        sc_encoding=sc_encoding,
        num_frequencies_sc=num_frequencies_sc,
        physics_encoding=physics_encoding,
    )

    if method_profile == "paper_declared":
        for name, parameter in physics_mlp.named_parameters():
            if name.endswith("freq_scale"):
                parameter.requires_grad_(False)

    return {
        "pipeline": pipeline,
        "unet": unet,
        "vae": vae,
        "image_encoder": image_encoder,
        "feature_extractor": feature_extractor,
        "scheduler": scheduler,
        "physics_mlp": physics_mlp,
    }


def encode_image_for_svd(image_encoder, feature_extractor, images, device, dtype):
    """
    Encode conditioning images using CLIP for SVD.

    Args:
        images: [B, C, H, W] tensor in [0, 1]

    Returns:
        image_embeddings: [B, 1, 1024] CLIP embeddings
    """
    # SVD expects CLIP image embeddings
    # The pipeline internally does this, but we need it for training
    from transformers import CLIPImageProcessor
    import numpy as np

    # Convert to PIL for feature_extractor
    images_np = (images.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    inputs = feature_extractor(
        images=[img for img in images_np],
        return_tensors="pt",
    ).pixel_values.to(device=device, dtype=dtype)

    image_embeddings = image_encoder(inputs).image_embeds
    image_embeddings = image_embeddings.unsqueeze(1)  # [B, 1, 1024]

    return image_embeddings


def prepare_encoder_hidden_states(image_embeddings, physics_embeddings):
    """
    Concatenate CLIP image embeddings with physics parameter embeddings
    for cross-attention conditioning.

    Args:
        image_embeddings: [B, 1, 1024] from CLIP
        physics_embeddings: [B, num_tokens, 1024] from PhysicsEmbeddingMLP

    Returns:
        encoder_hidden_states: [B, 1 + num_tokens, 1024]
    """
    return torch.cat([image_embeddings, physics_embeddings], dim=1)


def get_added_time_ids(fps=7, motion_bucket_id=127, noise_aug_strength=0.02, batch_size=1, dtype=torch.float16, device="cuda"):
    """Prepare added_time_ids for SVD UNet."""
    add_time_ids = torch.tensor(
        [fps, motion_bucket_id, noise_aug_strength],
        dtype=dtype, device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    return add_time_ids


if __name__ == "__main__":
    # Test DisentangledPhysicsEncoder
    print("Testing DisentangledPhysicsEncoder...")
    encoder = DisentangledPhysicsEncoder(cross_attn_dim=1024)

    params = torch.tensor([[0.5, 0.0, 0.3], [1.0, 1.0, 0.6]])  # [B=2, 3]
    all_tokens, indep_tokens = encoder(params)
    print(f"Input shape: {params.shape}")
    print(f"All tokens shape: {all_tokens.shape}")      # [2, 4, 1024]
    print(f"Independent tokens shape: {indep_tokens.shape}")  # [2, 3, 1024]

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"DisentangledPhysicsEncoder parameters: {total_params:,}")

    # Test orthogonality loss
    orth_loss = orthogonality_loss(indep_tokens)
    print(f"Orthogonality loss: {orth_loss.item():.6f}")

    # Test contrastive loss
    clip_loss_fn = PhysicsCLIPContrastiveLoss()
    clip_loss = clip_loss_fn(indep_tokens, params)
    print(f"Contrastive loss: {clip_loss.item():.6f}")

    # Test that gradients flow
    params_grad = params.clone().requires_grad_(True)
    tokens, indep = encoder(params_grad)
    (tokens.sum() + orthogonality_loss(indep)).backward()
    print(f"Gradients flow: {params_grad.grad is not None}")
