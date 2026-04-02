"""
config.py — Central configuration for the Text Style Transfer VAE.

All hyperparameters live here so nothing is scattered across notebooks.
Adjust values based on your GPU (V100 32GB vs A100 40/80GB).

Architecture overview:
    Input text
    → LLaMA 3.1 8B tokenizer (128K BPE vocab)
    → LLaMA 3.1 8B frozen embedding (128256 × 4096)
    → Projection layer (4096 → d_model=512)
    → Transformer encoder (4 layers, 8 heads)
    → Masked mean pool → content μ/logσ², style μ/logσ²
    → Reparameterize → z_c (128-dim), z_s (32-dim)
    → Concatenate [z_c; z_s] → project to d_model
    → Autoregressive Transformer decoder (4 layers, causal self-attention only)
    → Output projection → vocab logits (128256)

Loss = Reconstruction + β·KL + γ·HSIC(z_c, z_s) + λ·Contrastive(z_s)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    # --- Pretrained model ---
    llama_model_name: str = "meta-llama/Llama-3.1-8B"
    llama_embed_dim: int = 4096          # LLaMA 3.1 8B embedding dimension
    llama_vocab_size: int = 128256       # Actual vocab size (128K + special tokens)

    # --- Projection ---
    # We project LLaMA's 4096-dim embeddings down to d_model to make the
    # transformer layers feasible on a single GPU. Without this, a 6-layer
    # transformer at d_model=4096 would consume ~20GB+ for moderate batch sizes.
    d_model: int = 512                   # Working dimension after projection

    # --- Transformer encoder/decoder ---
    n_heads: int = 8                     # Attention heads (head_dim = 512/8 = 64)
    n_encoder_layers: int = 6            # Encoder depth
    n_decoder_layers: int = 6            # Decoder depth
    dim_feedforward: int = 2048          # FFN hidden dim (4× d_model is standard)
    dropout: float = 0.1                 # Dropout in transformer layers

    # --- Latent space ---
    # Content latent encodes *what* was said (semantics, entities, relationships).
    # Style latent encodes *how* it was said (Shakespeare vs modern).
    # Content needs more capacity because semantic meaning is higher-dimensional.
    # John et al. used content=128, style=8. We use style=32 because
    # Shakespeare style is richer than binary sentiment (archaic vocabulary,
    # inverted syntax, thee/thou pronouns, etc.)
    content_latent_dim: int = 128
    style_latent_dim: int = 32

    # --- Sequence length ---
    max_seq_len: int = 128               # LLaMA BPE produces longer seqs than word-level

    # --- Derived ---
    @property
    def total_latent_dim(self) -> int:
        return self.content_latent_dim + self.style_latent_dim


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    batch_size: int = 32                 # Adjust for your GPU memory
    learning_rate: float = 1e-3          # AdamW learning rate
    weight_decay: float = 0.01           # AdamW weight decay
    max_grad_norm: float = 1.0           # Gradient clipping
    num_epochs: int = 30                 # Total training epochs

    # --- KL annealing ---
    # Without annealing, the KL term dominates early in training and pushes
    # the posterior q(z|x) to match the prior N(0,I) before the decoder has
    # learned to use z. This is "posterior collapse" — z becomes noise and the
    # decoder ignores it. Sigmoid annealing starts with β≈0 (letting the model
    # learn reconstruction first) and slowly increases β to 1.0.
    kl_anneal_steps: int = 20000         # Steps to reach β≈1.0 # doubled 
    kl_anneal_midpoint: int = 10_000       # Step where β=0.5 (inflection point) #doubled 
    kl_anneal_steepness: float = 0.002   # Controls sigmoid sharpness

    # --- Loss weights ---
    # γ (gamma_hsic): Weight for HSIC independence loss.
    #   HSIC measures statistical dependence between z_c and z_s.
    #   Higher γ = stronger push for independence, but too high starves
    #   the model of capacity (can't encode anything if forced to be
    #   perfectly independent).
    gamma_hsic: float = 0.5

    # λ (lambda_contrast): Weight for contrastive style loss.
    #   Pulls same-style z_s vectors together, pushes different-style apart.
    #   Higher λ = tighter style clusters, but too high can cause z_s to
    #   memorize style labels rather than learning a smooth manifold.
    lambda_contrast: float = 0.3 # 1, 0,5, 0.3

    # Temperature for contrastive loss (lower = sharper similarities)
    contrastive_temperature: float = 0.1

    # --- Learning rate schedule ---
    warmup_steps: int = 500              # Linear warmup before cosine decay
    min_lr_ratio: float = 0.01           # Minimum LR as fraction of peak LR

    # --- Logging & checkpointing ---
    log_every: int = 50                  # Log metrics every N steps
    eval_every: int = 500                # Run validation every N steps
    save_every: int = 10_000               # Save checkpoint every N steps


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # --- Paths (relative to project root) ---
    # These point to the Shakespearizing-Modern-English .nltktok files.
    # The data is pre-tokenized at the word level by nltk — we rejoin
    # words into strings and let LLaMA's BPE tokenizer re-tokenize.
    data_dir: str = "/projectnb/scottml/seansal2/data/Shakespearizing-Modern-English/data"

    train_modern: str = "train.modern.nltktok"
    train_original: str = "train.original.nltktok"
    valid_modern: str = "valid.modern.nltktok"
    valid_original: str = "valid.original.nltktok"
    test_modern: str = "test.modern.nltktok"
    test_original: str = "test.original.nltktok"

    # --- Style labels ---
    # 0 = modern English, 1 = Shakespeare
    modern_label: int = 0
    shakespeare_label: int = 1
    num_styles: int = 2

    # --- Checkpoint/output directory ---
    output_dir: str = "checkpoints"

    def get_path(self, filename: str) -> Path:
        return Path(self.data_dir) / filename
