"""
train.py — Training loop for the Text Style Transfer VAE.

This is the main entry point. Run with:
    python train.py

Or from a notebook/SLURM script:
    from train import train_model
    train_model()

Training procedure:
═══════════════════
1. Load LLaMA tokenizer + frozen embeddings
2. Build non-parallel StyleDataset (modern label=0, Shakespeare label=1)
3. For each epoch:
   a. For each batch of mixed-style sentences:
      - Tokenize on the fly (in collate_fn)
      - Forward pass: encode → reparameterize → decode (teacher forcing)
      - Compute total loss = recon + β·KL + γ·HSIC + λ·contrastive
      - Backward pass + gradient clipping + optimizer step
   b. Run validation loss
   c. Save checkpoint if improved
4. After final epoch: compute and save average style embeddings (for inference)

The average style embeddings are the key to inference:
  - Collect z_s for all training sentences
  - Average z_s for modern sentences → avg_modern_style
  - Average z_s for Shakespeare sentences → avg_shakespeare_style
  - At inference: encode modern sentence → z_c, replace z_s with avg_shakespeare_style → decode

This follows the same approach as John et al.'s avg_style_emb dict.
"""

import os
import time
from pathlib import Path

# Redirect HuggingFace cache to project space to avoid home dir quota.
# /projectnb has a much larger group quota than ~/.cache (~16GB home limit).
_hf_cache = "/projectnb/scottml/seansal2/.cache/huggingface"
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, DataConfig
from model import build_model, VAETransformer
from data import build_dataloaders
from losses import compute_total_loss


# ──────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────

def get_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """
    Returns a lambda for torch.optim.lr_scheduler.LambdaLR.

    Phase 1 (step < warmup_steps): linear warmup from 0 to 1
    Phase 2 (step >= warmup_steps): cosine decay from 1 to min_lr_ratio

    This is the standard schedule used in most transformer training.
    Warmup prevents early gradient explosions when the model is randomly
    initialized and gradients are large.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + __import__("math").cos(progress * 3.14159))

    return lr_lambda


# ──────────────────────────────────────────────────────────
# Compute average style embeddings (for inference)
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_avg_style_embeddings(
    model: VAETransformer,
    dataloader: DataLoader,
    device: torch.device,
    num_styles: int = 2,
) -> dict[int, torch.Tensor]:
    """
    After training, compute the average z_s for each style class.

    These averages serve as "prototype" style vectors for inference.
    To transfer a modern sentence to Shakespeare:
      1. Encode modern sentence → z_c, z_s
      2. Replace z_s with avg_shakespeare_style
      3. Decode [z_c; avg_shakespeare_style] → Shakespeare output

    Returns:
        dict mapping style_label → average z_s tensor (style_latent_dim,)
    """
    model.eval()

    style_sums = {i: None for i in range(num_styles)}
    style_counts = {i: 0 for i in range(num_styles)}

    print("Computing average style embeddings...")
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        doc_ids = batch["doc_ids"]

        # Encode to get style latent
        c_mu, c_log_var, s_mu, s_log_var = model.encode(input_ids, attention_mask)
        # Use the mean (not sampled) for more stable prototypes
        z_s = s_mu

        for style_label in range(num_styles):
            mask = (doc_ids == style_label)
            if mask.any():
                style_vectors = z_s[mask]  # (num_matched, style_latent_dim)
                if style_sums[style_label] is None:
                    style_sums[style_label] = style_vectors.sum(dim=0)
                else:
                    style_sums[style_label] += style_vectors.sum(dim=0)
                style_counts[style_label] += mask.sum().item()

    avg_style_embs = {}
    for style_label in range(num_styles):
        if style_counts[style_label] > 0:
            avg_style_embs[style_label] = (
                style_sums[style_label] / style_counts[style_label]
            ).cpu()
            print(f"  Style {style_label}: averaged over {style_counts[style_label]} sentences")
        else:
            print(f"  WARNING: No sentences found for style {style_label}")

    return avg_style_embs


# ──────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: VAETransformer,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    global_step: int,
    train_config: TrainConfig,
) -> dict:
    """Run validation and return average metrics."""
    model.eval()
    total_metrics = {}
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        doc_ids = batch["doc_ids"].to(device)

        outputs = model(input_ids, attention_mask)

        _, metrics = compute_total_loss(
            logits=outputs["logits"],
            tgt_output=outputs["tgt_output"],
            pad_token_id=pad_token_id,
            c_mu=outputs["c_mu"],
            c_log_var=outputs["c_log_var"],
            s_mu=outputs["s_mu"],
            s_log_var=outputs["s_log_var"],
            z_c=outputs["z_c"],
            z_s=outputs["z_s"],
            doc_ids=doc_ids,
            global_step=global_step,
            gamma_hsic=train_config.gamma_hsic,
            lambda_contrast=train_config.lambda_contrast,
            contrastive_temperature=train_config.contrastive_temperature,
            kl_anneal_midpoint=train_config.kl_anneal_midpoint,
            kl_anneal_steepness=train_config.kl_anneal_steepness,
        )

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

    # Average
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_metrics


# ──────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────

def train_model(
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    data_config: DataConfig | None = None,
):
    """
    Full training pipeline.

    Can be called with custom configs or uses defaults.
    """
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()
    if data_config is None:
        data_config = DataConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Tokenizer ───
    print(f"\nLoading tokenizer: {model_config.llama_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # ─── Model ───
    model = build_model(model_config, device)

    # ─── Data ───
    print("\nBuilding dataloaders...")
    loaders = build_dataloaders(
        data_config=data_config,
        model_config=model_config,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
    )
    train_loader = loaders["train"]
    valid_loader = loaders["valid"]

    # ─── Optimizer ───
    # Only optimize trainable parameters (frozen embeddings are excluded)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # ─── LR scheduler ───
    total_steps = len(train_loader) * train_config.num_epochs
    lr_lambda = get_lr_lambda(
        train_config.warmup_steps, total_steps, train_config.min_lr_ratio
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ─── Output directory ───
    output_dir = Path(data_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Training loop ───
    print(f"\nStarting training for {train_config.num_epochs} epochs")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  KL annealing midpoint: step {train_config.kl_anneal_midpoint}")
    print()

    global_step = 0
    best_valid_loss = float("inf")

    for epoch in range(train_config.num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_metrics = {}
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_ids = batch["doc_ids"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Compute loss
            loss, metrics = compute_total_loss(
                logits=outputs["logits"],
                tgt_output=outputs["tgt_output"],
                pad_token_id=pad_token_id,
                c_mu=outputs["c_mu"],
                c_log_var=outputs["c_log_var"],
                s_mu=outputs["s_mu"],
                s_log_var=outputs["s_log_var"],
                z_c=outputs["z_c"],
                z_s=outputs["z_s"],
                doc_ids=doc_ids,
                global_step=global_step,
                gamma_hsic=train_config.gamma_hsic,
                lambda_contrast=train_config.lambda_contrast,
                contrastive_temperature=train_config.contrastive_temperature,
                kl_anneal_midpoint=train_config.kl_anneal_midpoint,
                kl_anneal_steepness=train_config.kl_anneal_steepness,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Accumulate epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            num_batches += 1

            # ─── Periodic logging ───
            if global_step % train_config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step:>6d} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"recon={metrics['recon']:.4f} | "
                    f"kl={metrics['kl_total']:.4f} (β={metrics['beta']:.3f}) | "
                    f"hsic={metrics['hsic']:.4f} | "
                    f"contrast={metrics['contrast']:.4f} | "
                    f"lr={lr:.2e}"
                )

            # ─── Periodic validation ───
            if global_step > 0 and global_step % train_config.eval_every == 0:
                valid_metrics = validate(
                    model, valid_loader, device, pad_token_id,
                    global_step, train_config,
                )
                print(
                    f"  [VALID] step {global_step} | "
                    f"loss={valid_metrics['loss']:.4f} | "
                    f"recon={valid_metrics['recon']:.4f} | "
                    f"kl={valid_metrics['kl_total']:.4f}"
                )

                # Save best model
                if valid_metrics["loss"] < best_valid_loss:
                    best_valid_loss = valid_metrics["loss"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_valid_loss": best_valid_loss,
                        "model_config": model_config,
                        "train_config": train_config,
                    }, output_dir / "best_model.pt")
                    print(f"  → Saved best model (valid_loss={best_valid_loss:.4f})")

                model.train()  # Back to training mode

            # ─── Periodic checkpoint ───
            if global_step > 0 and global_step % train_config.save_every == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_config": model_config,
                    "train_config": train_config,
                }, output_dir / f"checkpoint_step{global_step}.pt")

            global_step += 1

        # End-of-epoch summary
        epoch_time = time.time() - epoch_start
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        print(
            f"\nEpoch {epoch + 1}/{train_config.num_epochs} "
            f"({epoch_time:.1f}s) | "
            f"avg_loss={avg_metrics['loss']:.4f} | "
            f"avg_recon={avg_metrics['recon']:.4f} | "
            f"avg_kl={avg_metrics['kl_total']:.4f}"
        )

    # ─── Post-training: compute average style embeddings ───
    print("\n" + "=" * 60)
    print("Training complete. Computing average style embeddings...")
    avg_style_embs = compute_avg_style_embeddings(
        model, train_loader, device, num_styles=data_config.num_styles,
    )

    # Save everything needed for inference
    torch.save({
        "model_state_dict": model.state_dict(),
        "avg_style_embs": avg_style_embs,
        "model_config": model_config,
        "train_config": train_config,
        "data_config": data_config,
        "global_step": global_step,
    }, output_dir / "final_model.pt")
    print(f"Saved final model + style embeddings to {output_dir / 'final_model.pt'}")

    return model, avg_style_embs


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()
