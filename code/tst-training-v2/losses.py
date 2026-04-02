"""
losses.py — Loss functions for the Text Style Transfer VAE.

Total loss = L_recon + β(step)·(KL_c + KL_s) + γ·HSIC(z_c, z_s) + λ·Contrastive(z_s)

Each component explained:
═════════════════════════

1. RECONSTRUCTION LOSS (L_recon)
   Cross-entropy between predicted next-token logits and ground truth.
   This is the standard language modeling loss — it teaches the decoder
   to produce coherent text. Padding tokens are ignored.

2. KL DIVERGENCE (KL_c + KL_s)
   Pushes the learned posterior q(z|x) toward the prior N(0, I).
   Without this, the encoder could learn a delta function (zero variance)
   at a different location for each input — making the latent space
   non-continuous and useless for generation.

   β is annealed via a sigmoid schedule to prevent posterior collapse:
   early in training β≈0, so the model focuses on reconstruction;
   β gradually increases to 1.0, adding the KL regularization.

3. HSIC (Hilbert-Schmidt Independence Criterion)
   Measures statistical dependence between z_c and z_s using kernel
   methods. Minimizing HSIC pushes the content and style latent spaces
   toward independence. This replaces John et al.'s adversarial losses.

   Advantage over adversarial: single differentiable loss, no min-max
   game, no separate optimizer groups, more stable training.

   Disadvantage: only measures kernel-based dependence, not arbitrary
   nonlinear relationships. In practice, this is sufficient for the
   Shakespeare/modern distinction.

4. CONTRASTIVE STYLE LOSS
   InfoNCE-style loss on z_s. For each sentence, all other sentences
   with the SAME style label are positives, and sentences with different
   style labels are negatives. This clusters same-style z_s vectors
   together and separates different-style vectors.

   Replaces John et al.'s multitask style classifier. The contrastive
   formulation is more flexible and doesn't require a fixed classification
   head.

What's NOT included (compared to John et al.):
   - Content multitask loss (BoW prediction from z_c). Could be added
     later if content preservation is poor.
   - Adversarial losses. Replaced by HSIC.

Kept from professor's transformer_vae_original.ipynb:
   - All four loss functions (reconstruction, KL+annealing, HSIC, contrastive)
   - Fixed: stray 'b' character removed, consistent naming
"""

import math
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# 1. Reconstruction loss
# ──────────────────────────────────────────────────────────

def reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Token-level cross-entropy loss, ignoring padding positions.

    Args:
        logits:       (batch, seq_len, vocab_size) — predicted next-token logits
        targets:      (batch, seq_len)             — ground truth token IDs
        pad_token_id: Token ID used for padding (excluded from loss)

    Returns:
        Scalar loss (mean over non-padding tokens)
    """
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )
    return loss


# ──────────────────────────────────────────────────────────
# 2. KL divergence with sigmoid annealing
# ──────────────────────────────────────────────────────────

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I)) for a diagonal Gaussian posterior.

    Derivation:
        KL = -0.5 * Σ (1 + log(σ²) - μ² - σ²)
           = -0.5 * Σ (1 + log_var - mu² - exp(log_var))

    Averaged over the batch.

    Args:
        mu:      (batch, latent_dim) — posterior mean
        log_var: (batch, latent_dim) — posterior log-variance

    Returns:
        Scalar KL divergence
    """
    return -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    )


def kl_anneal_factor(step: int, midpoint: int = 5000, steepness: float = 0.002) -> float:
    """
    Sigmoid KL annealing schedule.

    Returns a weight β ∈ [0, 1] that multiplies the KL loss.

    Early in training (step << midpoint): β ≈ 0, model focuses on reconstruction.
    At step = midpoint: β = 0.5.
    Late in training (step >> midpoint): β ≈ 1.0, full KL regularization.

    This prevents posterior collapse — the phenomenon where the decoder
    learns to ignore z because the KL penalty pushes z to pure noise
    before the decoder has learned to use it.

    The sigmoid shape is:
        β(step) = σ(steepness · (step - midpoint))
                = 1 / (1 + exp(-steepness · (step - midpoint)))

    Args:
        step:      Current global training step
        midpoint:  Step where β=0.5 (inflection point)
        steepness: Controls how sharply β transitions from 0 to 1

    Returns:
        Float in [0, 1]
    """
    return 1.0 / (1.0 + math.exp(-steepness * (step - midpoint)))


# ──────────────────────────────────────────────────────────
# 3. HSIC (Hilbert-Schmidt Independence Criterion)
# ──────────────────────────────────────────────────────────

def rbf_kernel(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Radial Basis Function (Gaussian) kernel matrix.

    K(x_i, x_j) = exp(-||x_i - x_j||² / (2σ²))

    If sigma is None, uses the median heuristic: σ = sqrt(median_distance / 2).
    This adaptive bandwidth ensures the kernel is sensitive to the actual
    scale of the data.

    Args:
        x:     (B, D) — batch of vectors
        sigma: Kernel bandwidth (None = median heuristic)

    Returns:
        K: (B, B) — kernel matrix
    """
    # Pairwise squared Euclidean distances
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)      # (B, 1)
    dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())  # (B, B)
    dist = dist.clamp(min=0.0)  # Numerical stability

    if sigma is None:
        # Median heuristic: exclude diagonal (distance to self = 0)
        mask = ~torch.eye(dist.size(0), dtype=torch.bool, device=x.device)
        median_dist = torch.median(dist[mask])
        sigma = torch.sqrt(0.5 * median_dist).clamp(min=1e-5)

    K = torch.exp(-dist / (2.0 * sigma ** 2 + 1e-8))
    return K


def hsic_loss(
    z_content: torch.Tensor,
    z_style: torch.Tensor,
) -> torch.Tensor:
    """
    HSIC (Hilbert-Schmidt Independence Criterion) between z_c and z_s.

    HSIC is zero if and only if z_c and z_s are independent (in the RKHS
    induced by the chosen kernel). It's always non-negative, so minimizing
    it pushes the two spaces toward independence.

    Formula (biased estimator):
        HSIC = (1/(B-1)²) · trace(K·H·L·H)
    where:
        K = kernel matrix of z_content
        L = kernel matrix of z_style
        H = centering matrix = I - (1/B)·11ᵀ

    This replaces John et al.'s adversarial losses (style discriminator on
    content space + content discriminator on style space). Same goal
    (disentanglement), simpler optimization.

    Args:
        z_content: (batch, content_latent_dim) — sampled content vectors
        z_style:   (batch, style_latent_dim)   — sampled style vectors

    Returns:
        Scalar HSIC value (non-negative)
    """
    B = z_content.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_content.device)

    K = rbf_kernel(z_content)
    L = rbf_kernel(z_style)

    # Centering matrix H = I - (1/B)·11ᵀ
    H = torch.eye(B, device=z_content.device) - (1.0 / B) * torch.ones(B, B, device=z_content.device)

    # HSIC = (1/(B-1)²) · trace(K·H·L·H)
    hsic_val = torch.trace(K @ H @ L @ H) / ((B - 1) ** 2)

    return hsic_val


# ──────────────────────────────────────────────────────────
# 4. Contrastive style loss
# ──────────────────────────────────────────────────────────

def contrastive_style_loss(
    z_style: torch.Tensor,
    doc_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss on the style latent space.

    For each sentence i in the batch:
      - Positives: all other sentences j where doc_ids[j] == doc_ids[i]
      - Negatives: all sentences j where doc_ids[j] != doc_ids[i]

    The loss encourages:
      - Same-style z_s vectors to have high cosine similarity
      - Different-style z_s vectors to have low cosine similarity

    This clusters Shakespeare z_s vectors together and modern z_s vectors
    together, creating two distinct regions in the style latent space.
    At inference, you can swap between these regions to transfer style.

    Lower temperature = sharper similarity distinctions (harder to satisfy).
    temperature=0.1 is standard for contrastive learning.

    Args:
        z_style:     (batch, style_latent_dim) — style latent vectors
        doc_ids:     (batch,) — style labels (0=modern, 1=Shakespeare)
        temperature: Sharpness of similarity (lower = sharper)

    Returns:
        Scalar loss (negative of mean log-probability of positives)
    """
    # L2-normalize style vectors for cosine similarity
    z = F.normalize(z_style, dim=-1)
    sim = z @ z.t()  # (B, B) cosine similarity matrix

    # Mask out self-similarity (diagonal)
    B = z.size(0)
    self_mask = torch.eye(B, device=z.device).bool()
    sim.masked_fill_(self_mask, -1e9)

    # Build positive mask: same doc_id, different index
    doc_ids_col = doc_ids.view(-1, 1)
    pos_mask = (doc_ids_col == doc_ids_col.t()) & (~self_mask)  # (B, B)

    # Scaled logits → log-softmax over all non-self entries
    logits = sim / temperature
    log_probs = F.log_softmax(logits, dim=1)

    # Average log-probability over positive pairs
    # Each row: sum of log_probs at positive positions / number of positives
    num_positives = pos_mask.float().sum(dim=1).clamp(min=1e-8)
    pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1) / num_positives

    return -pos_log_probs.mean()


# ──────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────

def compute_total_loss(
    logits: torch.Tensor,
    tgt_output: torch.Tensor,
    pad_token_id: int,
    c_mu: torch.Tensor,
    c_log_var: torch.Tensor,
    s_mu: torch.Tensor,
    s_log_var: torch.Tensor,
    z_c: torch.Tensor,
    z_s: torch.Tensor,
    doc_ids: torch.Tensor,
    global_step: int,
    gamma_hsic: float = 1.0,
    lambda_contrast: float = 1.0,
    contrastive_temperature: float = 0.1,
    kl_anneal_midpoint: int = 5000,
    kl_anneal_steepness: float = 0.002,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the total VAE loss with all components.

    L = L_recon + β(step)·(KL_c + KL_s) + γ·HSIC(z_c, z_s) + λ·Contrastive(z_s)

    Args:
        logits:      (batch, seq_len-1, vocab_size) — decoder output
        tgt_output:  (batch, seq_len-1) — ground truth next tokens
        pad_token_id: Padding token ID
        c_mu, c_log_var: Content latent distribution parameters
        s_mu, s_log_var: Style latent distribution parameters
        z_c, z_s:    Sampled latent vectors
        doc_ids:     Style labels
        global_step: Current training step (for KL annealing)
        gamma_hsic:  Weight for HSIC loss
        lambda_contrast: Weight for contrastive loss
        contrastive_temperature: Temperature for contrastive loss
        kl_anneal_midpoint: Step where β=0.5
        kl_anneal_steepness: Sigmoid sharpness for annealing

    Returns:
        total_loss: Scalar tensor
        metrics:    Dict of individual loss values (for logging)
    """
    # 1. Reconstruction
    recon = reconstruction_loss(logits, tgt_output, pad_token_id)

    # 2. KL divergence (content + style) with annealing
    kl_c = kl_divergence(c_mu, c_log_var)
    kl_s = kl_divergence(s_mu, s_log_var)
    kl_total = kl_c + kl_s
    beta = kl_anneal_factor(global_step, kl_anneal_midpoint, kl_anneal_steepness)
    kl_term = beta * kl_total

    # 3. HSIC independence loss
    hsic_term = gamma_hsic * hsic_loss(z_c, z_s)

    # 4. Contrastive style loss
    contrast_term = lambda_contrast * contrastive_style_loss(
        z_s, doc_ids, contrastive_temperature
    )

    # Total
    total = recon + kl_term + hsic_term + contrast_term

    metrics = {
        "loss": total.item(),
        "recon": recon.item(),
        "kl_c": kl_c.item(),
        "kl_s": kl_s.item(),
        "kl_total": kl_total.item(),
        "beta": beta,
        "hsic": hsic_term.item(),
        "contrast": contrast_term.item(),
    }

    return total, metrics
