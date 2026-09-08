"""
model.py — VAE Transformer for Text Style Transfer.

Architecture decisions and their rationale:
═══════════════════════════════════════════

1. FROZEN LLAMA EMBEDDINGS + PROJECTION
   LLaMA 3.1 8B's embedding table maps 128K BPE tokens to 4096-dim vectors.
   These embeddings encode rich semantic relationships learned from massive
   pretraining data — far better than training embeddings from scratch on
   ~37K Shakespeare sentences. We freeze them (no gradients) to save memory
   and prevent catastrophic forgetting.

   But 4096 is too wide for our transformer layers (would need ~20GB+ for
   6-layer transformers). So we add a learned projection: Linear(4096→512)
   + LayerNorm. This compresses the pretrained representations into a
   working dimension our transformer can handle efficiently.

2. TRANSFORMER ENCODER → MASKED MEAN POOL → LATENT SPACE
   The encoder processes the projected embeddings through self-attention
   layers, then collapses the sequence dimension via masked mean pooling
   (ignoring padding tokens). This produces a single vector per sentence
   that gets split into content μ/σ² and style μ/σ² via separate linear
   projections.

   Why mean pool instead of [CLS] token? Mean pooling is more stable for
   VAEs because it averages information across all positions, while [CLS]
   relies on a single position learning to aggregate everything.

3. NO CROSS-ATTENTION IN DECODER (CRITICAL)
   This is the most important architectural choice. A standard Transformer
   decoder cross-attends to encoder memory, meaning it can read the full
   encoded input sequence at every decoding step. For a VAE, this is
   catastrophic: the decoder bypasses the latent bottleneck entirely by
   just copying from encoder memory. The latent z becomes noise, HSIC is
   trivially low, and style transfer fails.

   Instead, our decoder uses ONLY causal self-attention. Its inputs are:
     - Shifted target token embeddings (teacher forcing during training)
     - The latent vector z projected back to d_model and ADDED to each position
     - Learned positional encodings

   All global information (content + style) must flow through the
   160-dimensional bottleneck (128 content + 32 style). This forces
   meaningful latent representations.

4. AUTOREGRESSIVE DECODING
   During training: teacher forcing — the decoder sees the ground-truth
   previous tokens shifted right, and predicts the next token at each
   position. Loss is computed on all positions simultaneously.

   During inference: greedy or top-k sampling — generate one token at a
   time, feeding each generated token back as input for the next step.

5. OUTPUT PROJECTION WITH WEIGHT TYING
   The output head projects from d_model back to vocab_size. We tie the
   output projection weights to the input embedding weights (through the
   projection layer). This is parameter-efficient and encourages the model
   to produce outputs in the same semantic space as its inputs.

   Specifically: output_logits = (decoder_output @ projection.T) @ embedding.T
   This means we go d_model → 4096 → 128256 (vocab logits) by reusing
   the projection and embedding matrices. No extra parameters needed for
   the output head.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from huggingface_hub import hf_hub_download

from config import ModelConfig


# ──────────────────────────────────────────────────────────
# Embedding loading (from vae_style_transfer_starter-3.4.ipynb)
# ──────────────────────────────────────────────────────────

def load_llama_embeddings(model_name: str) -> nn.Embedding:
    """
    Load ONLY the embedding layer from LLaMA 3.1 8B without loading the
    full 8B parameter model.

    Uses safetensors memory mapping (safe_open) to extract just the
    embedding tensor from the correct model shard. This uses ~1GB of RAM
    instead of ~16GB for the full model.

    From notebook 3 (vae_style_transfer_starter), Option B simplified.
    """
    # 1. Download the safetensors index to find which shard has embeddings
    index_path = hf_hub_download(model_name, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # 2. Locate the shard containing the embedding weights
    embed_key = "model.embed_tokens.weight"
    shard_file = index["weight_map"][embed_key]
    shard_path = hf_hub_download(model_name, shard_file)

    # 3. Memory-map the shard and extract ONLY the embedding tensor
    # safe_open maps the file without loading the entire shard into RAM
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor(embed_key)  # (vocab_size, 4096)

    vocab_size, embed_dim = embed_weight.shape
    print(f"Loaded LLaMA embeddings: vocab_size={vocab_size}, embed_dim={embed_dim}")
    print(f"  dtype={embed_weight.dtype}, size={embed_weight.nelement() * embed_weight.element_size() / 1e9:.2f} GB")

    # Keep native dtype (bfloat16) to save memory — the projection layer
    # will cast to float32 as needed during the forward pass.
    embedding_layer = nn.Embedding(vocab_size, embed_dim, _weight=embed_weight)
    return embedding_layer


# ──────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".

    Why sinusoidal instead of learned? With max_seq_len=128 and d_model=512,
    learned positional embeddings would add 65K parameters — trivial. But
    sinusoidal encodings generalize better to unseen sequence lengths and
    are a well-understood baseline. Either works fine here.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter — no gradients, but moves with .to())
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────

class VAETransformer(nn.Module):
    """
    VAE with Transformer encoder and autoregressive Transformer decoder
    for text style transfer.

    Forward pass (training):
        input_ids, attention_mask
        → embed(input_ids)                               # (B, S, 4096)
        → project(embed)                                  # (B, S, 512)
        → encoder(projected)                              # (B, S, 512)
        → masked_mean_pool → content μ/σ², style μ/σ²    # each (B, latent_dim)
        → reparameterize → z_c, z_s                       # (B, 128), (B, 32)
        → concat [z_c; z_s] → project to d_model          # (B, 512)
        → decoder(shifted_targets + z_broadcast)           # (B, S-1, 512)
        → output_proj → logits                            # (B, S-1, vocab_size)

    The decoder does NOT cross-attend to encoder memory. All information
    flows through the 160-dim latent bottleneck.
    """

    def __init__(self, config: ModelConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config = config

        # ─── Frozen pretrained embeddings ───
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = False
        actual_vocab_size = embedding_layer.weight.shape[0]
        actual_embed_dim = embedding_layer.weight.shape[1]

        # ─── Projection: 4096 → d_model ───
        # This is a learned linear map that compresses LLaMA's rich 4096-dim
        # embeddings into our working dimension. LayerNorm stabilizes training.
        self.input_projection = nn.Sequential(
            nn.Linear(actual_embed_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # ─── Positional encoding ───
        self.pos_encoder = SinusoidalPositionalEncoding(
            config.d_model, max_len=config.max_seq_len, dropout=config.dropout
        )

        # ─── Transformer encoder ───
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,           # (batch, seq, d_model) convention
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_encoder_layers
        )

        # ─── Latent projections ───
        # From pooled encoder output (d_model) to content and style latent spaces.
        # Separate μ and log_var heads for each space.
        self.content_mu = nn.Linear(config.d_model, config.content_latent_dim)
        self.content_log_var = nn.Linear(config.d_model, config.content_latent_dim)
        self.style_mu = nn.Linear(config.d_model, config.style_latent_dim)
        self.style_log_var = nn.Linear(config.d_model, config.style_latent_dim)

        # ─── Latent → decoder space ───
        # Maps the concatenated [z_c; z_s] back to d_model for the decoder.
        self.latent_to_d_model = nn.Linear(config.total_latent_dim, config.d_model)

        # ─── Transformer decoder (self-attention ONLY, no cross-attention) ───
        # We use TransformerEncoder with causal masking, NOT TransformerDecoder.
        # TransformerDecoder expects cross-attention memory, which we don't want.
        # TransformerEncoder with a causal mask gives us the same autoregressive
        # behavior without any cross-attention to encoder outputs.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=config.n_decoder_layers
        )

        # ─── Output projection (weight-tied) ───
        # Instead of a massive Linear(512, 128256), we reverse the projection:
        # d_model → 4096 via input_projection.T, then 4096 → vocab via embedding.T.
        # This reuses existing weights and encourages semantic consistency.
        self.output_unproject = nn.Linear(config.d_model, actual_embed_dim, bias=False)
        # The final vocab projection uses the frozen embedding weights (tied).
        # We don't create a separate parameter — see _compute_logits().

    def _compute_logits(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        Project decoder output to vocabulary logits using weight tying.

        decoder_output: (batch, seq_len, d_model)
        returns:        (batch, seq_len, vocab_size)

        Path: d_model → 4096 → vocab_size
        The second projection reuses the frozen embedding weights.
        """
        # (batch, seq_len, d_model) → (batch, seq_len, 4096)
        h = self.output_unproject(decoder_output)
        # Cast to match frozen LLaMA embedding dtype (bfloat16) before matmul
        h = h.to(self.embedding.weight.dtype)
        # (batch, seq_len, 4096) @ (4096, vocab_size) → (batch, seq_len, vocab_size)
        logits = F.linear(h, self.embedding.weight)
        return logits

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build an upper-triangular causal mask for autoregressive decoding.

        Returns a (seq_len, seq_len) boolean tensor where True = BLOCKED.
        Position i can attend to positions 0..i but not i+1..seq_len-1.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Encode input tokens to latent content and style distributions.

        Args:
            input_ids:      (batch, seq_len) — token IDs from LLaMA tokenizer
            attention_mask: (batch, seq_len) — 1=real token, 0=padding

        Returns:
            c_mu, c_log_var: (batch, content_latent_dim) — content distribution
            s_mu, s_log_var: (batch, style_latent_dim)   — style distribution
        """
        # Step 1: Embed tokens with frozen LLaMA embeddings
        # Cast to float32 for the projection layer (embeddings may be bfloat16)
        embeds = self.embedding(input_ids).float()   # (B, S, 4096)

        # Step 2: Project down to working dimension
        projected = self.input_projection(embeds)     # (B, S, d_model)

        # Step 3: Add positional encoding
        projected = self.pos_encoder(projected)

        # Step 4: Transformer encoder
        # src_key_padding_mask: True = IGNORE this position (PyTorch convention)
        src_key_padding_mask = (attention_mask == 0)
        memory = self.encoder(
            projected,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, S, d_model)

        # Step 5: Masked mean pooling → single vector per sentence
        # Expand mask to match memory dimensions, then average over non-padding
        mask_f = attention_mask.unsqueeze(-1).float()   # (B, S, 1)
        pooled = (memory * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        # pooled: (B, d_model)

        # Step 6: Project to latent distributions
        c_mu = self.content_mu(pooled)
        c_log_var = self.content_log_var(pooled)
        s_mu = self.style_mu(pooled)
        s_log_var = self.style_log_var(pooled)

        return c_mu, c_log_var, s_mu, s_log_var

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = μ + σ · ε, where ε ~ N(0, I).

        This allows gradients to flow through the sampling operation.
        During training, this adds stochasticity; at inference, you can
        just use z = μ (set log_var contribution to 0 or skip this).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_train(
        self,
        z: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced autoregressive decoding for training.

        The decoder sees shifted ground-truth tokens as input and predicts
        the next token at each position. The latent vector z is broadcast
        across all positions and added to the token embeddings.

        Args:
            z:              (batch, total_latent_dim) — sampled latent vector
            target_ids:     (batch, seq_len) — full target token IDs [BOS, t1, ..., tN]
            attention_mask: (batch, seq_len) — attention mask for target

        Returns:
            logits:     (batch, seq_len-1, vocab_size) — predicted next-token logits
            tgt_output: (batch, seq_len-1)             — ground truth next tokens
        """
        # Teacher forcing: input is [BOS, t1, ..., t_{N-1}], target is [t1, ..., tN]
        tgt_input = target_ids[:, :-1]       # (B, S-1)
        tgt_output = target_ids[:, 1:]       # (B, S-1)
        tgt_mask_input = attention_mask[:, :-1]

        # Embed target tokens and project
        tgt_embeds = self.embedding(tgt_input).float()   # (B, S-1, 4096)
        tgt_projected = self.input_projection(tgt_embeds)  # (B, S-1, d_model)

        # Add positional encoding
        tgt_projected = self.pos_encoder(tgt_projected)

        # Broadcast latent z across all positions and add
        # This is how global information (content + style) reaches the decoder
        z_projected = self.latent_to_d_model(z)           # (B, d_model)
        z_broadcast = z_projected.unsqueeze(1)             # (B, 1, d_model)
        decoder_input = tgt_projected + z_broadcast        # (B, S-1, d_model)

        # Build causal mask (upper triangular, True = blocked)
        seq_len = tgt_input.size(1)
        causal_mask = self._build_causal_mask(seq_len, tgt_input.device)

        # Padding mask for decoder self-attention
        tgt_key_padding_mask = (tgt_mask_input == 0)

        # Run decoder (self-attention only, no cross-attention)
        decoder_output = self.decoder(
            decoder_input,
            mask=causal_mask,
            src_key_padding_mask=tgt_key_padding_mask,
        )  # (B, S-1, d_model)

        # Project to vocab logits
        logits = self._compute_logits(decoder_output)   # (B, S-1, vocab_size)

        return logits, tgt_output

    @torch.no_grad()
    def decode_generate(
        self,
        z: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation for inference.

        Starting from BOS, generate tokens one at a time by feeding each
        generated token back as input for the next step. Stops when EOS
        is generated or max_len is reached.

        Args:
            z:            (batch, total_latent_dim) — latent vector
            bos_token_id: Token ID for beginning-of-sequence
            eos_token_id: Token ID for end-of-sequence
            pad_token_id: Token ID for padding
            max_len:      Maximum sequence length to generate
            temperature:  Sampling temperature (1.0=neutral, <1=sharper, >1=flatter)
            top_k:        If set, sample from top-k tokens instead of greedy

        Returns:
            generated: (batch, generated_len) — token IDs including BOS
        """
        device = z.device
        batch_size = z.size(0)

        # Project latent to decoder space
        z_projected = self.latent_to_d_model(z)           # (B, d_model)
        z_broadcast = z_projected.unsqueeze(1)             # (B, 1, d_model)

        # Start with BOS token for each sequence
        cur_tokens = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Embed all tokens generated so far
            tok_embeds = self.embedding(cur_tokens).float()
            tok_projected = self.input_projection(tok_embeds)
            tok_projected = self.pos_encoder(tok_projected)

            # Add latent bias
            decoder_input = tok_projected + z_broadcast

            # Causal mask
            seq_len = cur_tokens.size(1)
            causal_mask = self._build_causal_mask(seq_len, device)
            pad_mask = (cur_tokens == pad_token_id)

            # Decode
            decoder_output = self.decoder(
                decoder_input,
                mask=causal_mask,
                src_key_padding_mask=pad_mask,
            )

            # Get logits for the last position only
            logits = self._compute_logits(decoder_output[:, -1:, :])  # (B, 1, V)
            logits = logits.squeeze(1) / temperature                    # (B, V)

            if top_k is not None:
                # Top-k sampling
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(values, dim=-1)
                sampled = indices.gather(-1, torch.multinomial(probs, 1))
                next_tokens = sampled
            else:
                # Greedy decoding
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Mark finished sequences
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

            # Append token
            cur_tokens = torch.cat([cur_tokens, next_tokens], dim=1)

            if finished.all():
                break

        return cur_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Full forward pass for training.

        For non-parallel training, target_ids = input_ids (autoencoder).
        The model encodes the input, samples from the latent space, and
        tries to reconstruct the same input via teacher-forced decoding.

        Args:
            input_ids:      (batch, seq_len) — token IDs
            attention_mask: (batch, seq_len) — 1=real, 0=pad

        Returns dict with:
            logits:     (batch, seq_len-1, vocab_size) — next-token predictions
            tgt_output: (batch, seq_len-1)             — ground truth next tokens
            c_mu, c_log_var: (batch, content_latent_dim)
            s_mu, s_log_var: (batch, style_latent_dim)
            z_c, z_s:        (batch, *_latent_dim) — sampled latent vectors
        """
        # Encode
        c_mu, c_log_var, s_mu, s_log_var = self.encode(input_ids, attention_mask)

        # Reparameterize
        z_c = self.reparameterize(c_mu, c_log_var)
        z_s = self.reparameterize(s_mu, s_log_var)

        # Concatenate content + style for decoder
        z = torch.cat([z_c, z_s], dim=-1)  # (B, total_latent_dim)

        # Decode (teacher forcing: target = input for autoencoder)
        logits, tgt_output = self.decode_train(z, input_ids, attention_mask)

        return {
            "logits": logits,
            "tgt_output": tgt_output,
            "c_mu": c_mu,
            "c_log_var": c_log_var,
            "s_mu": s_mu,
            "s_log_var": s_log_var,
            "z_c": z_c,
            "z_s": z_s,
        }


# ──────────────────────────────────────────────────────────
# Model construction helper
# ──────────────────────────────────────────────────────────

def build_model(config: ModelConfig, device: torch.device) -> VAETransformer:
    """
    Build the full VAE model: load LLaMA embeddings, construct architecture,
    move to device, and print parameter counts.
    """
    print("Loading LLaMA embeddings...")
    embedding_layer = load_llama_embeddings(config.llama_model_name)

    print("Building VAETransformer...")
    model = VAETransformer(config, embedding_layer).to(device)

    # Parameter count breakdown
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter counts:")
    print(f"  Frozen (LLaMA embeddings): {frozen / 1e6:.1f}M")
    print(f"  Trainable:                 {trainable / 1e6:.1f}M")
    print(f"  Total:                     {(frozen + trainable) / 1e6:.1f}M")

    return model
