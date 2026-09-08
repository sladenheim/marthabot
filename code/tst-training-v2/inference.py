"""
inference.py — Style transfer inference and evaluation.

How style transfer works at inference:
══════════════════════════════════════

1. ENCODE: Feed a modern English sentence through the encoder.
   This produces z_c (content latent) and z_s (style latent).
   z_c captures WHAT the sentence says (entities, relationships, meaning).
   z_s captures HOW it says it (modern style).

2. SWAP STYLE: Discard the modern z_s. Replace it with the average
   Shakespeare z_s computed from the training data. This tells the
   decoder "keep the same content, but generate in Shakespeare style."

3. DECODE: Feed [z_c; avg_shakespeare_z_s] through the autoregressive
   decoder. The decoder generates Shakespeare-style text one token at
   a time, preserving the content from z_c.

4. POST-PROCESS: The nltk tokenization in the training data added extra
   spaces before punctuation (e.g., "speak'st ."). The decoder may
   reproduce this pattern. We strip these artifacts.

Evaluation:
  - BLEU score against ground-truth parallel Shakespeare sentences
  - Style accuracy (optional: train a simple classifier)
  - Content preservation (BLEU between input content words and output)
"""

import os
import re
from pathlib import Path

# Redirect HuggingFace cache to project space to avoid home dir quota
_hf_cache = "/projectnb/scottml/seansal2/.cache/huggingface"
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig, DataConfig
from model import VAETransformer, load_llama_embeddings
from data import build_dataloaders, load_sentences


# ──────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────

def clean_nltk_artifacts(text: str) -> str:
    """
    Clean up spacing artifacts from nltk tokenization.

    The .nltktok training data has spaces before punctuation:
        "I have a mind to strike thee ."
    LLaMA's BPE may reproduce this pattern. This function normalizes:
        "I have a mind to strike thee."
    """
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:\'\")\]])', r'\1', text)
    # Remove spaces after opening brackets/quotes
    text = re.sub(r'([\[(\"\'])\s+', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ──────────────────────────────────────────────────────────
# Style transfer
# ──────────────────────────────────────────────────────────

class StyleTransfer:
    """
    Performs style transfer using a trained VAE model.

    Usage:
        transfer = StyleTransfer.from_checkpoint("checkpoints/final_model.pt")
        result = transfer.transfer("I have half a mind to hit you.", target_style=1)
        print(result)  # Shakespeare-style output
    """

    def __init__(
        self,
        model: VAETransformer,
        tokenizer: AutoTokenizer,
        avg_style_embs: dict[int, torch.Tensor],
        model_config: ModelConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.avg_style_embs = avg_style_embs
        self.config = model_config
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
    ) -> "StyleTransfer":
        """
        Load a trained model from a checkpoint file.

        The checkpoint contains:
          - model_state_dict: Trained model weights
          - avg_style_embs: Average z_s for each style (computed after training)
          - model_config: Architecture hyperparameters
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_config = ckpt["model_config"]
        avg_style_embs = ckpt["avg_style_embs"]

        # Rebuild model
        embedding_layer = load_llama_embeddings(model_config.llama_model_name)
        model = VAETransformer(model_config, embedding_layer).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.llama_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model loaded. Available styles: {list(avg_style_embs.keys())}")
        return cls(model, tokenizer, avg_style_embs, model_config, device)

    @torch.no_grad()
    def transfer(
        self,
        text: str,
        target_style: int,
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """
        Transfer the style of a single sentence.

        Args:
            text:         Input sentence (any style)
            target_style: Target style label (0=modern, 1=Shakespeare)
            max_len:      Maximum output length
            temperature:  Sampling temperature
            top_k:        Top-k sampling (None=greedy)

        Returns:
            Style-transferred text
        """
        # Tokenize input
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Encode → get content latent (keep) and style latent (discard)
        c_mu, c_log_var, s_mu, s_log_var = self.model.encode(input_ids, attention_mask)

        # Use content MEAN (not sampled) for more deterministic output
        z_c = c_mu

        # Replace style with target style prototype
        target_z_s = self.avg_style_embs[target_style].to(self.device)
        target_z_s = target_z_s.unsqueeze(0)  # (1, style_latent_dim)

        # Concatenate content + target style
        z = torch.cat([z_c, target_z_s], dim=-1)  # (1, total_latent_dim)

        # Generate
        generated_ids = self.model.decode_generate(
            z=z,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode tokens back to text
        output_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return clean_nltk_artifacts(output_text)

    @torch.no_grad()
    def transfer_batch(
        self,
        texts: list[str],
        target_style: int,
        max_len: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[str]:
        """Transfer style for a batch of sentences."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Encode
        c_mu, c_log_var, s_mu, s_log_var = self.model.encode(input_ids, attention_mask)
        z_c = c_mu

        # Replace style
        batch_size = z_c.size(0)
        target_z_s = self.avg_style_embs[target_style].to(self.device)
        target_z_s = target_z_s.unsqueeze(0).expand(batch_size, -1)

        z = torch.cat([z_c, target_z_s], dim=-1)

        # Generate
        generated_ids = self.model.decode_generate(
            z=z,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode
        results = []
        for i in range(batch_size):
            text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            results.append(clean_nltk_artifacts(text))

        return results


# ──────────────────────────────────────────────────────────
# BLEU evaluation
# ──────────────────────────────────────────────────────────

def compute_bleu_scores(
    transfer: StyleTransfer,
    test_modern: list[str],
    test_original: list[str],
    target_style: int = 1,
    max_samples: int | None = None,
    batch_size: int = 32,
) -> dict:
    """
    Evaluate style transfer quality using BLEU against parallel ground truth.

    For each modern sentence:
      1. Transfer to Shakespeare style using the model
      2. Compare generated output to the ground-truth parallel Shakespeare sentence
      3. Compute corpus-level and sentence-level BLEU

    Args:
        transfer:      StyleTransfer object
        test_modern:   List of modern English test sentences
        test_original: List of parallel Shakespeare test sentences (ground truth)
        target_style:  Target style label (1=Shakespeare)
        max_samples:   Limit number of samples (None=all)
        batch_size:    Batch size for generation

    Returns:
        dict with BLEU scores and sample outputs
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    except ImportError:
        print("nltk not installed. Run: pip install nltk")
        return {}

    if max_samples is not None:
        test_modern = test_modern[:max_samples]
        test_original = test_original[:max_samples]

    # Generate in batches
    all_generated = []
    for i in range(0, len(test_modern), batch_size):
        batch = test_modern[i:i + batch_size]
        generated = transfer.transfer_batch(batch, target_style=target_style)
        all_generated.extend(generated)
        if (i // batch_size) % 10 == 0:
            print(f"  Generated {len(all_generated)}/{len(test_modern)}...")

    # Compute BLEU
    # References are lists of lists of tokens (multiple references per sentence)
    # Hypotheses are lists of tokens
    references = [[ref.split()] for ref in test_original]
    hypotheses = [gen.split() for gen in all_generated]

    smoother = SmoothingFunction().method1
    bleu_4 = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    bleu_1 = corpus_bleu(
        references, hypotheses,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smoother,
    )

    # Sample outputs for inspection
    samples = []
    for i in range(min(10, len(test_modern))):
        samples.append({
            "input": test_modern[i],
            "generated": all_generated[i],
            "reference": test_original[i],
        })

    results = {
        "bleu_1": bleu_1,
        "bleu_4": bleu_4,
        "num_samples": len(test_modern),
        "samples": samples,
    }

    print(f"\n  BLEU-1: {bleu_1:.4f}")
    print(f"  BLEU-4: {bleu_4:.4f}")
    print(f"\n  Sample transfers:")
    for s in samples[:5]:
        print(f"    Input:     {s['input']}")
        print(f"    Generated: {s['generated']}")
        print(f"    Reference: {s['reference']}")
        print()

    return results


# ──────────────────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────────────────

def demo(checkpoint_path: str = "checkpoints/final_model.pt"):
    """
    Quick demo: transfer a few example sentences.
    """
    transfer = StyleTransfer.from_checkpoint(checkpoint_path)

    modern_sentences = [
        "I have half a mind to hit you before you speak again.",
        "You are an honest man.",
        "I'm going to make you a rich man.",
        "Have you given up so quickly on Rosaline, whom you loved so much?",
        "Holy Saint Francis, this is a drastic change!",
    ]

    print("\n" + "=" * 60)
    print("STYLE TRANSFER DEMO: Modern → Shakespeare")
    print("=" * 60)

    for sent in modern_sentences:
        result = transfer.transfer(sent, target_style=1)  # 1 = Shakespeare
        print(f"\n  Modern:      {sent}")
        print(f"  Shakespeare: {result}")


def evaluate(checkpoint_path: str = "checkpoints/final_model.pt"):
    """
    Full evaluation: BLEU scores on the test set.
    """
    data_config = DataConfig()
    transfer = StyleTransfer.from_checkpoint(checkpoint_path)

    test_modern = load_sentences(data_config.get_path(data_config.test_modern))
    test_original = load_sentences(data_config.get_path(data_config.test_original))

    print("\n" + "=" * 60)
    print("EVALUATION: Modern → Shakespeare (BLEU on test set)")
    print("=" * 60)

    results = compute_bleu_scores(transfer, test_modern, test_original)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/final_model.pt"
        evaluate(ckpt)
    else:
        ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/final_model.pt"
        demo(ckpt)
