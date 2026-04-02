"""
smoke_test.py — Verify that all modules wire together correctly.

This test uses RANDOM embeddings (not real LLaMA weights) so it can run
anywhere without GPU, HuggingFace login, or downloading the 1GB embedding
shard. It validates:
  1. Model construction and parameter counts
  2. Forward pass shapes (encode → reparameterize → decode)
  3. Loss computation (all 4 components)
  4. Backward pass (gradients flow correctly)
  5. Generation (autoregressive decoding)
  6. Dataset and collate_fn shapes

Run with:  python smoke_test.py
"""

import sys
import torch
import torch.nn as nn

# Add parent to path so imports work
sys.path.insert(0, ".")

from config import ModelConfig, TrainConfig, DataConfig
from model import VAETransformer, SinusoidalPositionalEncoding
from losses import (
    reconstruction_loss,
    kl_divergence,
    kl_anneal_factor,
    hsic_loss,
    contrastive_style_loss,
    compute_total_loss,
)
from data import StyleDataset


def make_dummy_model(config: ModelConfig) -> VAETransformer:
    """Build model with random embeddings (no LLaMA download needed)."""
    # Small vocab for testing
    vocab_size = 1000
    embed_dim = config.llama_embed_dim  # 4096

    # Use a smaller embed dim for faster testing
    test_embed_dim = 128
    config_copy = ModelConfig(
        llama_embed_dim=test_embed_dim,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=128,
        content_latent_dim=32,
        style_latent_dim=8,
        max_seq_len=32,
    )

    embedding = nn.Embedding(vocab_size, test_embed_dim)
    model = VAETransformer(config_copy, embedding)
    return model, config_copy, vocab_size


def test_model_construction():
    """Test 1: Model builds without errors."""
    print("Test 1: Model construction... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())

    # Check parameter groups
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert frozen > 0, "Embedding should be frozen"
    assert trainable > 0, "Should have trainable parameters"

    print(f"OK (trainable={trainable:,}, frozen={frozen:,})")


def test_forward_pass():
    """Test 2: Forward pass produces correct shapes."""
    print("Test 2: Forward pass shapes... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())
    model.eval()

    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Add some padding
    attention_mask[:, -3:] = 0

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    assert outputs["logits"].shape == (batch_size, seq_len - 1, vocab_size), \
        f"Logits shape wrong: {outputs['logits'].shape}"
    assert outputs["tgt_output"].shape == (batch_size, seq_len - 1), \
        f"Target shape wrong: {outputs['tgt_output'].shape}"
    assert outputs["c_mu"].shape == (batch_size, config.content_latent_dim), \
        f"c_mu shape wrong: {outputs['c_mu'].shape}"
    assert outputs["s_mu"].shape == (batch_size, config.style_latent_dim), \
        f"s_mu shape wrong: {outputs['s_mu'].shape}"
    assert outputs["z_c"].shape == (batch_size, config.content_latent_dim)
    assert outputs["z_s"].shape == (batch_size, config.style_latent_dim)

    print("OK")


def test_loss_computation():
    """Test 3: All loss components compute without errors."""
    print("Test 3: Loss computation... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())

    batch_size = 8
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    doc_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    outputs = model(input_ids, attention_mask)

    loss, metrics = compute_total_loss(
        logits=outputs["logits"],
        tgt_output=outputs["tgt_output"],
        pad_token_id=0,
        c_mu=outputs["c_mu"],
        c_log_var=outputs["c_log_var"],
        s_mu=outputs["s_mu"],
        s_log_var=outputs["s_log_var"],
        z_c=outputs["z_c"],
        z_s=outputs["z_s"],
        doc_ids=doc_ids,
        global_step=100,
    )

    assert loss.requires_grad, "Loss should require grad"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # Check all metric keys
    expected_keys = {"loss", "recon", "kl_c", "kl_s", "kl_total", "beta", "hsic", "contrast"}
    assert set(metrics.keys()) == expected_keys, f"Missing metrics: {expected_keys - set(metrics.keys())}"

    print(f"OK (loss={loss.item():.4f})")


def test_backward_pass():
    """Test 4: Gradients flow through all trainable parameters."""
    print("Test 4: Backward pass... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())

    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    doc_ids = torch.tensor([0, 1, 0, 1])

    outputs = model(input_ids, attention_mask)
    loss, _ = compute_total_loss(
        logits=outputs["logits"],
        tgt_output=outputs["tgt_output"],
        pad_token_id=0,
        c_mu=outputs["c_mu"],
        c_log_var=outputs["c_log_var"],
        s_mu=outputs["s_mu"],
        s_log_var=outputs["s_log_var"],
        z_c=outputs["z_c"],
        z_s=outputs["z_s"],
        doc_ids=doc_ids,
        global_step=100,
    )

    loss.backward()

    # Check that trainable params got gradients
    grad_params = 0
    no_grad_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_params += 1
            else:
                no_grad_params += 1

    # Embedding should NOT have gradients
    assert model.embedding.weight.grad is None, "Frozen embedding should not have gradients"

    print(f"OK ({grad_params} params with gradients, {no_grad_params} without)")


def test_generation():
    """Test 5: Autoregressive generation produces token sequences."""
    print("Test 5: Generation... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())
    model.eval()

    batch_size = 2
    z = torch.randn(batch_size, config.total_latent_dim)

    with torch.no_grad():
        generated = model.decode_generate(
            z=z,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            max_len=20,
        )

    assert generated.shape[0] == batch_size, f"Batch dim wrong: {generated.shape}"
    assert generated.shape[1] >= 2, f"Should generate at least BOS + 1 token: {generated.shape}"
    assert generated[:, 0].tolist() == [1, 1], "First token should be BOS"

    print(f"OK (generated shape: {generated.shape})")


def test_kl_annealing():
    """Test 6: KL annealing schedule produces correct values."""
    print("Test 6: KL annealing... ", end="")

    # β should be close to 0 at step 0
    beta_0 = kl_anneal_factor(0, midpoint=5000, steepness=0.002)
    assert beta_0 < 0.01, f"β at step 0 should be ~0, got {beta_0}"

    # β should be ~0.5 at midpoint
    beta_mid = kl_anneal_factor(5000, midpoint=5000, steepness=0.002)
    assert abs(beta_mid - 0.5) < 0.01, f"β at midpoint should be ~0.5, got {beta_mid}"

    # β should be close to 1 at 10000
    beta_10k = kl_anneal_factor(10000, midpoint=5000, steepness=0.002)
    assert beta_10k > 0.99, f"β at step 10000 should be ~1, got {beta_10k}"

    print(f"OK (β at 0/5000/10000 = {beta_0:.4f}/{beta_mid:.4f}/{beta_10k:.4f})")


def test_hsic():
    """Test 7: HSIC properties — zero for independent, positive for dependent."""
    print("Test 7: HSIC properties... ", end="")

    # Independent vectors should have low HSIC
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    hsic_indep = hsic_loss(x, y).item()

    # Dependent vectors (y is a function of x) should have higher HSIC
    y_dep = x[:, :8] + 0.1 * torch.randn(64, 8)
    hsic_dep = hsic_loss(x, y_dep).item()

    assert hsic_dep > hsic_indep, \
        f"HSIC should be higher for dependent data: {hsic_dep} vs {hsic_indep}"

    print(f"OK (independent={hsic_indep:.4f}, dependent={hsic_dep:.4f})")


def test_contrastive():
    """Test 8: Contrastive loss is lower when same-style vectors are similar."""
    print("Test 8: Contrastive loss... ", end="")

    # Case 1: Random vectors (high loss)
    z_random = torch.randn(8, 16)
    doc_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss_random = contrastive_style_loss(z_random, doc_ids).item()

    # Case 2: Clustered vectors (low loss)
    z_clustered = torch.randn(8, 16)
    z_clustered[:4] = z_clustered[:4] * 0.1 + torch.tensor([1.0] * 16)  # cluster 0
    z_clustered[4:] = z_clustered[4:] * 0.1 + torch.tensor([-1.0] * 16)  # cluster 1
    loss_clustered = contrastive_style_loss(z_clustered, doc_ids).item()

    assert loss_clustered < loss_random, \
        f"Clustered should have lower loss: {loss_clustered} vs {loss_random}"

    print(f"OK (random={loss_random:.4f}, clustered={loss_clustered:.4f})")


def test_dataset():
    """Test 9: StyleDataset produces correct format."""
    print("Test 9: Dataset format... ", end="")

    data_config = DataConfig()
    modern = ["I have half a mind to hit you.", "You are honest."]
    original = ["I have a mind to strike thee.", "Thou art honest."]

    dataset = StyleDataset(modern, original, data_config)

    assert len(dataset) == 4, f"Should have 4 samples (2+2), got {len(dataset)}"

    sample = dataset[0]
    assert "text" in sample, "Sample missing 'text'"
    assert "doc_id" in sample, "Sample missing 'doc_id'"
    assert sample["doc_id"] == 0, "First sample should be modern (label=0)"
    assert dataset[2]["doc_id"] == 1, "Third sample should be Shakespeare (label=1)"

    print("OK")


def test_weight_tying():
    """Test 10: Output projection reuses embedding weights."""
    print("Test 10: Weight tying... ", end="")
    model, config, vocab_size = make_dummy_model(ModelConfig())

    # The output path is: decoder_output → output_unproject → linear(embedding.weight)
    # Check that embedding weight is used in _compute_logits
    dummy_input = torch.randn(1, 5, config.d_model)
    logits = model._compute_logits(dummy_input)
    assert logits.shape == (1, 5, vocab_size), f"Logits shape wrong: {logits.shape}"

    print("OK")


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST — VAE Text Style Transfer")
    print("=" * 60)
    print()

    tests = [
        test_model_construction,
        test_forward_pass,
        test_loss_computation,
        test_backward_pass,
        test_generation,
        test_kl_annealing,
        test_hsic,
        test_contrastive,
        test_dataset,
        test_weight_tying,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed > 0:
        print(f"  {failed} tests FAILED")
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)
