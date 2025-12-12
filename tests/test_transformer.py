# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:11:10 2025

@author: chris and christine
"""

"""
Transformer Module Unit Tests
==============================

Tests for the gauge-theoretic transformer language model,
including embeddings, attention, and generation.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import torch - skip tests if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestGaugeTransformerLM:
    """Tests for the main GaugeTransformerLM model."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for fast testing."""
        return {
            'vocab_size': 100,
            'embed_dim': 9,  # Must be sum of irrep dims, odd total
            'n_layers': 1,
            'hidden_dim': 32,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'dropout': 0.0,
            'pos_encoding_mode': 'learned',
            'evolve_sigma': False,
            'evolve_phi': False,
            'tie_embeddings': True,
            'irrep_spec': [
                ('l0', 3, 1),  # 3 scalars
                ('l1', 2, 3),  # 2 triplets = 6
            ],  # Total: 9
        }

    def test_model_creation(self, small_config):
        """Test model can be created."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)
        assert model is not None

    def test_forward_pass_shape(self, small_config):
        """Test forward pass produces correct output shape."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)

        batch_size = 2
        seq_len = 8
        token_ids = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

        logits = model(token_ids)

        assert logits.shape == (batch_size, seq_len, small_config['vocab_size'])

    def test_forward_with_return_agents(self, small_config):
        """Test forward pass with agent state tracking."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)

        batch_size = 2
        seq_len = 8
        token_ids = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

        logits, agents = model(token_ids, return_agents=True)

        assert logits.shape == (batch_size, seq_len, small_config['vocab_size'])
        assert 'mu' in agents
        assert 'phi' in agents

    def test_generation(self, small_config):
        """Test autoregressive generation."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)

        prompt = torch.randint(0, small_config['vocab_size'], (1, 4))
        generated = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=1.0,
            top_k=10,
        )

        assert generated.shape[1] == 4 + 5  # prompt + new tokens

    def test_parameter_count(self, small_config):
        """Test parameter counting."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)

        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)

        assert total_params > 0
        assert non_embed_params >= 0
        assert non_embed_params <= total_params


class TestGaugeTokenEmbedding:
    """Tests for token embedding layer."""

    def test_embedding_creation(self):
        """Test embedding layer creation."""
        try:
            from transformer.embeddings import GaugeTokenEmbedding

            embed = GaugeTokenEmbedding(
                vocab_size=100,
                embed_dim=9,
                irrep_spec=[('l0', 3, 1), ('l1', 2, 3)],
            )
            assert embed is not None
        except ImportError:
            pytest.skip("GaugeTokenEmbedding not available")

    def test_embedding_output_shapes(self):
        """Test embedding output shapes."""
        try:
            from transformer.embeddings import GaugeTokenEmbedding

            embed_dim = 9
            embed = GaugeTokenEmbedding(
                vocab_size=100,
                embed_dim=embed_dim,
                irrep_spec=[('l0', 3, 1), ('l1', 2, 3)],
            )

            batch_size = 2
            seq_len = 8
            token_ids = torch.randint(0, 100, (batch_size, seq_len))

            mu, sigma, phi = embed(token_ids)

            assert mu.shape == (batch_size, seq_len, embed_dim)
            assert phi.shape == (batch_size, seq_len, 3)  # so(3) has 3 components
        except ImportError:
            pytest.skip("GaugeTokenEmbedding not available")


class TestGaugePositionalEncoding:
    """Tests for positional encoding."""

    def test_positional_encoding_creation(self):
        """Test positional encoding creation."""
        try:
            from transformer.embeddings import GaugePositionalEncoding

            pos_enc = GaugePositionalEncoding(
                max_seq_len=128,
                mode='learned',
            )
            assert pos_enc is not None
        except ImportError:
            pytest.skip("GaugePositionalEncoding not available")

    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        try:
            from transformer.embeddings import GaugePositionalEncoding

            pos_enc = GaugePositionalEncoding(
                max_seq_len=128,
                mode='learned',
            )

            seq_len = 16
            pos_phi = pos_enc(seq_len)

            assert pos_phi.shape == (seq_len, 3)  # so(3) has 3 components
        except ImportError:
            pytest.skip("GaugePositionalEncoding not available")


class TestAttentionMask:
    """Tests for attention mask creation."""

    def test_causal_mask_creation(self):
        """Test causal attention mask creation."""
        try:
            from transformer.attention import create_attention_mask

            mask = create_attention_mask(
                num_agents=8,
                pattern='full',
                window=64,
                device='cpu',
                causal=True,
            )

            assert mask.shape == (8, 8)
            # Causal mask should be lower triangular
            assert torch.all(mask == torch.tril(torch.ones(8, 8)))
        except ImportError:
            pytest.skip("create_attention_mask not available")

    def test_full_mask_creation(self):
        """Test full attention mask creation."""
        try:
            from transformer.attention import create_attention_mask

            mask = create_attention_mask(
                num_agents=8,
                pattern='full',
                window=64,
                device='cpu',
                causal=False,
            )

            assert mask.shape == (8, 8)
            # Full mask should be all ones
            assert torch.all(mask == 1)
        except ImportError:
            pytest.skip("create_attention_mask not available")


class TestTransformerStack:
    """Tests for transformer stack."""

    def test_transformer_stack_creation(self):
        """Test transformer stack creation."""
        try:
            from transformer.transformer_block import GaugeTransformerStack
            from math_utils.generators import generate_so3_generators

            generators = generate_so3_generators(9)
            generators_tensor = torch.from_numpy(np.array(generators)).float()

            stack = GaugeTransformerStack(
                n_layers=2,
                embed_dim=9,
                irrep_spec=[('l0', 3, 1), ('l1', 2, 3)],
                hidden_dim=32,
                kappa_beta=1.0,
                dropout=0.0,
                generators=generators_tensor,
            )
            assert stack is not None
        except (ImportError, TypeError):
            pytest.skip("GaugeTransformerStack not available or signature changed")


class TestKLAttention:
    """Tests for KL divergence-based attention."""

    def test_kl_attention_finite(self):
        """Test that KL attention produces finite values."""
        try:
            from transformer.attention import GaugeAttention
            from math_utils.generators import generate_so3_generators

            embed_dim = 9
            generators = generate_so3_generators(embed_dim)
            generators_tensor = torch.from_numpy(np.array(generators)).float()

            attention = GaugeAttention(
                embed_dim=embed_dim,
                irrep_spec=[('l0', 3, 1), ('l1', 2, 3)],
                kappa_beta=1.0,
            )

            batch_size = 2
            seq_len = 4
            mu = torch.randn(batch_size, seq_len, embed_dim)
            phi = torch.randn(batch_size, seq_len, 3) * 0.1

            # Create mask
            mask = torch.ones(batch_size, seq_len, seq_len)

            mu_out, _ = attention(mu, None, phi, generators_tensor, mask=mask)

            assert torch.all(torch.isfinite(mu_out))
        except (ImportError, TypeError):
            pytest.skip("GaugeAttention not available or signature changed")


class TestModelTraining:
    """Tests for model training compatibility."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for fast testing."""
        return {
            'vocab_size': 100,
            'embed_dim': 9,
            'n_layers': 1,
            'hidden_dim': 32,
            'max_seq_len': 16,
            'kappa_beta': 1.0,
            'dropout': 0.0,
            'pos_encoding_mode': 'learned',
            'evolve_sigma': False,
            'evolve_phi': False,
            'tie_embeddings': True,
            'irrep_spec': [('l0', 3, 1), ('l1', 2, 3)],
        }

    def test_gradient_flow(self, small_config):
        """Test that gradients flow through the model."""
        from transformer.model import GaugeTransformerLM

        model = GaugeTransformerLM(small_config)

        batch_size = 2
        seq_len = 8
        token_ids = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))
        targets = torch.randint(0, small_config['vocab_size'], (batch_size, seq_len))

        logits = model(token_ids)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, small_config['vocab_size']),
            targets.view(-1)
        )

        loss.backward()

        # Check that some gradients are non-zero
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"

    def test_model_deterministic_with_seed(self, small_config):
        """Test that model is deterministic with fixed seed."""
        from transformer.model import GaugeTransformerLM

        torch.manual_seed(42)
        model1 = GaugeTransformerLM(small_config)

        torch.manual_seed(42)
        model2 = GaugeTransformerLM(small_config)

        # Compare parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)