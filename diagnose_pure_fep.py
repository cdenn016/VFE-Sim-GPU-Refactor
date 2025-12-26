"""
Diagnostic script to identify pure FEP learning issues.
"""

import torch
from transformer.model import GaugeTransformerLM
from transformer.pure_fep_transformer import PureFEPTransformer

print("="*70)
print("PURE FEP DIAGNOSTIC")
print("="*70)

# Test 1: Check which pure FEP implementation exists
print("\n1. Checking available implementations...")

try:
    from transformer.pure_fep_transformer import PureFEPTransformer
    print("   ✓ PureFEPTransformer available (CORRECT - token-dependent priors)")
except ImportError:
    print("   ✗ PureFEPTransformer not available")

try:
    from transformer.variational_ffn import VariationalFFNDynamic
    print("   ✓ VariationalFFNDynamic available (may have position-dependent priors)")
except ImportError:
    print("   ✗ VariationalFFNDynamic not available")

# Test 2: Check prior shapes in a model
print("\n2. Creating test models to check prior shapes...")

# Config for standard VFE transformer with pure_fep_mode
config_vfe = {
    'vocab_size': 100,
    'embed_dim': 32,
    'n_layers': 2,
    'irrep_spec': [(0, 10, 1), (1, 3, 3)],  # 10×ℓ0 + 3×ℓ1 = 10+9=19 (not 32, but ok for test)
    'hidden_dim': 64,
    'max_seq_len': 64,
    'kappa_beta': 1.0,
    'ffn_mode': 'VFE_dynamic',
    'ffn_pure_fep_mode': True,  # Enable position-dependent priors (WRONG!)
    'ffn_prior_lr': 0.01,
    'ffn_n_iterations': 5,
}

try:
    model_vfe = GaugeTransformerLM(config_vfe)

    # Check if it has position-dependent priors
    for name, module in model_vfe.named_modules():
        if hasattr(module, 'prior_mu'):
            shape = module.prior_mu.shape
            print(f"\n   Found prior_mu in {name}:")
            print(f"   Shape: {shape}")
            if len(shape) == 2 and shape[0] == config_vfe['max_seq_len']:
                print(f"   ⚠️  PROBLEM: Position-dependent priors ({shape[0]} positions)")
                print(f"   This won't work for language modeling!")
            elif len(shape) == 2 and shape[0] == config_vfe['vocab_size']:
                print(f"   ✓ Token-dependent priors ({shape[0]} tokens)")
            break
except Exception as e:
    print(f"   Error creating VFE model: {e}")

# Test 3: Check PureFEPTransformer
print("\n3. Checking PureFEPTransformer (if available)...")

try:
    from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer, PriorBank

    # Create prior bank
    prior_bank = PriorBank(
        vocab_size=100,
        embed_dim=32,
        gauge_fixed_priors=False  # Use per-token priors
    )

    print(f"   PriorBank prior_mu shape: {prior_bank.prior_mu.shape}")
    if prior_bank.prior_mu.shape[0] == 100:
        print(f"   ✓ Correct: Token-dependent priors (vocab_size=100)")

except Exception as e:
    print(f"   Error: {e}")

# Test 4: Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("""
If you saw "⚠️ PROBLEM: Position-dependent priors" above:

ISSUE: You're using variational_ffn.py with pure_fep_mode=True,
       which uses position-dependent priors. This doesn't work for
       language modeling where the same token appears at different
       positions!

FIX 1: Use the correct pure FEP implementation
       python transformer/train_pure_fep.py --dataset shakespeare

FIX 2: Use PureFEPTransformer directly in your code
       from transformer.pure_fep_transformer import PureFEPTransformer

       model = PureFEPTransformer(config)  # Has token-dependent PriorBank

EXPLANATION:
- Position-dependent: prior[pos=5] applies to ANY token at position 5
- Token-dependent: prior[token_id=42] applies to token "bank" anywhere

Language needs token-dependent priors!
""")

print("="*70)
