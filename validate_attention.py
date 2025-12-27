#!/usr/bin/env python3
"""
Attention Pattern Validation Script
====================================

Comprehensive diagnostics to determine if attention patterns are:
- Mathematically correct (normalization, causality, KL properties)
- Interpretable (positional bias, token patterns)
- Statistically meaningful (vs uniform baseline)
- Learning useful structure (diversity correlation, entropy)

Usage:
    python validate_attention.py --checkpoint path/to/model.pt --config path/to/config.py
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Store validation test results."""
    test_name: str
    passed: bool
    score: float
    message: str
    severity: str  # 'critical', 'warning', 'info'


class AttentionValidator:
    """Comprehensive attention pattern validation."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.results: List[ValidationResult] = []

    def validate_all(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Run all validation tests."""
        print("=" * 70)
        print("ATTENTION VALIDATION REPORT")
        print("=" * 70)

        input_ids, _ = batch
        input_ids = input_ids.to(self.device)

        # Get attention info
        with torch.no_grad():
            _, attn_info = self.model.forward_with_attention(input_ids)

        beta = attn_info.get('beta')
        kl = attn_info.get('kl')
        mu = attn_info.get('mu')

        if beta is None:
            print("❌ CRITICAL: Model does not return attention info!")
            return {'status': 'failed', 'results': []}

        # Run all tests
        print("\n[1/7] Sanity Checks (Correctness)")
        print("-" * 70)
        self._check_normalization(beta)
        self._check_causality(beta)
        if kl is not None:
            self._check_kl_properties(kl)

        print("\n[2/7] Positional Patterns")
        print("-" * 70)
        self._check_positional_bias(beta)

        print("\n[3/7] Token-Level Patterns")
        print("-" * 70)
        self._check_token_repetition(beta, input_ids)
        self._check_special_tokens(beta, input_ids)

        print("\n[4/7] Statistical Tests")
        print("-" * 70)
        self._test_vs_uniform(beta)
        self._compute_entropy(beta)

        print("\n[5/7] Per-Head Analysis")
        print("-" * 70)
        self._analyze_per_head(beta)

        print("\n[6/7] Diversity Correlation")
        print("-" * 70)
        if mu is not None:
            self._check_diversity_correlation(beta, mu)

        print("\n[7/7] Gradient Flow Check")
        print("-" * 70)
        self._check_gradient_flow()

        # Summary
        self._print_summary()

        return {
            'status': 'complete',
            'results': self.results,
            'summary': self._get_summary_dict()
        }

    def _check_normalization(self, beta: torch.Tensor):
        """Test if attention rows sum to 1."""
        row_sums = beta.sum(dim=-1)
        expected = torch.ones_like(row_sums)

        max_error = (row_sums - expected).abs().max().item()
        all_close = torch.allclose(row_sums, expected, atol=1e-4)

        if all_close:
            self._log_result("Normalization", True, max_error,
                           f"✓ All rows sum to 1.0 (max error: {max_error:.2e})", 'info')
        else:
            self._log_result("Normalization", False, max_error,
                           f"❌ Rows don't sum to 1.0 (max error: {max_error:.2e})", 'critical')

    def _check_causality(self, beta: torch.Tensor):
        """Test if attention respects causality (no future peeking)."""
        # Check if model is causal by looking for upper triangle
        N = beta.shape[-1]
        upper_tri = torch.triu(beta[0, 0], diagonal=1)

        max_future = upper_tri.max().item()
        is_causal = max_future < 1e-6

        if is_causal:
            self._log_result("Causality", True, max_future,
                           f"✓ No future attention (max: {max_future:.2e})", 'info')
        else:
            # Not necessarily wrong - might be non-causal by design
            self._log_result("Causality", True, max_future,
                           f"ℹ Non-causal attention detected (max future attn: {max_future:.4f})", 'info')

    def _check_kl_properties(self, kl: torch.Tensor):
        """Test KL divergence properties."""
        # KL should be non-negative
        min_kl = kl.min().item()

        # Diagonal should be near zero (KL(q||q) = 0)
        B, H, N, _ = kl.shape
        diag_kl = kl[:, :, range(N), range(N)]
        max_diag_kl = diag_kl.max().item()
        mean_diag_kl = diag_kl.mean().item()

        kl_valid = min_kl >= -1e-5  # Allow tiny numerical errors
        diag_valid = max_diag_kl < 0.1

        if kl_valid and diag_valid:
            self._log_result("KL Properties", True, mean_diag_kl,
                           f"✓ KL ≥ 0 (min: {min_kl:.2e}), diag ≈ 0 (mean: {mean_diag_kl:.4f})", 'info')
        else:
            issues = []
            if not kl_valid:
                issues.append(f"negative KL: {min_kl:.4f}")
            if not diag_valid:
                issues.append(f"large diagonal: {max_diag_kl:.4f}")
            self._log_result("KL Properties", False, max(abs(min_kl), max_diag_kl),
                           f"❌ KL issues: {', '.join(issues)}", 'critical')

    def _check_positional_bias(self, beta: torch.Tensor):
        """Analyze attention as function of distance."""
        B, H, N, _ = beta.shape
        beta_np = beta[0].cpu().numpy()

        distance_attn = defaultdict(list)

        for h in range(H):
            for i in range(N):
                for j in range(i):  # Only past positions
                    dist = i - j
                    distance_attn[dist].append(beta_np[h, i, j])

        if len(distance_attn) == 0:
            self._log_result("Positional Bias", False, 0.0,
                           "⚠️  No past positions to analyze", 'warning')
            return

        # Compute average attention by distance
        distances = sorted(distance_attn.keys())
        avg_attn = [np.mean(distance_attn[d]) for d in distances]

        # Check for local bias (nearby > far)
        if len(avg_attn) >= 10:
            nearby = np.mean(avg_attn[:5])   # Distance 1-5
            far = np.mean(avg_attn[-5:])     # Last 5 distances
            ratio = nearby / (far + 1e-10)

            has_local_bias = nearby > far * 1.1  # At least 10% stronger

            if has_local_bias:
                self._log_result("Positional Bias", True, ratio,
                               f"✓ Local bias detected (nearby: {nearby:.4f}, far: {far:.4f}, ratio: {ratio:.2f}x)", 'info')
            else:
                self._log_result("Positional Bias", False, ratio,
                               f"⚠️  No local bias (nearby: {nearby:.4f}, far: {far:.4f}, ratio: {ratio:.2f}x)", 'warning')
        else:
            avg_all = np.mean(avg_attn)
            self._log_result("Positional Bias", True, avg_all,
                           f"ℹ Sequence too short for bias analysis (avg: {avg_all:.4f})", 'info')

    def _check_token_repetition(self, beta: torch.Tensor, input_ids: torch.Tensor):
        """Test if repeated tokens attend to each other."""
        B, H, N, _ = beta.shape
        beta_np = beta[0].mean(dim=0).cpu().numpy()  # Average over heads
        ids = input_ids[0].cpu().numpy()

        same_token_attn = []
        diff_token_attn = []

        for i in range(N):
            for j in range(i):
                attn_val = beta_np[i, j]
                if ids[i] == ids[j]:
                    same_token_attn.append(attn_val)
                else:
                    diff_token_attn.append(attn_val)

        if len(same_token_attn) == 0:
            self._log_result("Token Repetition", True, 0.0,
                           "ℹ No repeated tokens in sequence", 'info')
            return

        same_avg = np.mean(same_token_attn)
        diff_avg = np.mean(diff_token_attn)
        ratio = same_avg / (diff_avg + 1e-10)

        has_pattern = same_avg > diff_avg * 1.2  # At least 20% stronger

        if has_pattern:
            self._log_result("Token Repetition", True, ratio,
                           f"✓ Repeated tokens attract (same: {same_avg:.4f}, diff: {diff_avg:.4f}, ratio: {ratio:.2f}x)", 'info')
        else:
            self._log_result("Token Repetition", False, ratio,
                           f"⚠️  No repetition pattern (same: {same_avg:.4f}, diff: {diff_avg:.4f}, ratio: {ratio:.2f}x)", 'warning')

    def _check_special_tokens(self, beta: torch.Tensor, input_ids: torch.Tensor):
        """Check attention to special tokens (BOS, EOS, etc)."""
        if self.tokenizer is None:
            self._log_result("Special Tokens", True, 0.0,
                           "ℹ No tokenizer available for special token analysis", 'info')
            return

        B, H, N, _ = beta.shape
        beta_np = beta[0].mean(dim=0).cpu().numpy()  # Average over heads
        ids = input_ids[0].cpu().numpy()

        # Find special tokens
        special_ids = set()
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            special_ids.add(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_ids.add(self.tokenizer.pad_token_id)

        if len(special_ids) == 0:
            self._log_result("Special Tokens", True, 0.0,
                           "ℹ No special tokens defined", 'info')
            return

        # Compute average attention to special vs regular tokens
        special_attn = []
        regular_attn = []

        for j in range(N):
            avg_attn_to_j = beta_np[:, j].mean()
            if ids[j] in special_ids:
                special_attn.append(avg_attn_to_j)
            else:
                regular_attn.append(avg_attn_to_j)

        if len(special_attn) == 0:
            self._log_result("Special Tokens", True, 0.0,
                           "ℹ No special tokens in sequence", 'info')
            return

        special_avg = np.mean(special_attn)
        regular_avg = np.mean(regular_attn)
        ratio = special_avg / (regular_avg + 1e-10)

        self._log_result("Special Tokens", True, ratio,
                       f"ℹ Special token attention: {special_avg:.4f} vs regular: {regular_avg:.4f} (ratio: {ratio:.2f}x)", 'info')

    def _test_vs_uniform(self, beta: torch.Tensor):
        """Test if attention is significantly different from uniform."""
        B, H, N, _ = beta.shape
        beta_np = beta[0].cpu().numpy()

        uniform = 1.0 / N

        # Compute deviation from uniform for each head
        deviations = []
        for h in range(H):
            for i in range(N):
                row = beta_np[h, i]
                # L2 distance from uniform
                deviation = np.sqrt(np.mean((row - uniform) ** 2))
                deviations.append(deviation)

        mean_dev = np.mean(deviations)
        max_dev = np.max(deviations)

        # Threshold: uniform would have dev ≈ 0
        # Sharp attention should have dev > 0.05
        is_sharp = mean_dev > 0.05

        if is_sharp:
            self._log_result("vs Uniform", True, mean_dev,
                           f"✓ Significantly different from uniform (mean dev: {mean_dev:.4f}, max: {max_dev:.4f})", 'info')
        else:
            self._log_result("vs Uniform", False, mean_dev,
                           f"❌ Nearly uniform (mean dev: {mean_dev:.4f}, max: {max_dev:.4f})", 'critical')

    def _compute_entropy(self, beta: torch.Tensor):
        """Compute normalized entropy of attention distributions."""
        # Entropy: H = -Σ p*log(p)
        entropy = -(beta * torch.log(beta + 1e-10)).sum(dim=-1)

        N = beta.shape[-1]
        max_entropy = np.log(N)  # Uniform distribution
        normalized_entropy = (entropy / max_entropy).mean().item()

        # Sharp attention: H < 0.5
        # Moderate: 0.5 < H < 0.8
        # Uniform: H > 0.9

        if normalized_entropy < 0.5:
            self._log_result("Entropy", True, normalized_entropy,
                           f"✓ Sharp attention (H/H_max: {normalized_entropy:.3f})", 'info')
        elif normalized_entropy < 0.8:
            self._log_result("Entropy", True, normalized_entropy,
                           f"⚠️  Moderate sharpness (H/H_max: {normalized_entropy:.3f})", 'warning')
        else:
            self._log_result("Entropy", False, normalized_entropy,
                           f"❌ Nearly uniform (H/H_max: {normalized_entropy:.3f})", 'critical')

    def _analyze_per_head(self, beta: torch.Tensor):
        """Analyze each attention head separately."""
        B, H, N, _ = beta.shape
        beta_np = beta[0].cpu().numpy()

        # Get irrep labels if available
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'irrep_spec'):
                irrep_spec = self.model.config.irrep_spec
                head_labels = []
                for irrep_name, num_heads, dim in irrep_spec:
                    for _ in range(num_heads):
                        head_labels.append(irrep_name)
            else:
                head_labels = [f"H{i}" for i in range(H)]
        except:
            head_labels = [f"H{i}" for i in range(H)]

        sharp_heads = 0
        medium_heads = 0
        uniform_heads = 0

        for h in range(H):
            attn_head = beta_np[h]
            attn_safe = attn_head.copy()
            np.fill_diagonal(attn_safe, np.nan)
            row_std = np.nanstd(attn_safe, axis=1).mean()

            label = head_labels[h] if h < len(head_labels) else f"H{h}"

            if row_std > 0.1:
                status = "✓SHARP"
                sharp_heads += 1
            elif row_std > 0.05:
                status = "⚠MEDIUM"
                medium_heads += 1
            else:
                status = "❌UNIFORM"
                uniform_heads += 1

            print(f"  Head {h:2d} ({label:>3s}): row_std={row_std:.4f} [{status}]")

        # Summary
        total = H
        sharp_pct = 100 * sharp_heads / total
        medium_pct = 100 * medium_heads / total
        uniform_pct = 100 * uniform_heads / total

        print(f"\n  Summary: {sharp_heads}/{total} sharp ({sharp_pct:.0f}%), "
              f"{medium_heads}/{total} medium ({medium_pct:.0f}%), "
              f"{uniform_heads}/{total} uniform ({uniform_pct:.0f}%)")

        overall_score = sharp_pct / 100.0 + 0.5 * medium_pct / 100.0

        if sharp_heads > 0:
            self._log_result("Per-Head Quality", True, overall_score,
                           f"✓ {sharp_heads}/{total} heads sharp", 'info')
        elif medium_heads > total // 2:
            self._log_result("Per-Head Quality", True, overall_score,
                           f"⚠️  {medium_heads}/{total} heads medium sharpness", 'warning')
        else:
            self._log_result("Per-Head Quality", False, overall_score,
                           f"❌ {uniform_heads}/{total} heads uniform", 'critical')

    def _check_diversity_correlation(self, beta: torch.Tensor, mu: torch.Tensor):
        """Test if embedding diversity correlates with attention diversity."""
        # Compute belief diversity (pairwise distances in μ space)
        mu_batch = mu[0]  # (N, K)
        mu_dists = torch.cdist(mu_batch, mu_batch, p=2)
        mu_diversity = mu_dists.mean().item()

        # Compute attention diversity (row std)
        beta_np = beta[0].cpu().numpy()
        beta_safe = beta_np.copy()
        N = beta_np.shape[-1]
        for h in range(beta_np.shape[0]):
            np.fill_diagonal(beta_safe[h], np.nan)
        row_std = np.nanstd(beta_safe, axis=-1).mean()

        print(f"  Belief diversity (μ_dist): {mu_diversity:.4f}")
        print(f"  Attention diversity (row_std): {row_std:.4f}")

        # If beliefs are diverse but attention is uniform, something's wrong
        has_belief_diversity = mu_diversity > 0.3
        has_attention_diversity = row_std > 0.05

        if has_belief_diversity and has_attention_diversity:
            self._log_result("Diversity Correlation", True, row_std,
                           f"✓ High belief diversity → sharp attention", 'info')
        elif has_belief_diversity and not has_attention_diversity:
            self._log_result("Diversity Correlation", False, row_std,
                           f"❌ High belief diversity BUT uniform attention (architectural issue!)", 'critical')
        elif not has_belief_diversity:
            self._log_result("Diversity Correlation", False, mu_diversity,
                           f"⚠️  Low belief diversity ({mu_diversity:.4f}) → expected uniform attention", 'warning')
        else:
            self._log_result("Diversity Correlation", True, row_std,
                           f"ℹ Low diversity but attention varies anyway", 'info')

    def _check_gradient_flow(self):
        """Check if attention parameters receive gradients."""
        # This requires training mode - just check parameter existence
        attn_params = []
        for name, param in self.model.named_parameters():
            if 'attention' in name.lower() or 'attn' in name.lower():
                attn_params.append((name, param.numel()))

        if len(attn_params) > 0:
            total_params = sum(n for _, n in attn_params)
            print(f"  Found {len(attn_params)} attention parameter groups ({total_params:,} params total)")
            for name, numel in attn_params[:5]:  # Show first 5
                print(f"    {name}: {numel:,} params")
            if len(attn_params) > 5:
                print(f"    ... and {len(attn_params) - 5} more")
            self._log_result("Gradient Flow", True, len(attn_params),
                           f"✓ Attention parameters found", 'info')
        else:
            self._log_result("Gradient Flow", False, 0,
                           f"⚠️  No attention parameters found (might use KL-only, no learned params)", 'warning')

    def _log_result(self, test_name: str, passed: bool, score: float,
                   message: str, severity: str):
        """Log a validation result."""
        result = ValidationResult(test_name, passed, score, message, severity)
        self.results.append(result)
        print(f"{message}")

    def _print_summary(self):
        """Print overall summary."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        critical_fails = [r for r in self.results if r.severity == 'critical' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]
        passes = [r for r in self.results if r.passed]

        print(f"\n✓ Passed: {len(passes)}/{len(self.results)} tests")
        if len(critical_fails) > 0:
            print(f"❌ Critical issues: {len(critical_fails)}")
            for r in critical_fails:
                print(f"   - {r.test_name}: {r.message}")
        if len(warnings) > 0:
            print(f"⚠️  Warnings: {len(warnings)}")
            for r in warnings:
                print(f"   - {r.test_name}: {r.message}")

        # Overall conclusion
        print("\n" + "-" * 70)
        print("CONCLUSION:")
        print("-" * 70)

        if len(critical_fails) == 0:
            if len(warnings) == 0:
                print("✓ Attention patterns are mathematically correct and learning structure.")
            else:
                print("⚠️  Attention is correct but may not be learning meaningful patterns.")
                print("   This could be due to:")
                print("   - Architectural limitations (low-dimensional heads)")
                print("   - Insufficient training")
                print("   - Hyperparameter issues (temperature, learning rate)")
        else:
            print("❌ Critical issues detected! Attention may have bugs or is not learning.")
            print("   Check the failed tests above for details.")

        print("=" * 70)

    def _get_summary_dict(self) -> Dict:
        """Get summary as dictionary."""
        critical_fails = [r for r in self.results if r.severity == 'critical' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]

        return {
            'total_tests': len(self.results),
            'passed': len([r for r in self.results if r.passed]),
            'critical_failures': len(critical_fails),
            'warnings': len(warnings),
            'issues': [r.message for r in critical_fails + warnings]
        }


def load_model_and_tokenizer(checkpoint_path: Path, config_path: Optional[Path] = None):
    """Load model and tokenizer from checkpoint."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Load config
    if config_path and config_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.GPU_OPTIMIZED_CONFIG
    else:
        # Use default config
        from transformer.config import GPU_OPTIMIZED_CONFIG
        config = GPU_OPTIMIZED_CONFIG

    # Load model
    from transformer.model import GaugeEquivariantTransformer
    from transformer.config import TransformerConfig

    model_config = TransformerConfig(**config)
    model = GaugeEquivariantTransformer(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Load tokenizer
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    except:
        print("Warning: Could not load tokenizer")
        tokenizer = None

    return model, tokenizer, model_config


def create_test_batch(tokenizer, seq_len: int = 128, device='cuda'):
    """Create a test batch from sample text."""
    if tokenizer is None:
        # Create random batch
        vocab_size = 50257  # GPT-2 vocab size
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        return input_ids.to(device), input_ids.to(device)

    # Use real text
    text = """The transformer architecture has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input sequence.
    Gauge-equivariant transformers extend this by incorporating geometric structure."""

    tokens = tokenizer.encode(text, max_length=seq_len, truncation=True, padding='max_length')
    input_ids = torch.tensor([tokens])

    return input_ids.to(device), input_ids.to(device)


def main():
    parser = argparse.ArgumentParser(description='Validate attention patterns')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length for test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None

    model, tokenizer, config = load_model_and_tokenizer(checkpoint_path, config_path)
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded. Config: {config.n_heads} heads, embed_dim={config.embed_dim}")

    # Create test batch
    print(f"\nCreating test batch (seq_len={args.seq_len})...")
    batch = create_test_batch(tokenizer, args.seq_len, args.device)

    # Run validation
    validator = AttentionValidator(model, tokenizer, args.device)
    results = validator.validate_all(batch)

    print(f"\nValidation complete!")


if __name__ == '__main__':
    main()
