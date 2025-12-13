#!/usr/bin/env python3
"""
Benchmark Script for Pure FEP Transformer Optimizations
========================================================

Tests the performance impact of:
1. Transport caching (cache_transport=True)
2. Fast matrix exponential (use_fast_matrix_exp=True)
3. VFE momentum (use_vfe_momentum=True)
4. Local attention (use_local_attention=True)

Usage:
    python transformer/benchmark_optimizations.py
"""

import torch
import time
from contextlib import contextmanager
from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.1f}ms")


def benchmark_config(config: PureFEPConfig, n_warmup: int = 2, n_runs: int = 5) -> float:
    """
    Benchmark a configuration and return average time per forward pass.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PureFEPTransformer(config).to(device)
    model.eval()

    B, N = 2, 12
    input_ids = torch.randint(0, config.vocab_size, (B, N), device=device)
    targets = torch.randint(0, config.vocab_size, (B, N), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids, targets=targets, n_vfe_steps=config.belief_steps)

    # Synchronize before timing (for CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, targets=targets, n_vfe_steps=config.belief_steps)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    return avg_time


def main():
    print("=" * 60)
    print("Pure FEP Transformer Optimization Benchmark")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Base configuration (matching user's reported settings)
    base_params = dict(
        embed_dim=127,
        num_layers=2,
        seq_length=24,
        vocab_size=1000,
        belief_steps=20,  # Test with fewer steps for faster benchmark
        pure_fep_mode=True,
    )

    print(f"\nBase config: K={base_params['embed_dim']}, N=12, B=2, VFE_steps={base_params['belief_steps']}")
    print()

    # Test 1: Baseline (no optimizations)
    print("=" * 60)
    print("Test 1: Baseline (no optimizations)")
    print("=" * 60)
    config_baseline = PureFEPConfig(
        **base_params,
        cache_transport=False,
        use_fast_matrix_exp=False,
        use_vfe_momentum=False,
        use_local_attention=False,
    )
    time_baseline = benchmark_config(config_baseline)
    print(f"  Time: {time_baseline*1000:.1f}ms per forward pass")

    # Test 2: Transport caching only
    print("\n" + "=" * 60)
    print("Test 2: Transport caching only")
    print("=" * 60)
    config_cache = PureFEPConfig(
        **base_params,
        cache_transport=True,
        use_fast_matrix_exp=False,
        use_vfe_momentum=False,
        use_local_attention=False,
    )
    time_cache = benchmark_config(config_cache)
    speedup = time_baseline / time_cache
    print(f"  Time: {time_cache*1000:.1f}ms per forward pass")
    print(f"  Speedup: {speedup:.2f}x")

    # Test 3: Fast matrix exp only
    print("\n" + "=" * 60)
    print("Test 3: Fast matrix exponential only")
    print("=" * 60)
    config_fast_exp = PureFEPConfig(
        **base_params,
        cache_transport=False,
        use_fast_matrix_exp=True,
        use_vfe_momentum=False,
        use_local_attention=False,
    )
    time_fast_exp = benchmark_config(config_fast_exp)
    speedup = time_baseline / time_fast_exp
    print(f"  Time: {time_fast_exp*1000:.1f}ms per forward pass")
    print(f"  Speedup: {speedup:.2f}x")

    # Test 4: Cache + Fast exp
    print("\n" + "=" * 60)
    print("Test 4: Transport caching + Fast matrix exp")
    print("=" * 60)
    config_cache_fast = PureFEPConfig(
        **base_params,
        cache_transport=True,
        use_fast_matrix_exp=True,
        use_vfe_momentum=False,
        use_local_attention=False,
    )
    time_cache_fast = benchmark_config(config_cache_fast)
    speedup = time_baseline / time_cache_fast
    print(f"  Time: {time_cache_fast*1000:.1f}ms per forward pass")
    print(f"  Speedup: {speedup:.2f}x")

    # Test 5: All optimizations
    print("\n" + "=" * 60)
    print("Test 5: ALL optimizations (cache + fast exp + momentum)")
    print("=" * 60)
    config_all = PureFEPConfig(
        **base_params,
        cache_transport=True,
        use_fast_matrix_exp=True,
        use_vfe_momentum=True,
        use_local_attention=False,  # Keep full attention for fair comparison
    )
    time_all = benchmark_config(config_all)
    speedup = time_baseline / time_all
    print(f"  Time: {time_all*1000:.1f}ms per forward pass")
    print(f"  Speedup: {speedup:.2f}x")

    # Test 6: All + Local attention
    print("\n" + "=" * 60)
    print("Test 6: ALL optimizations + Local attention (window=8)")
    print("=" * 60)
    config_local = PureFEPConfig(
        **base_params,
        cache_transport=True,
        use_fast_matrix_exp=True,
        use_vfe_momentum=True,
        use_local_attention=True,
        attention_window=8,
    )
    time_local = benchmark_config(config_local)
    speedup = time_baseline / time_local
    print(f"  Time: {time_local*1000:.1f}ms per forward pass")
    print(f"  Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline:                {time_baseline*1000:.1f}ms")
    print(f"  + Transport caching:     {time_cache*1000:.1f}ms ({time_baseline/time_cache:.2f}x)")
    print(f"  + Fast matrix exp:       {time_fast_exp*1000:.1f}ms ({time_baseline/time_fast_exp:.2f}x)")
    print(f"  + Cache + Fast exp:      {time_cache_fast*1000:.1f}ms ({time_baseline/time_cache_fast:.2f}x)")
    print(f"  + All optimizations:     {time_all*1000:.1f}ms ({time_baseline/time_all:.2f}x)")
    print(f"  + All + Local attention: {time_local*1000:.1f}ms ({time_baseline/time_local:.2f}x)")

    # Check VFE convergence with momentum
    print("\n" + "=" * 60)
    print("VFE Convergence Test: Momentum vs No Momentum")
    print("=" * 60)

    for use_momentum in [False, True]:
        config = PureFEPConfig(
            **base_params,
            cache_transport=True,
            use_fast_matrix_exp=True,
            use_vfe_momentum=use_momentum,
        )
        model = PureFEPTransformer(config).to(device)

        B, N = 2, 12
        input_ids = torch.randint(0, config.vocab_size, (B, N), device=device)
        targets = torch.randint(0, config.vocab_size, (B, N), device=device)

        with torch.no_grad():
            logits, info = model(input_ids, targets=targets, n_vfe_steps=20)

        metrics = info['metrics']
        momentum_str = "with momentum" if use_momentum else "no momentum"
        print(f"  {momentum_str}: VFE loss = {metrics['vfe_loss']:.4f}, CE = {metrics['ce_loss']:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
