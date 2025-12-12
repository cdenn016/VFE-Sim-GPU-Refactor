# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:08:01 2025

@author: chris and christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Transition Experiment: κ-Sweep for Meta-Agent Emergence
==============================================================

This experiment sweeps the temperature parameter κ (kappa_beta) from high to low
values, measuring RG metrics at each point to detect the critical temperature κ*
where meta-agent structure emerges.

Theory Prediction
-----------------
The VFE-RG correspondence predicts a phase transition at critical κ*:

    κ > κ* (high temp): Disordered phase
        - Low modularity Q ≈ 0
        - High effective rank ≈ N
        - Uniform attention (high entropy)
        - No clusters

    κ < κ* (low temp): Ordered phase
        - High modularity Q > 0
        - Low effective rank << N
        - Sparse/focused attention (low entropy)
        - Distinct clusters (meta-agents)

At κ = κ*, we expect:
    - Maximum susceptibility dQ/dκ
    - Sharp change in modularity
    - Rapid cluster formation

Information Bottleneck Connection
---------------------------------
The correspondence κ ↔ 1/β_IB connects this to the IB phase diagram:
    - High κ (low β_IB): Compression dominates → trivial representations
    - Low κ (high β_IB): Relevance dominates → complex representations
    - Critical κ*: Optimal tradeoff → emergent structure

Usage
-----
    python experiments/phase_transition_sweep.py
    python experiments/phase_transition_sweep.py --kappa_min 0.1 --kappa_max 10.0 --n_points 50
    python experiments/phase_transition_sweep.py --preset emergence --n_steps 200

Author: Chris and Christine
Date: December 2025
"""

import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation_config import SimulationConfig
from simulation_runner import (
    build_manifold,
    build_supports,
    build_agents,
    build_system,
)
from experiments.rg_simulation_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
    extract_beta_matrix,
    compute_modularity,
    compute_effective_rank,
    detect_clusters_spectral,
)


@dataclass
class PhaseSweepConfig:
    """Configuration for phase transition sweep experiment."""

    # Kappa sweep parameters
    kappa_min: float = 0.05
    kappa_max: float = 10.0
    n_kappa_points: int = 30
    kappa_scale: str = 'log'  # 'linear' or 'log'

    # Training at each κ
    n_steps: int = 100  # Steps to equilibrate at each κ
    n_warmup: int = 20  # Warmup steps (discard for metrics)
    n_samples: int = 5  # Samples to average after equilibration

    # Agent configuration
    n_agents: int = 12
    K_latent: int = 8

    # System configuration
    lambda_self: float = 1.0
    lambda_belief_align: float = 1.0
    lambda_prior_align: float = 0.0
    lambda_obs: float = 0.0
    lambda_phi: float = 1.0

    # Output
    output_dir: str = "_results/phase_transition"
    experiment_name: str = "kappa_sweep"
    seed: int = 42

    # Advanced
    reuse_system: bool = True  # Keep system between κ values (adiabatic sweep)
    bidirectional: bool = True  # Sweep κ down then up (hysteresis check)

    def get_kappa_values(self) -> np.ndarray:
        """Generate κ values for sweep."""
        if self.kappa_scale == 'log':
            return np.logspace(
                np.log10(self.kappa_min),
                np.log10(self.kappa_max),
                self.n_kappa_points
            )
        else:
            return np.linspace(self.kappa_min, self.kappa_max, self.n_kappa_points)


@dataclass
class PhaseSweepResults:
    """Results from phase transition sweep."""
    kappa_values: np.ndarray = None
    modularity: np.ndarray = None
    modularity_std: np.ndarray = None
    effective_rank: np.ndarray = None
    effective_rank_std: np.ndarray = None
    n_clusters: np.ndarray = None
    beta_entropy: np.ndarray = None
    kl_within: np.ndarray = None
    kl_between: np.ndarray = None

    # Phase transition analysis
    critical_kappa: float = None
    transition_sharpness: float = None
    susceptibility: np.ndarray = None

    def save(self, filepath: str):
        """Save results to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'PhaseSweepResults':
        """Load results from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def run_single_kappa(
    system,
    kappa: float,
    cfg: PhaseSweepConfig,
) -> Tuple[RGDiagnostics, List[RGDiagnostics]]:
    """
    Run simulation at a single κ value and collect RG metrics.

    Args:
        system: Multi-agent system
        kappa: Temperature parameter
        cfg: Experiment configuration

    Returns:
        final_diagnostics: RG metrics at equilibrium
        sample_diagnostics: List of sampled diagnostics for error estimation
    """
    from gradients.gradient_engine import compute_natural_gradients
    from gradients.free_energy_clean import compute_total_free_energy
    from meta.gradient_adapter import GradientSystemAdapter

    # Update kappa in system config
    system.config.kappa_beta = kappa
    system.config.kappa_gamma = kappa

    # Get agents
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        # MultiScaleSystem
        agents = system.agents.get(0, [])
    else:
        # MultiAgentSystem
        agents = system.agents

    # Training loop
    sample_diagnostics = []

    for step in range(cfg.n_steps):
        # Create adapter
        adapter = GradientSystemAdapter(agents, system.config)

        # Compute gradients
        agent_grads = compute_natural_gradients(adapter)

        # Apply gradient updates (simplified - no momentum)
        lr = 0.01  # Small learning rate for equilibration
        for agent, grads in zip(agents, agent_grads):
            agent.mu_q = agent.mu_q - lr * grads.grad_mu_q
            # Clamp sigma to stay positive
            agent.Sigma_q = agent.Sigma_q - lr * grads.grad_Sigma_q
            agent.Sigma_q = np.maximum(agent.Sigma_q, 1e-4)

        # Collect samples after warmup
        if step >= cfg.n_warmup:
            sample_interval = (cfg.n_steps - cfg.n_warmup) // cfg.n_samples
            if sample_interval > 0 and (step - cfg.n_warmup) % sample_interval == 0:
                diag = compute_rg_diagnostics(
                    system=adapter,
                    step=step,
                    auto_cluster=True,
                    mode='belief',
                )
                sample_diagnostics.append(diag)

    # Final diagnostics
    adapter = GradientSystemAdapter(agents, system.config)
    final_diag = compute_rg_diagnostics(
        system=adapter,
        step=cfg.n_steps,
        auto_cluster=True,
        mode='belief',
    )

    return final_diag, sample_diagnostics


def run_phase_sweep(cfg: PhaseSweepConfig) -> PhaseSweepResults:
    """
    Run full phase transition sweep over κ values.

    Args:
        cfg: Experiment configuration

    Returns:
        PhaseSweepResults with all metrics
    """
    print("=" * 70)
    print("PHASE TRANSITION SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"  κ range: [{cfg.kappa_min}, {cfg.kappa_max}]")
    print(f"  Points: {cfg.n_kappa_points}")
    print(f"  Scale: {cfg.kappa_scale}")
    print(f"  Steps per κ: {cfg.n_steps}")
    print(f"  Agents: {cfg.n_agents}")
    print("=" * 70)

    # Setup
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create simulation config (for building system)
    sim_cfg = SimulationConfig(
        experiment_name=cfg.experiment_name,
        n_agents=cfg.n_agents,
        K_latent=cfg.K_latent,
        lambda_self=cfg.lambda_self,
        lambda_belief_align=cfg.lambda_belief_align,
        lambda_prior_align=cfg.lambda_prior_align,
        lambda_obs=cfg.lambda_obs,
        lambda_phi=cfg.lambda_phi,
        kappa_beta=cfg.kappa_max,  # Start at high κ
        kappa_gamma=cfg.kappa_max,
        n_steps=cfg.n_steps,
        enable_emergence=False,  # Flat system for phase transition
        seed=cfg.seed,
    )

    # Build system
    manifold = build_manifold(sim_cfg)
    supports = build_supports(manifold, sim_cfg, rng)
    agents = build_agents(manifold, supports, sim_cfg, rng)
    system = build_system(agents, sim_cfg, rng)

    # Get κ values (descending for physical interpretation)
    kappa_values = cfg.get_kappa_values()[::-1]  # High to low

    # Storage
    n_points = len(kappa_values)
    modularity = np.zeros(n_points)
    modularity_std = np.zeros(n_points)
    effective_rank = np.zeros(n_points)
    effective_rank_std = np.zeros(n_points)
    n_clusters = np.zeros(n_points)
    beta_entropy = np.zeros(n_points)
    kl_within = np.zeros(n_points)
    kl_between = np.zeros(n_points)

    # Sweep
    print("\nRunning κ sweep (high → low)...")
    for i, kappa in enumerate(kappa_values):
        print(f"  [{i+1}/{n_points}] κ = {kappa:.4f}", end="", flush=True)

        # Run at this κ
        final_diag, samples = run_single_kappa(system, kappa, cfg)

        # Store final metrics
        modularity[i] = final_diag.modularity
        effective_rank[i] = final_diag.effective_rank
        n_clusters[i] = final_diag.n_clusters
        beta_entropy[i] = final_diag.beta_entropy
        kl_within[i] = final_diag.kl_within_mean
        kl_between[i] = final_diag.kl_between_mean

        # Compute std from samples
        if len(samples) > 1:
            modularity_std[i] = np.std([s.modularity for s in samples])
            effective_rank_std[i] = np.std([s.effective_rank for s in samples])
        else:
            modularity_std[i] = 0.0
            effective_rank_std[i] = 0.0

        print(f" | Q={final_diag.modularity:.4f} | rank={final_diag.effective_rank:.2f} | "
              f"clusters={final_diag.n_clusters}")

        # Reset system for next κ if not reusing
        if not cfg.reuse_system:
            # Reinitialize agent states
            for agent in agents:
                agent._initialize_belief_covariance()
                agent._initialize_prior_covariance()

    # Reverse to get ascending κ
    kappa_values = kappa_values[::-1]
    modularity = modularity[::-1]
    modularity_std = modularity_std[::-1]
    effective_rank = effective_rank[::-1]
    effective_rank_std = effective_rank_std[::-1]
    n_clusters = n_clusters[::-1]
    beta_entropy = beta_entropy[::-1]
    kl_within = kl_within[::-1]
    kl_between = kl_between[::-1]

    # Compute susceptibility (dQ/dκ)
    susceptibility = np.gradient(modularity, kappa_values)

    # Find critical κ (maximum susceptibility)
    critical_idx = np.argmax(np.abs(susceptibility))
    critical_kappa = kappa_values[critical_idx]
    transition_sharpness = np.abs(susceptibility[critical_idx])

    print(f"\n" + "=" * 70)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 70)
    print(f"  Critical κ*: {critical_kappa:.4f}")
    print(f"  Transition sharpness: {transition_sharpness:.4f}")
    print(f"  Max susceptibility: {np.max(np.abs(susceptibility)):.4f}")
    print(f"  Modularity range: [{modularity.min():.4f}, {modularity.max():.4f}]")
    print(f"  Effective rank range: [{effective_rank.min():.2f}, {effective_rank.max():.2f}]")
    print("=" * 70)

    # Create results
    results = PhaseSweepResults(
        kappa_values=kappa_values,
        modularity=modularity,
        modularity_std=modularity_std,
        effective_rank=effective_rank,
        effective_rank_std=effective_rank_std,
        n_clusters=n_clusters,
        beta_entropy=beta_entropy,
        kl_within=kl_within,
        kl_between=kl_between,
        critical_kappa=critical_kappa,
        transition_sharpness=transition_sharpness,
        susceptibility=susceptibility,
    )

    # Save results
    results.save(str(output_dir / "phase_sweep_results.pkl"))
    print(f"\n✓ Saved results to {output_dir / 'phase_sweep_results.pkl'}")

    # Generate plots
    plot_phase_transition(results, output_dir)

    return results


def plot_phase_transition(results: PhaseSweepResults, output_dir: Path):
    """Generate phase transition plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    kappa = results.kappa_values

    # 1. Modularity vs κ
    ax = axes[0, 0]
    ax.errorbar(kappa, results.modularity, yerr=results.modularity_std,
                fmt='b-o', markersize=4, capsize=3)
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7,
               label=f'κ* = {results.critical_kappa:.3f}')
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('Modularity Q')
    ax.set_title('Order Parameter: Modularity')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Susceptibility (dQ/dκ)
    ax = axes[0, 1]
    ax.plot(kappa, np.abs(results.susceptibility), 'r-o', markersize=4)
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('|dQ/dκ| (susceptibility)')
    ax.set_title('Phase Transition Indicator')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # 3. Effective Rank vs κ
    ax = axes[0, 2]
    ax.errorbar(kappa, results.effective_rank, yerr=results.effective_rank_std,
                fmt='g-o', markersize=4, capsize=3)
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Degrees of Freedom')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # 4. Number of Clusters vs κ
    ax = axes[1, 0]
    ax.plot(kappa, results.n_clusters, 'purple', marker='s', markersize=4)
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('# Clusters')
    ax.set_title('Meta-Agent Count')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # 5. Beta Entropy vs κ
    ax = axes[1, 1]
    ax.plot(kappa, results.beta_entropy, 'orange', marker='^', markersize=4)
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('β Entropy')
    ax.set_title('Attention Entropy')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)

    # 6. KL Within/Between vs κ
    ax = axes[1, 2]
    ax.plot(kappa, results.kl_within, 'm-o', markersize=4, label='KL within')
    ax.plot(kappa, results.kl_between, 'c-s', markersize=4, label='KL between')
    ax.axvline(results.critical_kappa, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('κ (temperature)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Cluster Coherence')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Add overall title
    fig.suptitle(f'Phase Transition in VFE System (κ* = {results.critical_kappa:.4f})',
                 fontsize=14, y=1.02)

    fig_path = output_dir / "phase_transition.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved phase transition plot to {fig_path}")

    # Additional: Phase diagram (Modularity vs Entropy)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(results.beta_entropy, results.modularity,
                        c=np.log10(kappa), cmap='coolwarm', s=50)
    ax.set_xlabel('β Entropy (disorder)')
    ax.set_ylabel('Modularity Q (order)')
    ax.set_title('Phase Diagram: Order vs Disorder')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(κ)')
    ax.grid(alpha=0.3)

    fig_path = output_dir / "phase_diagram.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"✓ Saved phase diagram to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase transition sweep experiment for VFE-RG theory validation"
    )

    # Kappa sweep
    parser.add_argument('--kappa_min', type=float, default=0.05,
                       help='Minimum κ value')
    parser.add_argument('--kappa_max', type=float, default=10.0,
                       help='Maximum κ value')
    parser.add_argument('--n_points', type=int, default=30,
                       help='Number of κ values to sample')
    parser.add_argument('--kappa_scale', type=str, default='log',
                       choices=['linear', 'log'],
                       help='Scale for κ sampling')

    # Training
    parser.add_argument('--n_steps', type=int, default=100,
                       help='Steps per κ value')
    parser.add_argument('--n_warmup', type=int, default=20,
                       help='Warmup steps (discarded)')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Samples for error estimation')

    # System
    parser.add_argument('--n_agents', type=int, default=12,
                       help='Number of agents')
    parser.add_argument('--K', type=int, default=8,
                       help='Latent dimension')

    # Output
    parser.add_argument('--output_dir', type=str, default='_results/phase_transition',
                       help='Output directory')
    parser.add_argument('--name', type=str, default='kappa_sweep',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Options
    parser.add_argument('--no_reuse', action='store_true',
                       help='Reset system between κ values')

    args = parser.parse_args()

    # Create config
    cfg = PhaseSweepConfig(
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        n_kappa_points=args.n_points,
        kappa_scale=args.kappa_scale,
        n_steps=args.n_steps,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        n_agents=args.n_agents,
        K_latent=args.K,
        output_dir=args.output_dir,
        experiment_name=args.name,
        seed=args.seed,
        reuse_system=not args.no_reuse,
    )

    # Run experiment
    results = run_phase_sweep(cfg)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results: {Path(cfg.output_dir) / cfg.experiment_name}")
    print(f"Critical κ*: {results.critical_kappa:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()