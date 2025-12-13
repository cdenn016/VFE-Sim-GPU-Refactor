# -*- coding: utf-8 -*-
"""
RG Metrics for Multi-Agent VFE Simulations
============================================

Wrapper around transformer/rg_metrics.py for multi-agent systems.

This module re-exports the core RG metrics from transformer/rg_metrics.py
and adds multi-agent specific utilities:
- extract_beta_matrix: Extract coupling matrix from MultiAgentSystem
- extract_belief_params: Extract belief parameters from agents
- compute_rg_diagnostics: System-level RG diagnostics (takes system object)

The core metrics (compute_modularity, compute_effective_rank, etc.) now
support both NumPy and PyTorch inputs via automatic conversion.

Author: Chris and Christine
Date: December 2025
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# Re-export everything from canonical implementation
from transformer.rg_metrics import (
    # Data structures
    RGDiagnostics,
    RGFlowSummary,
    # Core metrics (now accept NumPy or PyTorch)
    compute_modularity,
    compute_effective_rank,
    compute_beta_entropy,
    detect_clusters_spectral,
    compute_kl_within_clusters,
    compute_kl_between_clusters,
    # Analysis
    analyze_rg_flow,
    # Helpers
    _to_tensor,
    _to_numpy,
)


# =============================================================================
# Multi-Agent Specific: Extract Data from System Objects
# =============================================================================

def extract_beta_matrix(
    system,
    mode: str = 'belief',
) -> np.ndarray:
    """
    Extract full coupling matrix β from multi-agent system.

    Constructs N×N matrix where β[i,j] is the coupling weight from agent i to j.
    Uses softmax weights computed via compute_softmax_weights().

    Args:
        system: MultiAgentSystem or GradientSystemAdapter
        mode: 'belief' or 'prior' (which coupling weights to use)

    Returns:
        beta: (N, N) coupling matrix
    """
    from gradients.softmax_grads import compute_softmax_weights

    # Get number of agents
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        # MultiScaleSystem - get base agents
        agents = system.agents.get(0, [])
        n_agents = len(agents)
    elif hasattr(system, 'agents') and isinstance(system.agents, list):
        # MultiAgentSystem or GradientSystemAdapter
        n_agents = len(system.agents)
    else:
        n_agents = system.n_agents

    if n_agents == 0:
        return np.array([[]])

    # Get kappa from config
    if hasattr(system, 'config'):
        kappa = system.config.kappa_beta if mode == 'belief' else system.config.kappa_gamma
    elif hasattr(system, 'system_config'):
        kappa = system.system_config.kappa_beta if mode == 'belief' else system.system_config.kappa_gamma
    else:
        kappa = 1.0  # Default

    # Build full β matrix
    beta = np.zeros((n_agents, n_agents), dtype=np.float32)

    for i in range(n_agents):
        try:
            # Get softmax weights for agent i
            beta_fields = compute_softmax_weights(system, i, mode=mode, kappa=kappa)

            for j, weight_field in beta_fields.items():
                # For 0D manifold, weight_field is scalar or 0D array
                # For ND, take spatial mean
                if np.isscalar(weight_field):
                    beta[i, j] = weight_field
                elif weight_field.size == 1:
                    beta[i, j] = float(weight_field.flat[0])
                else:
                    beta[i, j] = float(weight_field.mean())
        except Exception:
            # Skip agents with no neighbors
            continue

    return beta


def extract_belief_params(system) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract belief parameters (μ, Σ) from all agents.

    For spatial agents, takes the mean over spatial dimensions.

    Returns:
        mu: (N, K) belief means
        sigma: (N, K) diagonal variances or (N, K, K) full covariances
    """
    if hasattr(system, 'agents') and isinstance(system.agents, dict):
        agents = system.agents.get(0, [])
    elif hasattr(system, 'agents'):
        agents = system.agents
    else:
        return np.array([]), np.array([])

    if len(agents) == 0:
        return np.array([]), np.array([])

    mus = []
    sigmas = []

    for agent in agents:
        mu = agent.mu_q
        sigma = agent.Sigma_q

        # Handle spatial dimensions
        if mu.ndim > 1:
            # Average over spatial dimensions, keep K
            mu = mu.reshape(-1, mu.shape[-1]).mean(axis=0)

        if sigma.ndim > 2:
            # Average covariance over spatial dimensions
            sigma = sigma.reshape(-1, sigma.shape[-2], sigma.shape[-1]).mean(axis=0)

        mus.append(mu)
        sigmas.append(sigma)

    return np.stack(mus), np.stack(sigmas)


# =============================================================================
# System-Level RG Diagnostics (Multi-Agent Specific)
# =============================================================================

def compute_rg_diagnostics(
    system,
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
    mode: str = 'belief',
) -> RGDiagnostics:
    """
    Compute full RG diagnostics for multi-agent system.

    This is the multi-agent version that takes a system object
    and extracts beta/mu/sigma automatically.

    Args:
        system: MultiAgentSystem, MultiScaleSystem, or GradientSystemAdapter
        step: Training iteration number
        auto_cluster: Auto-detect clusters if True
        n_clusters: Fixed number of clusters (if not auto)
        mode: 'belief' or 'prior' coupling weights

    Returns:
        RGDiagnostics with all metrics
    """
    # Extract coupling matrix
    beta = extract_beta_matrix(system, mode=mode)

    if beta.size == 0:
        return RGDiagnostics(
            step=step,
            modularity=0.0,
            effective_rank=0.0,
            n_clusters=0,
        )

    # Extract belief parameters
    mu, sigma = extract_belief_params(system)

    # Detect clusters (uses canonical implementation which auto-converts NumPy)
    cluster_labels = detect_clusters_spectral(beta, n_clusters=n_clusters)

    # Convert back to numpy for storage
    cluster_labels_np = _to_numpy(cluster_labels)

    # Compute metrics (all accept NumPy now)
    modularity = compute_modularity(beta, cluster_labels)
    effective_rank = compute_effective_rank(beta)
    beta_entropy = compute_beta_entropy(beta)

    # KL statistics (only if we have belief data)
    if mu.size > 0:
        kl_within_mean, kl_within_std = compute_kl_within_clusters(
            mu, sigma, cluster_labels
        )
        kl_between_mean, kl_between_std = compute_kl_between_clusters(
            mu, sigma, cluster_labels
        )
    else:
        kl_within_mean, kl_within_std = 0.0, 0.0
        kl_between_mean, kl_between_std = 0.0, 0.0

    # Cluster sizes
    unique_clusters = np.unique(cluster_labels_np)
    n_clusters_detected = len(unique_clusters)
    meta_agent_sizes = [
        int((cluster_labels_np == c).sum())
        for c in unique_clusters
    ]

    return RGDiagnostics(
        step=step,
        modularity=modularity,
        effective_rank=effective_rank,
        n_clusters=n_clusters_detected,
        cluster_labels=cluster_labels_np,
        kl_within_mean=kl_within_mean,
        kl_within_std=kl_within_std,
        kl_between_mean=kl_between_mean,
        kl_between_std=kl_between_std,
        beta_entropy=beta_entropy,
        meta_agent_sizes=meta_agent_sizes,
    )


# =============================================================================
# Information Bottleneck Metrics (Multi-Agent Specific)
# =============================================================================

def kl_gaussian(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute KL(q1 || q2) for Gaussians.

    For diagonal covariance:
        KL = 0.5 * [Σ(σ2/σ1) + Σ((μ1-μ2)²/σ1) - K + Σ log(σ1/σ2)]
    """
    from scipy import linalg

    K = mu1.shape[-1]

    if sigma1.ndim == 1:
        # Diagonal
        sigma1_safe = np.clip(sigma1, eps, None)
        sigma2_safe = np.clip(sigma2, eps, None)

        kl = 0.5 * (
            (sigma1_safe / sigma2_safe).sum()
            + ((mu2 - mu1) ** 2 / sigma2_safe).sum()
            - K
            + (np.log(sigma2_safe) - np.log(sigma1_safe)).sum()
        )
    else:
        # Full covariance
        sigma1 = sigma1 + eps * np.eye(K)
        sigma2 = sigma2 + eps * np.eye(K)

        sigma2_inv = linalg.inv(sigma2)

        kl = 0.5 * (
            np.trace(sigma2_inv @ sigma1)
            + (mu2 - mu1) @ sigma2_inv @ (mu2 - mu1)
            - K
            + np.log(linalg.det(sigma2)) - np.log(linalg.det(sigma1))
        )

    return float(kl)


def estimate_mutual_information_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Estimate mutual information I(X;T) for Gaussian representations.

    Args:
        mu: (N, K) belief means (representations T)
        sigma: (N, K) or (N, K, K) covariances

    Returns:
        Estimated mutual information (bits)
    """
    # Variance of means across agents (spread in representation space)
    mu_var = np.var(mu, axis=0).sum()

    # Mean variance (average uncertainty)
    if sigma.ndim == 2:
        sigma_mean = sigma.mean(axis=0).sum()
    else:
        sigma_mean = np.trace(sigma.mean(axis=0))

    # I(X;T) ≈ 0.5 * log(1 + var(μ)/mean(σ))
    ratio = mu_var / (sigma_mean + eps)
    I_XT = 0.5 * np.log(1 + ratio) / np.log(2)  # Convert to bits

    return float(I_XT)


def estimate_relevance_information(
    mu: np.ndarray,
    cluster_labels: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Estimate mutual information I(T;Y) between representations and targets.

    Args:
        mu: (N, K) belief means
        cluster_labels: (N,) cluster assignments

    Returns:
        Estimated mutual information I(T;Y) in bits
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    N = len(mu)

    if n_clusters <= 1:
        return 0.0

    # Compute cluster centroids
    centroids = np.array([
        mu[cluster_labels == c].mean(axis=0)
        for c in unique_clusters
    ])

    # Between-cluster variance
    global_mean = mu.mean(axis=0)
    between_var = np.array([
        (cluster_labels == c).sum() * np.sum((centroids[i] - global_mean) ** 2)
        for i, c in enumerate(unique_clusters)
    ]).sum() / N

    # Within-cluster variance
    within_var = np.array([
        np.sum((mu[cluster_labels == c] - centroids[i]) ** 2)
        for i, c in enumerate(unique_clusters)
    ]).sum() / N

    # I(T;Y) ≈ 0.5 * log(1 + between/within)
    ratio = between_var / (within_var + eps)
    I_TY = 0.5 * np.log(1 + ratio) / np.log(2)

    return float(I_TY)


def compute_ib_metrics(
    mu: np.ndarray,
    sigma: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict:
    """
    Compute Information Bottleneck metrics.

    Args:
        mu: (N, K) belief means
        sigma: (N, K) or (N, K, K) covariances
        cluster_labels: (N,) cluster assignments

    Returns:
        Dictionary with IB metrics
    """
    I_XT = estimate_mutual_information_gaussian(mu, sigma)
    I_TY = estimate_relevance_information(mu, cluster_labels)

    return {
        'I_XT': I_XT,
        'I_TY': I_TY,
        'ib_ratio': I_TY / (I_XT + 1e-10),
        'sufficiency_gap': I_XT - I_TY,
    }


@dataclass
class IBDiagnostics:
    """Information Bottleneck diagnostics at each step."""
    step: int
    I_XT: float  # Compression
    I_TY: float  # Relevance
    ib_ratio: float  # Efficiency
    sufficiency_gap: float  # Excess compression

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'I_XT': self.I_XT,
            'I_TY': self.I_TY,
            'ib_ratio': self.ib_ratio,
            'sufficiency_gap': self.sufficiency_gap,
        }


@dataclass
class IBFlowSummary:
    """Summary of IB metrics across training."""
    I_XT_history: List[float] = field(default_factory=list)
    I_TY_history: List[float] = field(default_factory=list)
    ib_ratio_history: List[float] = field(default_factory=list)
    sufficiency_gap_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)

    def add_step(self, ib_diag: IBDiagnostics, kappa: float = None):
        """Add IB diagnostics from a single step."""
        self.I_XT_history.append(ib_diag.I_XT)
        self.I_TY_history.append(ib_diag.I_TY)
        self.ib_ratio_history.append(ib_diag.ib_ratio)
        self.sufficiency_gap_history.append(ib_diag.sufficiency_gap)
        if kappa is not None:
            self.kappa_history.append(kappa)

    def get_information_plane_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (I_XT, I_TY) trajectory for plotting on information plane."""
        return np.array(self.I_XT_history), np.array(self.I_TY_history)


def compute_full_diagnostics(
    system,
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
    mode: str = 'belief',
) -> Tuple[RGDiagnostics, IBDiagnostics]:
    """
    Compute full RG + IB diagnostics.

    Args:
        system: Multi-agent system
        step: Training iteration
        auto_cluster: Auto-detect clusters
        n_clusters: Fixed cluster count
        mode: 'belief' or 'prior'

    Returns:
        Tuple of (RGDiagnostics, IBDiagnostics)
    """
    # Get RG diagnostics (includes cluster detection)
    rg_diag = compute_rg_diagnostics(
        system, step, auto_cluster, n_clusters, mode
    )

    # Extract belief parameters for IB computation
    mu, sigma = extract_belief_params(system)

    if mu.size > 0 and rg_diag.cluster_labels is not None:
        ib_metrics = compute_ib_metrics(mu, sigma, rg_diag.cluster_labels)
        ib_diag = IBDiagnostics(
            step=step,
            I_XT=ib_metrics['I_XT'],
            I_TY=ib_metrics['I_TY'],
            ib_ratio=ib_metrics['ib_ratio'],
            sufficiency_gap=ib_metrics['sufficiency_gap'],
        )
    else:
        ib_diag = IBDiagnostics(
            step=step,
            I_XT=0.0,
            I_TY=0.0,
            ib_ratio=0.0,
            sufficiency_gap=0.0,
        )

    return rg_diag, ib_diag
