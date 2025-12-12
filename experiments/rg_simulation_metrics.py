# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:07:43 2025

@author: chris and christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RG Metrics for Multi-Agent VFE Simulations (NumPy Version)
============================================================

NumPy-compatible implementation of Renormalization Group metrics for analyzing
emergent structure in multi-agent VFE systems.

This mirrors transformer/rg_metrics.py but works with:
- NumPy arrays (not PyTorch tensors)
- Multi-agent systems (not transformer attention)
- Softmax coupling weights β_ij from gradients/softmax_grads.py

Key Metrics
-----------
- Modularity Q(β): Block structure in coupling matrix
- Effective rank: Effective degrees of freedom
- Beta entropy: Sparsity/focus of attention
- KL within/between clusters: Meta-agent coherence

Phase Transition Detection
--------------------------
At critical temperature κ*, we expect:
- Sharp jump in modularity Q
- Rapid decrease in effective rank
- Emergence of distinct clusters

Author: Chris and Christine
Date: December 2025
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from scipy import linalg


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RGDiagnostics:
    """
    Container for RG diagnostics at each training step.

    NumPy version of transformer/rg_metrics.py RGDiagnostics.
    """
    step: int
    modularity: float
    effective_rank: float
    n_clusters: int
    cluster_labels: Optional[np.ndarray] = None
    kl_within_mean: float = 0.0
    kl_within_std: float = 0.0
    kl_between_mean: float = 0.0
    kl_between_std: float = 0.0
    beta_entropy: float = 0.0
    meta_agent_sizes: Optional[List[int]] = None

    # Phase transition indicators
    modularity_derivative: float = 0.0  # dQ/dκ
    susceptibility: float = 0.0         # χ = d²Q/dκ²

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'step': self.step,
            'modularity': self.modularity,
            'effective_rank': self.effective_rank,
            'n_clusters': self.n_clusters,
            'kl_within_mean': self.kl_within_mean,
            'kl_within_std': self.kl_within_std,
            'kl_between_mean': self.kl_between_mean,
            'kl_between_std': self.kl_between_std,
            'beta_entropy': self.beta_entropy,
            'meta_agent_sizes': self.meta_agent_sizes,
            'modularity_derivative': self.modularity_derivative,
            'susceptibility': self.susceptibility,
        }


@dataclass
class RGFlowSummary:
    """
    Summary of RG flow across training.

    Tracks evolution of observables to verify RG predictions:
    - Modularity should increase (block structure emerges)
    - Effective rank should decrease (fewer DOF)
    - KL within clusters should decrease (meta-agents tighten)
    - KL between clusters should stabilize (groups remain distinct)
    """
    n_steps: int = 0
    modularity_history: List[float] = field(default_factory=list)
    effective_rank_history: List[float] = field(default_factory=list)
    n_clusters_history: List[int] = field(default_factory=list)
    kl_within_history: List[float] = field(default_factory=list)
    kl_between_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)  # Temperature

    def add_step(self, diagnostics: RGDiagnostics, kappa: float = None):
        """Add diagnostics from a single step."""
        self.n_steps += 1
        self.modularity_history.append(diagnostics.modularity)
        self.effective_rank_history.append(diagnostics.effective_rank)
        self.n_clusters_history.append(diagnostics.n_clusters)
        self.kl_within_history.append(diagnostics.kl_within_mean)
        self.kl_between_history.append(diagnostics.kl_between_mean)
        self.entropy_history.append(diagnostics.beta_entropy)
        if kappa is not None:
            self.kappa_history.append(kappa)

    def get_rg_trends(self) -> dict:
        """
        Compute trends in RG observables.

        Returns slopes of linear fits to each observable.
        """
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            y = np.array(values)
            if np.std(y) < 1e-10:
                return 0.0
            # Linear regression slope
            slope = np.corrcoef(x, y)[0, 1] * np.std(y) / np.std(x)
            return float(slope)

        return {
            'modularity_trend': compute_trend(self.modularity_history),
            'effective_rank_trend': compute_trend(self.effective_rank_history),
            'kl_within_trend': compute_trend(self.kl_within_history),
            'kl_between_trend': compute_trend(self.kl_between_history),
            'entropy_trend': compute_trend(self.entropy_history),
        }

    def is_rg_behavior(self, threshold: float = 0.01) -> dict:
        """
        Check if observed behavior matches RG predictions.
        """
        trends = self.get_rg_trends()
        return {
            'modularity_increasing': trends['modularity_trend'] > threshold,
            'effective_rank_decreasing': trends['effective_rank_trend'] < -threshold,
            'kl_within_decreasing': trends['kl_within_trend'] < -threshold,
            'kl_between_stable': abs(trends['kl_between_trend']) < threshold * 10,
        }

    def detect_phase_transition(self) -> Optional[dict]:
        """
        Detect phase transition in κ sweep data.

        Looks for:
        - Maximum in dQ/dκ (susceptibility peak)
        - Inflection point in modularity curve

        Returns:
            dict with critical κ* and transition indicators, or None
        """
        if len(self.kappa_history) < 5:
            return None

        Q = np.array(self.modularity_history)
        kappa = np.array(self.kappa_history)

        # Sort by κ (in case not monotonic)
        sort_idx = np.argsort(kappa)
        kappa = kappa[sort_idx]
        Q = Q[sort_idx]

        # Compute numerical derivatives
        dQ_dk = np.gradient(Q, kappa)
        d2Q_dk2 = np.gradient(dQ_dk, kappa)

        # Find peak in |dQ/dκ| (susceptibility)
        susceptibility = np.abs(dQ_dk)
        peak_idx = np.argmax(susceptibility)

        # Critical temperature at susceptibility peak
        kappa_critical = kappa[peak_idx]

        # Transition sharpness (max susceptibility value)
        transition_sharpness = susceptibility[peak_idx]

        return {
            'kappa_critical': float(kappa_critical),
            'transition_sharpness': float(transition_sharpness),
            'peak_susceptibility': float(susceptibility[peak_idx]),
            'modularity_at_critical': float(Q[peak_idx]),
            'is_sharp_transition': transition_sharpness > 0.5,  # Threshold for "sharp"
        }


# =============================================================================
# Extract Coupling Matrix from Multi-Agent System
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
        except Exception as e:
            # Skip agents with no neighbors
            continue

    # Ensure proper normalization (rows should sum to 1 for attended agents)
    row_sums = beta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
    # Don't renormalize - keep as computed (may not sum to 1 if isolated agents)

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
# Core RG Metrics (NumPy Implementation)
# =============================================================================

def compute_modularity(
    beta: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    resolution: float = 1.0,
) -> float:
    """
    Compute Newman-Girvan modularity Q(β).

    Q = (1/2m) Σ_{ij} [β_ij - γ·k_i·k_j/(2m)] δ(c_i, c_j)

    High modularity indicates emergent block structure (meta-agents).

    Args:
        beta: (N, N) coupling matrix
        cluster_labels: (N,) cluster assignments (auto-detect if None)
        resolution: Resolution parameter γ

    Returns:
        modularity: Q value in [-0.5, 1]
    """
    N = beta.shape[0]

    if N == 0:
        return 0.0

    # Auto-detect clusters if not provided
    if cluster_labels is None:
        cluster_labels = detect_clusters_spectral(beta)

    # Symmetrize for undirected modularity
    beta_sym = 0.5 * (beta + beta.T)

    # Degree: k_i = Σ_j β_ij
    k = beta_sym.sum(axis=1)

    # Total edge weight
    m = beta_sym.sum() / 2 + 1e-10

    # Null model: k_i * k_j / (2m)
    null_model = np.outer(k, k) / (2 * m)

    # Same-cluster indicator
    same_cluster = (cluster_labels[:, None] == cluster_labels[None, :]).astype(float)

    # Modularity matrix
    B_matrix = beta_sym - resolution * null_model

    # Q = sum over same-cluster pairs
    Q = (B_matrix * same_cluster).sum() / (2 * m)

    return float(Q)


def compute_effective_rank(
    beta: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute effective rank via spectral entropy.

    eff_rank = exp(H) where H = -Σ p_i log(p_i)
    and p_i = σ_i / Σ σ_j (normalized singular values)

    Lower effective rank = more concentrated structure.

    Args:
        beta: (N, N) coupling matrix
        eps: Numerical stability

    Returns:
        effective_rank: Value in [1, N]
    """
    N = beta.shape[0]

    if N == 0:
        return 0.0

    try:
        # SVD
        U, S, Vh = linalg.svd(beta, full_matrices=False)
    except linalg.LinAlgError:
        return float(N)

    # Normalize singular values
    S_norm = S / (S.sum() + eps)
    S_norm = np.clip(S_norm, eps, None)

    # Spectral entropy
    H = -(S_norm * np.log(S_norm)).sum()

    # Effective rank
    return float(np.exp(H))


def compute_beta_entropy(
    beta: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Compute mean entropy of coupling distributions.

    H_i = -Σ_j β_ij log(β_ij)

    Lower entropy = more focused/sparse attention.
    Higher entropy = more diffuse/uniform attention.

    Args:
        beta: (N, N) coupling matrix
        eps: Numerical stability

    Returns:
        mean_entropy: Average entropy across agents
    """
    beta_safe = np.clip(beta, eps, None)

    # Row-wise entropy
    H = -(beta_safe * np.log(beta_safe)).sum(axis=1)

    return float(H.mean())


# =============================================================================
# Cluster Detection (NumPy)
# =============================================================================

def detect_clusters_spectral(
    beta: np.ndarray,
    n_clusters: Optional[int] = None,
    min_clusters: int = 2,
    max_clusters: Optional[int] = None,
) -> np.ndarray:
    """
    Detect clusters using spectral clustering.

    Uses eigendecomposition of normalized Laplacian.

    Args:
        beta: (N, N) coupling matrix
        n_clusters: Number of clusters (auto if None)
        min_clusters: Minimum clusters
        max_clusters: Maximum clusters (default N//2)

    Returns:
        labels: (N,) cluster assignments
    """
    N = beta.shape[0]

    if N <= 1:
        return np.zeros(N, dtype=int)

    if max_clusters is None:
        max_clusters = max(min_clusters + 1, N // 2)

    # Symmetrize
    W = 0.5 * (beta + beta.T)

    # Degree matrix
    d = W.sum(axis=1)
    d_safe = np.clip(d, 1e-10, None)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))

    # Normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    try:
        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(L_norm)
    except linalg.LinAlgError:
        return np.zeros(N, dtype=int)

    # Auto-detect cluster count via eigengap
    if n_clusters is None:
        # Find largest gap in eigenvalues
        max_idx = min(max_clusters, len(eigenvalues) - 1)
        eigengaps = eigenvalues[1:max_idx] - eigenvalues[:max_idx-1]
        if len(eigengaps) > 0:
            n_clusters = np.argmax(eigengaps) + 2
            n_clusters = max(min_clusters, min(n_clusters, max_clusters))
        else:
            n_clusters = min_clusters

    # Use first k eigenvectors
    features = eigenvectors[:, :n_clusters]

    # Normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    features = features / norms

    # K-means clustering
    labels = _kmeans_numpy(features, n_clusters)

    return labels


def _kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    max_iters: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Simple k-means clustering in NumPy.
    """
    N, K = X.shape
    rng = np.random.default_rng(seed)

    # Initialize centroids via k-means++
    centroids = [X[rng.integers(N)]]

    for _ in range(n_clusters - 1):
        # Distance to nearest centroid
        dists = np.stack([
            ((X - c) ** 2).sum(axis=1)
            for c in centroids
        ], axis=1).min(axis=1)

        # Sample proportional to squared distance
        probs = dists / (dists.sum() + 1e-10)
        idx = rng.choice(N, p=probs)
        centroids.append(X[idx])

    centroids = np.stack(centroids)

    # Iterate
    for _ in range(max_iters):
        # Assign to nearest centroid
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)

        # Update centroids
        new_centroids = []
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                new_centroids.append(X[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[c])

        new_centroids = np.stack(new_centroids)

        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return labels


# =============================================================================
# KL Statistics Within/Between Clusters
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


def compute_kl_within_clusters(
    mu: np.ndarray,
    sigma: np.ndarray,
    cluster_labels: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute mean and std of KL divergence within clusters.

    Lower values indicate tighter clusters (more coherent meta-agents).
    """
    N = mu.shape[0]
    unique_clusters = np.unique(cluster_labels)

    all_kl = []

    for c in unique_clusters:
        indices = np.where(cluster_labels == c)[0]

        if len(indices) < 2:
            continue

        # Pairwise KL within cluster
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                kl_ij = kl_gaussian(mu[idx_i], sigma[idx_i], mu[idx_j], sigma[idx_j], eps)
                all_kl.append(kl_ij)

    if len(all_kl) == 0:
        return 0.0, 0.0

    all_kl = np.array(all_kl)
    return float(all_kl.mean()), float(all_kl.std())


def compute_kl_between_clusters(
    mu: np.ndarray,
    sigma: np.ndarray,
    cluster_labels: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Compute mean and std of KL divergence between cluster centroids.

    Stable values indicate clusters remain distinct.
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return 0.0, 0.0

    # Compute cluster centroids
    centroids_mu = []
    centroids_sigma = []

    for c in unique_clusters:
        mask = cluster_labels == c
        centroids_mu.append(mu[mask].mean(axis=0))
        centroids_sigma.append(sigma[mask].mean(axis=0))

    centroids_mu = np.stack(centroids_mu)
    centroids_sigma = np.stack(centroids_sigma)

    # Pairwise KL between centroids
    all_kl = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            kl_ij = kl_gaussian(
                centroids_mu[i], centroids_sigma[i],
                centroids_mu[j], centroids_sigma[j],
                eps
            )
            all_kl.append(kl_ij)

    if len(all_kl) == 0:
        return 0.0, 0.0

    all_kl = np.array(all_kl)
    return float(all_kl.mean()), float(all_kl.std())


# =============================================================================
# Full RG Diagnostics
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

    # Detect clusters
    if auto_cluster:
        cluster_labels = detect_clusters_spectral(beta, n_clusters=n_clusters)
    else:
        # Use KL-based detection
        cluster_labels = detect_clusters_spectral(beta, n_clusters=n_clusters)

    # Compute metrics
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
    unique_clusters = np.unique(cluster_labels)
    n_clusters_detected = len(unique_clusters)
    meta_agent_sizes = [
        int((cluster_labels == c).sum())
        for c in unique_clusters
    ]

    return RGDiagnostics(
        step=step,
        modularity=modularity,
        effective_rank=effective_rank,
        n_clusters=n_clusters_detected,
        cluster_labels=cluster_labels,
        kl_within_mean=kl_within_mean,
        kl_within_std=kl_within_std,
        kl_between_mean=kl_between_mean,
        kl_between_std=kl_between_std,
        beta_entropy=beta_entropy,
        meta_agent_sizes=meta_agent_sizes,
    )


def analyze_rg_flow(
    history: List[RGDiagnostics],
    kappa_values: Optional[List[float]] = None,
) -> RGFlowSummary:
    """
    Analyze RG flow from list of diagnostics.

    Args:
        history: List of RGDiagnostics from each step
        kappa_values: Optional list of κ values (for phase transition detection)

    Returns:
        RGFlowSummary with evolution data
    """
    summary = RGFlowSummary()

    for i, diag in enumerate(history):
        kappa = kappa_values[i] if kappa_values else None
        summary.add_step(diag, kappa=kappa)

    return summary


# =============================================================================
# Information Bottleneck Metrics
# =============================================================================

def estimate_mutual_information_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Estimate mutual information I(X;T) for Gaussian representations.

    For Gaussians, the entropy H(T) = 0.5 * log((2πe)^K * det(Σ)).
    We use the variance of means as a proxy for I(X;T).

    Args:
        mu: (N, K) belief means (representations T)
        sigma: (N, K) or (N, K, K) covariances

    Returns:
        Estimated mutual information (bits)
    """
    N, K = mu.shape

    # Variance of means across agents (spread in representation space)
    mu_var = np.var(mu, axis=0).sum()

    # Mean variance (average uncertainty)
    if sigma.ndim == 2:
        sigma_mean = sigma.mean(axis=0).sum()
    else:
        sigma_mean = np.trace(sigma.mean(axis=0))

    # I(X;T) ≈ 0.5 * log(1 + var(μ)/mean(σ))
    # This is a simplified estimate assuming Gaussian structure
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

    Here we use cluster assignments as proxy for Y (what's being predicted).
    I(T;Y) is estimated via the discriminability of cluster centroids.

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

    # Between-cluster variance (how spread out are clusters?)
    global_mean = mu.mean(axis=0)
    between_var = np.array([
        (cluster_labels == c).sum() * np.sum((centroids[i] - global_mean) ** 2)
        for i, c in enumerate(unique_clusters)
    ]).sum() / N

    # Within-cluster variance (how tight are clusters?)
    within_var = np.array([
        np.sum((mu[cluster_labels == c] - centroids[i]) ** 2)
        for i, c in enumerate(unique_clusters)
    ]).sum() / N

    # I(T;Y) ≈ 0.5 * log(1 + between/within)
    # Higher between/within ratio → more discriminable → higher I(T;Y)
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

    The IB principle: min I(X;T) - β·I(T;Y)
    Equivalence: κ ↔ 1/β_IB

    Args:
        mu: (N, K) belief means
        sigma: (N, K) or (N, K, K) covariances
        cluster_labels: (N,) cluster assignments

    Returns:
        Dictionary with IB metrics:
        - I_XT: Compression (should decrease with lower κ)
        - I_TY: Relevance (should increase with lower κ)
        - ib_ratio: I_TY/I_XT (efficiency of representation)
        - sufficiency_gap: I_XT - I_TY (excess compression)
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
    """
    Information Bottleneck diagnostics at each step.

    Tracks the information plane trajectory:
    - I(X;T): How much information is retained (compression axis)
    - I(T;Y): How much relevant information is kept (relevance axis)

    IB Theory Prediction:
    - High κ (low β_IB): I(X;T) low, I(T;Y) low (trivial representations)
    - Low κ (high β_IB): I(X;T) high, I(T;Y) high (complex representations)
    - Optimal κ*: Best tradeoff on information plane frontier
    """
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


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("RG SIMULATION METRICS TEST (NumPy)")
    print("=" * 70)

    np.random.seed(42)

    N = 20  # 20 agents
    K = 8   # Latent dimension

    # Create synthetic block-structured coupling matrix (4 clusters of 5)
    beta = np.zeros((N, N))
    for i in range(4):
        start = i * 5
        end = (i + 1) * 5
        beta[start:end, start:end] = np.random.rand(5, 5) * 0.8 + 0.2

    # Add some cross-cluster coupling
    beta = beta + np.random.rand(N, N) * 0.1

    # Normalize rows
    beta = beta / beta.sum(axis=1, keepdims=True)

    print(f"\n[1] Testing modularity...")
    Q = compute_modularity(beta)
    print(f"    Modularity Q = {Q:.4f}")

    print(f"\n[2] Testing effective rank...")
    eff_rank = compute_effective_rank(beta)
    print(f"    Effective rank = {eff_rank:.2f}")

    print(f"\n[3] Testing cluster detection...")
    clusters = detect_clusters_spectral(beta)
    print(f"    Detected clusters: {np.unique(clusters).tolist()}")
    print(f"    Cluster sizes: {[int((clusters == c).sum()) for c in np.unique(clusters)]}")

    print(f"\n[4] Testing KL within/between...")
    mu = np.random.randn(N, K)
    sigma = np.abs(np.random.randn(N, K)) * 0.5 + 0.1

    kl_within, kl_within_std = compute_kl_within_clusters(mu, sigma, clusters)
    kl_between, kl_between_std = compute_kl_between_clusters(mu, sigma, clusters)
    print(f"    KL within: {kl_within:.4f} ± {kl_within_std:.4f}")
    print(f"    KL between: {kl_between:.4f} ± {kl_between_std:.4f}")

    print(f"\n[5] Testing beta entropy...")
    entropy = compute_beta_entropy(beta)
    print(f"    Beta entropy = {entropy:.4f}")

    print(f"\n[6] Testing Information Bottleneck metrics...")
    I_XT = estimate_mutual_information_gaussian(mu, sigma)
    I_TY = estimate_relevance_information(mu, clusters)
    ib_metrics = compute_ib_metrics(mu, sigma, clusters)
    print(f"    I(X;T) = {I_XT:.4f} bits (compression)")
    print(f"    I(T;Y) = {I_TY:.4f} bits (relevance)")
    print(f"    IB ratio = {ib_metrics['ib_ratio']:.4f}")
    print(f"    Sufficiency gap = {ib_metrics['sufficiency_gap']:.4f}")

    print("\n" + "=" * 70)
    print("All NumPy RG + IB metrics tests passed!")
    print("=" * 70)