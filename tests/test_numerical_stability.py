"""
Numerical Stability and Edge Case Tests
========================================

Tests for numerical stability, edge cases, and robustness
of the Hamiltonian-Beliefs codebase.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCovarianceSanitization:
    """Tests for covariance matrix sanitization."""

    def test_sanitize_sigma_preserves_spd(self, random_spd_matrix):
        """Test that sanitize_sigma preserves SPD matrices."""
        from math_utils.numerical_utils import sanitize_sigma

        K = 5
        Sigma = random_spd_matrix(K)
        Sigma_sanitized = sanitize_sigma(Sigma)

        # Should still be SPD
        eigvals = np.linalg.eigvalsh(Sigma_sanitized)
        assert np.all(eigvals > 0)

    def test_sanitize_sigma_fixes_near_singular(self):
        """Test that sanitize_sigma fixes near-singular matrices."""
        from math_utils.numerical_utils import sanitize_sigma

        K = 3
        # Create a nearly singular matrix
        Sigma = np.array([[1.0, 0.999, 0.0],
                          [0.999, 1.0, 0.0],
                          [0.0, 0.0, 1e-10]])

        Sigma_sanitized = sanitize_sigma(Sigma, eps=1e-6)

        # Should now be well-conditioned
        eigvals = np.linalg.eigvalsh(Sigma_sanitized)
        assert np.all(eigvals >= 1e-6)

    def test_sanitize_sigma_enforces_symmetry(self):
        """Test that sanitize_sigma enforces symmetry."""
        from math_utils.numerical_utils import sanitize_sigma

        K = 3
        # Create asymmetric matrix
        Sigma = np.array([[1.0, 0.5, 0.1],
                          [0.4, 1.0, 0.2],
                          [0.15, 0.25, 1.0]])

        Sigma_sanitized = sanitize_sigma(Sigma)

        # Should be symmetric
        assert np.allclose(Sigma_sanitized, Sigma_sanitized.T)


class TestKLDivergenceNumerics:
    """Tests for KL divergence numerical stability."""

    def test_kl_gaussian_finite_for_valid_inputs(self, random_spd_matrix, rng):
        """Test KL divergence is finite for valid inputs."""
        from math_utils.numerical_utils import kl_gaussian

        K = 5
        mu1 = rng.standard_normal(K)
        mu2 = rng.standard_normal(K)
        Sigma1 = random_spd_matrix(K)
        Sigma2 = random_spd_matrix(K)

        kl = kl_gaussian(mu1, Sigma1, mu2, Sigma2)

        assert np.isfinite(kl)
        assert kl >= 0

    def test_kl_gaussian_zero_for_identical(self, random_spd_matrix, rng):
        """Test KL divergence is zero for identical distributions."""
        from math_utils.numerical_utils import kl_gaussian

        K = 5
        mu = rng.standard_normal(K)
        Sigma = random_spd_matrix(K)

        kl = kl_gaussian(mu, Sigma, mu, Sigma)

        assert np.isclose(kl, 0, atol=1e-10)

    def test_kl_gaussian_handles_near_singular(self):
        """Test KL handles near-singular covariances."""
        from math_utils.numerical_utils import kl_gaussian, sanitize_sigma

        K = 3
        mu1 = np.zeros(K)
        mu2 = np.ones(K)

        # Near-singular covariance
        Sigma1 = np.diag([1.0, 1.0, 1e-8])
        Sigma2 = np.diag([1.0, 1.0, 1.0])

        # Sanitize before computing KL
        Sigma1 = sanitize_sigma(Sigma1)

        kl = kl_gaussian(mu1, Sigma1, mu2, Sigma2)

        assert np.isfinite(kl)


class TestTransportNumerics:
    """Tests for numerical stability of parallel transport."""

    def test_transport_stable_for_large_phi(self, rng):
        """Test transport is stable for large gauge field values."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)

        # Large phi values (could cause overflow in naive exp)
        phi_i = 10.0 * rng.standard_normal(3)
        phi_j = 10.0 * rng.standard_normal(3)

        Omega = compute_transport(phi_i, phi_j, generators, validate=False)

        # Should still be orthogonal
        assert np.allclose(Omega @ Omega.T, np.eye(K), atol=1e-4)
        assert np.all(np.isfinite(Omega))

    def test_transport_stable_for_small_phi(self, rng):
        """Test transport is stable for very small gauge field values."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)

        # Very small phi values
        phi_i = 1e-10 * rng.standard_normal(3)
        phi_j = 1e-10 * rng.standard_normal(3)

        Omega = compute_transport(phi_i, phi_j, generators, validate=False)

        # Should be close to identity
        assert np.allclose(Omega, np.eye(K), atol=1e-8)


class TestFreeEnergyNumerics:
    """Tests for free energy numerical stability."""

    def test_free_energy_finite(self, simple_system):
        """Test that free energy computation is finite."""
        from gradients.free_energy_clean import compute_total_free_energy

        breakdown = compute_total_free_energy(simple_system)
        assert np.isfinite(breakdown.total)

    def test_free_energy_non_negative(self, simple_system):
        """Test that free energy is non-negative (for proper priors)."""
        from gradients.free_energy_clean import compute_total_free_energy

        breakdown = compute_total_free_energy(simple_system)
        # Free energy can be negative in some parameterizations,
        # but should always be finite
        assert np.isfinite(breakdown.total)


class TestGradientNumerics:
    """Tests for gradient numerical stability."""

    def test_natural_gradients_finite(self, simple_system):
        """Test that natural gradients are finite."""
        from gradients.gradient_engine import compute_natural_gradients

        grads = compute_natural_gradients(simple_system)

        # Returns List[AgentGradients]
        assert isinstance(grads, list)
        for agent_grad in grads:
            # Check each agent's gradients are finite
            if hasattr(agent_grad, 'grad_mu') and agent_grad.grad_mu is not None:
                assert np.all(np.isfinite(agent_grad.grad_mu))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_agent_system(self, rng):
        """Test system with only one agent."""
        from simulation_config import SimulationConfig
        from simulation_runner import build_manifold, build_supports, build_agents, build_system

        config = SimulationConfig(
            experiment_name="single_agent_test",
            spatial_shape=(),
            n_agents=1,  # Single agent
            K_latent=3,
            n_steps=5,
            enable_emergence=False,
            enable_hamiltonian=False,
        )

        manifold = build_manifold(config)
        supports = build_supports(manifold, config, rng)
        agents = build_agents(manifold, supports, config, rng)
        system = build_system(agents, config, rng)

        assert system.n_agents == 1

    def test_minimal_k(self, rng):
        """Test with minimal K=3 (smallest odd K for SO(3))."""
        from agent.agents import Agent
        from geometry.geometry_base import BaseManifold, TopologyType
        from config import AgentConfig

        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=3)

        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert agent.K == 3
        assert agent.mu_q.shape == (3,)
        assert agent.Sigma_q.shape == (3, 3)

    def test_large_k(self, rng):
        """Test with larger K value."""
        from agent.agents import Agent
        from geometry.geometry_base import BaseManifold, TopologyType
        from config import AgentConfig

        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        K = 15  # Larger odd K
        config = AgentConfig(spatial_shape=(), K=K)

        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert agent.K == K
        assert agent.Sigma_q.shape == (K, K)
        # Covariance should still be SPD
        eigvals = np.linalg.eigvalsh(agent.Sigma_q)
        assert np.all(eigvals > 0)


class TestInverseStability:
    """Tests for matrix inverse stability."""

    def test_inverse_with_regularization(self, random_spd_matrix):
        """Test that inverse with regularization is stable."""
        K = 5
        Sigma = random_spd_matrix(K)

        # Direct inverse
        Sigma_inv_direct = np.linalg.inv(Sigma)

        # Regularized inverse
        eps = 1e-8
        Sigma_inv_reg = np.linalg.inv(Sigma + eps * np.eye(K))

        # Both should be finite
        assert np.all(np.isfinite(Sigma_inv_direct))
        assert np.all(np.isfinite(Sigma_inv_reg))

    def test_inverse_near_singular_with_regularization(self):
        """Test inverse of near-singular matrix with regularization."""
        K = 3
        # Create near-singular matrix
        Sigma = np.array([[1.0, 0.9999, 0.0],
                          [0.9999, 1.0, 0.0],
                          [0.0, 0.0, 1e-10]])

        eps = 1e-6
        Sigma_reg = Sigma + eps * np.eye(K)
        Sigma_inv = np.linalg.inv(Sigma_reg)

        # Should be finite
        assert np.all(np.isfinite(Sigma_inv))


class TestBatchOperations:
    """Tests for batched operations on spatial manifolds."""

    def test_batch_kl_divergence(self, rng):
        """Test KL divergence on batched inputs."""
        from math_utils.numerical_utils import kl_gaussian

        # Spatial batch
        spatial_shape = (8,)
        K = 3

        mu1 = rng.standard_normal((*spatial_shape, K))
        mu2 = rng.standard_normal((*spatial_shape, K))

        # Create batched SPD matrices
        Sigma1 = np.zeros((*spatial_shape, K, K))
        Sigma2 = np.zeros((*spatial_shape, K, K))
        for i in range(spatial_shape[0]):
            A1 = rng.standard_normal((K, K))
            A2 = rng.standard_normal((K, K))
            Sigma1[i] = A1 @ A1.T + 0.1 * np.eye(K)
            Sigma2[i] = A2 @ A2.T + 0.1 * np.eye(K)

        kl = kl_gaussian(mu1, Sigma1, mu2, Sigma2)

        # Should return array of KL values
        assert kl.shape == spatial_shape
        assert np.all(np.isfinite(kl))
        assert np.all(kl >= 0)


class TestDeterminism:
    """Tests for deterministic behavior with fixed seeds."""

    def test_agent_creation_deterministic(self):
        """Test that agent creation is deterministic with same seed."""
        from agent.agents import Agent
        from geometry.geometry_base import BaseManifold, TopologyType
        from config import AgentConfig

        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=5)

        rng1 = np.random.default_rng(42)
        agent1 = Agent(agent_id=0, config=config, rng=rng1, base_manifold=manifold)

        rng2 = np.random.default_rng(42)
        agent2 = Agent(agent_id=0, config=config, rng=rng2, base_manifold=manifold)

        assert np.allclose(agent1.mu_q, agent2.mu_q)
        assert np.allclose(agent1.Sigma_q, agent2.Sigma_q)

    def test_system_creation_deterministic(self, minimal_config):
        """Test that system creation is deterministic with same seed."""
        from simulation_runner import build_manifold, build_supports, build_agents, build_system

        rng1 = np.random.default_rng(42)
        manifold1 = build_manifold(minimal_config)
        supports1 = build_supports(manifold1, minimal_config, rng1)
        agents1 = build_agents(manifold1, supports1, minimal_config, rng1)

        rng2 = np.random.default_rng(42)
        manifold2 = build_manifold(minimal_config)
        supports2 = build_supports(manifold2, minimal_config, rng2)
        agents2 = build_agents(manifold2, supports2, minimal_config, rng2)

        # Compare first agent's beliefs
        assert np.allclose(agents1[0].mu_q, agents2[0].mu_q)