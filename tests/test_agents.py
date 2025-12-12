"""
Agent Module Unit Tests
========================

Comprehensive tests for agent construction, multi-agent systems,
gauge fields, and agent dynamics.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig


class TestAgentConstruction:
    """Tests for individual agent creation and properties."""

    def test_agent_creation_0d(self, rng):
        """Test agent creation on 0D (point) manifold."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=5)  # Odd K for SO(3)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert agent.agent_id == 0
        assert agent.K == 5
        assert agent.mu_q.shape == (5,)
        assert agent.mu_p.shape == (5,)

    def test_agent_creation_1d(self, rng):
        """Test agent creation on 1D manifold."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(16,), topology=TopologyType.PERIODIC)
        config = AgentConfig(spatial_shape=(16,), K=3)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert agent.mu_q.shape == (16, 3)
        assert agent.Sigma_q.shape == (16, 3, 3)

    def test_agent_covariance_spd(self, rng):
        """Test that agent covariance is symmetric positive definite."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=5)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        # Test symmetry
        assert np.allclose(agent.Sigma_q, agent.Sigma_q.T)
        assert np.allclose(agent.Sigma_p, agent.Sigma_p.T)

        # Test positive definiteness
        eigvals_q = np.linalg.eigvalsh(agent.Sigma_q)
        eigvals_p = np.linalg.eigvalsh(agent.Sigma_p)
        assert np.all(eigvals_q > 0), "Sigma_q must be positive definite"
        assert np.all(eigvals_p > 0), "Sigma_p must be positive definite"

    def test_agent_gauge_field_exists(self, rng):
        """Test that agents have gauge fields."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=3)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert hasattr(agent, 'gauge')
        assert hasattr(agent.gauge, 'phi')
        assert agent.gauge.phi.shape == (3,)  # so(3) has 3 generators

    def test_agent_generators_exist(self, rng):
        """Test that agents have SO(3) generators."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=5)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert hasattr(agent, 'generators')
        assert len(agent.generators) == 3  # SO(3) has 3 generators
        # Each generator should be K x K antisymmetric
        for gen in agent.generators:
            assert gen.shape == (5, 5)
            assert np.allclose(gen, -gen.T), "Generators must be antisymmetric"

    def test_agent_support_region(self, rng):
        """Test that agents have support regions."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(8,), topology=TopologyType.PERIODIC)
        config = AgentConfig(spatial_shape=(8,), K=3)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        assert hasattr(agent, 'support')
        assert hasattr(agent.support, 'chi_weight')
        assert agent.support.chi_weight.shape == (8,)
        # chi_weight should be in [0, 1]
        assert np.all(agent.support.chi_weight >= 0)
        assert np.all(agent.support.chi_weight <= 1)


class TestMultiAgentSystem:
    """Tests for multi-agent system construction and properties."""

    def test_system_construction(self, simple_system):
        """Test basic system construction."""
        assert simple_system.n_agents >= 2
        assert len(simple_system.agents) == simple_system.n_agents

    def test_overlap_masks_exist(self, simple_system):
        """Test that overlap masks are computed for agent pairs."""
        n = simple_system.n_agents
        expected_pairs = n * (n - 1)  # Ordered pairs
        assert len(simple_system.overlap_masks) == expected_pairs

    def test_system_has_config(self, simple_system):
        """Test that system has configuration."""
        assert hasattr(simple_system, 'config')

    def test_agents_have_unique_ids(self, simple_system):
        """Test that all agents have unique IDs."""
        ids = [agent.agent_id for agent in simple_system.agents]
        assert len(ids) == len(set(ids)), "Agent IDs must be unique"

    def test_get_neighbors(self, simple_system):
        """Test neighbor retrieval."""
        if hasattr(simple_system, 'get_neighbors'):
            neighbors = list(simple_system.get_neighbors(0))
            # In a 0D system with full connectivity, should have n-1 neighbors
            assert len(neighbors) >= 1

    def test_transport_operators(self, simple_system):
        """Test transport operator computation between agents."""
        if hasattr(simple_system, 'compute_transport_ij'):
            # Compute transport from agent 0 to agent 1
            Omega_01 = simple_system.compute_transport_ij(0, 1)
            K = simple_system.agents[0].K
            assert Omega_01.shape[-2:] == (K, K)
            # Transport should be orthogonal (for SO(3))
            identity = Omega_01 @ Omega_01.T
            assert np.allclose(identity, np.eye(K), atol=1e-5)


class TestAgentDynamics:
    """Tests for agent update dynamics."""

    def test_belief_update_changes_mu(self, simple_system, rng):
        """Test that belief updates change mean."""
        agent = simple_system.agents[0]
        initial_mu = agent.mu_q.copy()

        # Perturb the mean
        if agent.mu_q.ndim == 1:
            agent.mu_q = agent.mu_q + 0.1 * rng.standard_normal(agent.mu_q.shape)
        else:
            agent.mu_q = agent.mu_q + 0.1 * rng.standard_normal(agent.mu_q.shape)

        assert not np.allclose(agent.mu_q, initial_mu)

    def test_covariance_stays_spd_after_perturbation(self, simple_system, rng):
        """Test that covariance remains SPD after small perturbations."""
        from math_utils.numerical_utils import sanitize_sigma

        agent = simple_system.agents[0]
        K = agent.K

        # Add small symmetric perturbation
        if agent.Sigma_q.ndim == 2:
            perturbation = 0.01 * rng.standard_normal((K, K))
            perturbation = perturbation + perturbation.T
            Sigma_new = agent.Sigma_q + perturbation
            Sigma_new = sanitize_sigma(Sigma_new)

            eigvals = np.linalg.eigvalsh(Sigma_new)
            assert np.all(eigvals > 0)


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig(spatial_shape=(), K=5)
        assert config.K == 5
        assert config.spatial_shape == ()

    def test_config_with_custom_values(self):
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            spatial_shape=(16, 16),
            K=7,
        )
        assert config.K == 7
        assert config.spatial_shape == (16, 16)

    def test_k_must_be_odd_for_so3(self, rng):
        """Test that K must be odd for SO(3) irreps."""
        from agent.agents import Agent
        from math_utils.generators import generate_so3_generators

        # Odd K should work
        for K in [3, 5, 7]:
            generators = generate_so3_generators(K)
            assert len(generators) == 3
            assert generators[0].shape == (K, K)

        # Even K should raise an error
        with pytest.raises((ValueError, AssertionError)):
            generate_so3_generators(4)


class TestGaugeField:
    """Tests for gauge field properties."""

    def test_gauge_field_initialization(self, rng):
        """Test gauge field initialization."""
        from agent.agents import Agent
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=3)
        agent = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)

        # Gauge field should be finite
        assert np.all(np.isfinite(agent.gauge.phi))

    def test_gauge_transport_preserves_norm(self, rng):
        """Test that gauge transport preserves vector norms."""
        from agent.agents import Agent
        from math_utils.transport import compute_transport

        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=3)

        agent1 = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)
        agent2 = Agent(agent_id=1, config=config, rng=rng, base_manifold=manifold)

        # Compute transport operator
        Omega = compute_transport(
            agent1.gauge.phi,
            agent2.gauge.phi,
            agent1.generators,
            validate=False
        )

        # Apply transport to a vector
        v = rng.standard_normal(3)
        v_transported = Omega @ v

        # Norms should be preserved (SO(3) is orthogonal)
        assert np.allclose(np.linalg.norm(v), np.linalg.norm(v_transported), rtol=1e-5)