"""
Pytest Configuration and Fixtures
"""

import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)

@pytest.fixture
def seed():
    return 42

@pytest.fixture
def minimal_config():
    """Minimal simulation configuration for fast tests."""
    from simulation_config import SimulationConfig
    return SimulationConfig(
        experiment_name="test",
        spatial_shape=(),
        n_agents=2,
        K_latent=3,  # Must be odd for SO(3)
        n_steps=5,
        enable_emergence=False,
        enable_hamiltonian=False,
    )

@pytest.fixture
def emergence_config():
    """Configuration for emergence tests."""
    from simulation_config import SimulationConfig
    return SimulationConfig(
        experiment_name="test_emergence",
        spatial_shape=(),
        n_agents=5,
        K_latent=3,  # Must be odd
        n_steps=20,
        enable_emergence=True,
        consensus_threshold=0.5,
        min_cluster_size=2,
        max_scale=3,
    )

@pytest.fixture
def simple_system(minimal_config, rng):
    """Build a minimal multi-agent system for testing."""
    from simulation_runner import (
        build_manifold, build_supports, build_agents, build_system
    )
    manifold = build_manifold(minimal_config)
    supports = build_supports(manifold, minimal_config, rng)
    agents = build_agents(manifold, supports, minimal_config, rng)
    system = build_system(agents, minimal_config, rng)
    return system

@pytest.fixture
def simple_agents(minimal_config, rng):
    """Build minimal agents for testing."""
    from simulation_runner import (
        build_manifold, build_supports, build_agents,
    )
    manifold = build_manifold(minimal_config)
    supports = build_supports(manifold, minimal_config, rng)
    agents = build_agents(manifold, supports, minimal_config, rng)
    return agents

@pytest.fixture
def hamiltonian_trainer(simple_system):
    """Hamiltonian dynamics trainer for testing."""
    from agent.hamiltonian_trainer import HamiltonianTrainer
    from config import TrainingConfig
    config = TrainingConfig(n_steps=10)
    return HamiltonianTrainer(
        simple_system,
        config=config,
        friction=0.0,
        mass_scale=1.0,
        enable_geodesic_correction=True,
        track_phase_space=False
    )

@pytest.fixture
def random_spd_matrix(rng):
    """Factory for random symmetric positive definite matrices."""
    def _make_spd(K: int = 3):
        A = rng.standard_normal((K, K))
        return A @ A.T + 0.1 * np.eye(K)
    return _make_spd

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


# =============================================================================
# Tensor Agent Fixtures (PyTorch)
# =============================================================================

@pytest.fixture
def torch_device():
    """Device for tensor tests - CPU for CI compatibility."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def tensor_agent(torch_device):
    """Single TensorAgent for testing."""
    from agent.tensor_agent import TensorAgent
    agent = TensorAgent(K=3, spatial_shape=(), device=torch_device)
    agent.initialize(seed=42)
    return agent


@pytest.fixture
def tensor_agent_cholesky(torch_device):
    """TensorAgent with Cholesky parameterization."""
    from agent.tensor_agent import TensorAgent
    agent = TensorAgent(
        K=3, spatial_shape=(), device=torch_device,
        use_cholesky_param=True
    )
    agent.initialize(seed=42)
    return agent


@pytest.fixture
def tensor_system(torch_device):
    """TensorSystem with multiple agents."""
    from agent.tensor_system import TensorSystem
    system = TensorSystem(
        N=3, K=3, spatial_shape=(), device=torch_device,
        lambda_self=1.0, lambda_belief=0.5
    )
    system.initialize(seed=42)
    return system


@pytest.fixture
def so3_generators(torch_device):
    """SO(3) Lie algebra generators."""
    import torch
    K = 3
    generators = torch.zeros(3, K, K, device=torch_device)
    # J_1
    generators[0, 1, 2] = -1.0
    generators[0, 2, 1] = 1.0
    # J_2
    generators[1, 0, 2] = 1.0
    generators[1, 2, 0] = -1.0
    # J_3
    generators[2, 0, 1] = -1.0
    generators[2, 1, 0] = 1.0
    return generators