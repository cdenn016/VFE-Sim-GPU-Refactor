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