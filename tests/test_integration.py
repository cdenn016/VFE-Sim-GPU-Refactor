"""
Integration Tests
==================

End-to-end tests for complete simulation pipelines
and module interactions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineConstruction:
    """Tests for simulation pipeline construction."""

    def test_build_manifold(self, minimal_config):
        """Test manifold construction."""
        from simulation_runner import build_manifold
        manifold = build_manifold(minimal_config)
        assert manifold is not None

    def test_build_full_system(self, minimal_config, rng):
        """Test complete system construction."""
        from simulation_runner import build_manifold, build_supports, build_agents, build_system
        manifold = build_manifold(minimal_config)
        supports = build_supports(manifold, minimal_config, rng)
        agents = build_agents(manifold, supports, minimal_config, rng)
        system = build_system(agents, minimal_config, rng)
        assert system is not None
        assert system.n_agents == minimal_config.n_agents

    def test_pipeline_with_different_configs(self, rng):
        """Test pipeline with various configurations."""
        from simulation_config import SimulationConfig
        from simulation_runner import (
            build_manifold, build_supports, build_agents, build_system
        )

        configs = [
            SimulationConfig(
                experiment_name="test_0d",
                spatial_shape=(),
                n_agents=2,
                K_latent=3,
                n_steps=5,
            ),
            SimulationConfig(
                experiment_name="test_3agents",
                spatial_shape=(),
                n_agents=3,
                K_latent=5,
                n_steps=5,
            ),
        ]

        for config in configs:
            manifold = build_manifold(config)
            supports = build_supports(manifold, config, rng)
            agents = build_agents(manifold, supports, config, rng)
            system = build_system(agents, config, rng)
            assert system.n_agents == config.n_agents


class TestGradientTraining:
    """Tests for gradient-based training."""

    def test_trainer_runs(self, simple_system):
        """Test that gradient trainer runs."""
        from agent.trainer import Trainer
        from config import TrainingConfig
        config = TrainingConfig(n_steps=3, lr_mu_q=0.1, lr_sigma_q=0.01)
        trainer = Trainer(simple_system, config)
        history = trainer.train()
        assert len(history.steps) == 3


class TestConfigPresets:
    """Tests for configuration presets."""

    def test_default_config(self):
        """Test default configuration."""
        from simulation_config import default_config
        cfg = default_config()
        assert cfg.n_agents > 0

    def test_hamiltonian_config(self):
        """Test Hamiltonian configuration."""
        from simulation_config import hamiltonian_config
        cfg = hamiltonian_config()
        assert cfg.enable_hamiltonian is True


class TestHamiltonianTraining:
    """Tests for Hamiltonian dynamics training."""

    def test_hamiltonian_training_pipeline(self, simple_system):
        """Test complete Hamiltonian training pipeline."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system,
            config=config,
            friction=0.1,
            mass_scale=1.0,
            track_phase_space=False
        )

        history = trainer.train(n_steps=10, dt=0.01)

        assert len(history.steps) == 10
        assert len(history.total_energy) == 10
        assert all(np.isfinite(e) for e in history.total_energy)

    def test_hamiltonian_with_geodesic_correction(self, simple_system):
        """Test Hamiltonian training with geodesic correction."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system,
            config=config,
            friction=0.0,
            mass_scale=1.0,
            enable_geodesic_correction=True,
            track_phase_space=False
        )

        history = trainer.train(n_steps=10, dt=0.01)
        assert len(history.steps) == 10


class TestConsensusIntegration:
    """Tests for consensus detection integration."""

    def test_consensus_detection_on_system(self, simple_system):
        """Test consensus detection on a multi-agent system."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(
            belief_threshold=0.5,
            model_threshold=0.5
        )

        for i in range(simple_system.n_agents):
            for j in range(i + 1, simple_system.n_agents):
                state = detector.check_full_consensus(
                    simple_system.agents[i],
                    simple_system.agents[j]
                )
                assert hasattr(state, 'belief_consensus')
                assert hasattr(state, 'is_epistemically_dead')


class TestFreeEnergyMinimization:
    """Tests for free energy minimization."""

    def test_free_energy_computation(self, simple_system):
        """Test free energy computation."""
        from gradients.free_energy_clean import compute_total_free_energy

        breakdown = compute_total_free_energy(simple_system)
        assert np.isfinite(breakdown.total)

    def test_natural_gradient_computation(self, simple_system):
        """Test natural gradient computation."""
        from gradients.gradient_engine import compute_natural_gradients

        grads = compute_natural_gradients(simple_system)

        # Returns List[AgentGradients]
        assert isinstance(grads, list)
        assert len(grads) == simple_system.n_agents


class TestModuleImports:
    """Tests that all key modules can be imported."""

    def test_import_agent_modules(self):
        """Test agent module imports."""
        from agent.agents import Agent
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from agent.system import MultiAgentSystem

    def test_import_geometry_modules(self):
        """Test geometry module imports."""
        from geometry.geometry_base import BaseManifold, TopologyType
        from geometry.phase_space_tracker import PhaseSpaceTracker

    def test_import_gradient_modules(self):
        """Test gradient module imports."""
        from gradients.free_energy_clean import compute_total_free_energy
        from gradients.gradient_engine import compute_natural_gradients
        from gradients.softmax_grads import compute_softmax_weights

    def test_import_meta_modules(self):
        """Test meta module imports."""
        from meta.consensus import ConsensusDetector, ConsensusState

    def test_import_math_utils(self):
        """Test math_utils imports."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport
        from math_utils.numerical_utils import kl_gaussian, sanitize_sigma

    def test_import_config_modules(self):
        """Test config module imports."""
        from config import AgentConfig, TrainingConfig
        from simulation_config import SimulationConfig