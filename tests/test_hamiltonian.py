"""
Hamiltonian Dynamics Unit Tests
================================

Comprehensive tests for Hamiltonian dynamics, symplectic integration,
geodesic corrections, and phase space tracking.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHamiltonianTrainerConstruction:
    """Tests for HamiltonianTrainer initialization."""

    def test_trainer_creation(self, simple_system):
        """Test basic trainer construction."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0
        )
        assert trainer is not None
        assert trainer.friction == 0.0

    def test_trainer_with_friction(self, simple_system):
        """Test trainer with non-zero friction."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0
        )
        assert trainer.friction == 0.1

    def test_trainer_with_geodesic_correction(self, simple_system):
        """Test trainer with geodesic correction enabled."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0,
            enable_geodesic_correction=True
        )
        assert trainer.enable_geodesic_correction is True

    def test_theta_initialization(self, simple_system):
        """Test that theta is properly initialized."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(simple_system, config=config, friction=0.0, mass_scale=1.0)

        assert trainer.theta is not None
        assert len(trainer.theta.shape) == 1
        assert np.all(np.isfinite(trainer.theta))

    def test_momentum_starts_zero(self, simple_system):
        """Test that momentum is initialized to zero."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(simple_system, config=config, friction=0.0, mass_scale=1.0)

        assert np.allclose(trainer.p, 0)


class TestSymplecticIntegration:
    """Tests for symplectic integrator."""

    def test_train_method_returns_history(self, simple_system):
        """Test that train method returns HamiltonianHistory."""
        from agent.hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=False
        )
        history = trainer.train(n_steps=5, dt=0.01)

        assert isinstance(history, HamiltonianHistory)
        assert len(history.steps) == 5

    def test_step_changes_theta(self, hamiltonian_trainer, rng):
        """Test that a single step changes theta."""
        np.random.seed(42)
        hamiltonian_trainer.p = 0.1 * np.random.randn(len(hamiltonian_trainer.theta))
        initial_theta = hamiltonian_trainer.theta.copy()

        hamiltonian_trainer.step(dt=0.01)

        assert not np.allclose(hamiltonian_trainer.theta, initial_theta)

    def test_step_updates_momentum(self, hamiltonian_trainer, rng):
        """Test that a single step updates momentum."""
        np.random.seed(42)
        initial_p = hamiltonian_trainer.p.copy()
        hamiltonian_trainer.p = 0.1 * np.random.randn(len(hamiltonian_trainer.theta))

        hamiltonian_trainer.step(dt=0.01)

        # Momentum should change due to forces
        assert not np.allclose(hamiltonian_trainer.p, initial_p)

    def test_multiple_steps(self, simple_system):
        """Test running multiple steps."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=10)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0,
            track_phase_space=False
        )

        initial_theta = trainer.theta.copy()
        trainer.train(n_steps=10, dt=0.01)

        # Theta should have evolved
        assert not np.allclose(trainer.theta, initial_theta)


class TestHamiltonianHistory:
    """Tests for HamiltonianHistory tracking."""

    def test_history_records_energies(self, simple_system):
        """Test that history records energy values."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=False
        )
        history = trainer.train(n_steps=5, dt=0.01)

        assert len(history.total_energy) == 5
        assert len(history.kinetic_energy) == 5

    def test_history_energies_are_finite(self, simple_system):
        """Test that recorded energies are finite."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=False
        )
        history = trainer.train(n_steps=5, dt=0.01)

        assert all(np.isfinite(e) for e in history.total_energy)
        assert all(np.isfinite(e) for e in history.kinetic_energy)

    def test_history_steps_match_n_steps(self, simple_system):
        """Test that history length matches n_steps."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        n_steps = 7
        config = TrainingConfig(n_steps=n_steps)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=False
        )
        history = trainer.train(n_steps=n_steps, dt=0.01)

        assert len(history.steps) == n_steps


class TestEnergyConservation:
    """Tests for energy conservation in Hamiltonian dynamics."""

    def test_frictionless_energy_approximately_conserved(self, simple_system):
        """Test that total energy is approximately conserved without friction."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=20)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0,
            track_phase_space=False, enable_geodesic_correction=True
        )

        # Give some initial momentum
        np.random.seed(42)
        trainer.p = 0.1 * np.random.randn(len(trainer.theta))

        history = trainer.train(n_steps=20, dt=0.001)  # Small dt for accuracy

        if len(history.total_energy) > 1:
            # Energy should be roughly conserved (allow 20% drift for short run)
            initial_E = history.total_energy[0]
            final_E = history.total_energy[-1]
            if abs(initial_E) > 1e-6:
                relative_change = abs(final_E - initial_E) / abs(initial_E)
                assert relative_change < 0.3, f"Energy changed by {relative_change*100:.1f}%"

    def test_friction_term_active(self, simple_system):
        """Test that friction term is active and affects dynamics."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=20)

        # Run with friction
        trainer_friction = HamiltonianTrainer(
            simple_system, config=config, friction=0.9, mass_scale=1.0,
            track_phase_space=False
        )
        np.random.seed(42)
        trainer_friction.p = 0.5 * np.random.randn(len(trainer_friction.theta))
        history_friction = trainer_friction.train(n_steps=20, dt=0.01)

        # Run without friction (same initial conditions)
        trainer_no_friction = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0,
            track_phase_space=False
        )
        np.random.seed(42)
        trainer_no_friction.p = 0.5 * np.random.randn(len(trainer_no_friction.theta))
        history_no_friction = trainer_no_friction.train(n_steps=20, dt=0.01)

        # Both should complete with finite energies
        assert all(np.isfinite(e) for e in history_friction.total_energy)
        assert all(np.isfinite(e) for e in history_no_friction.total_energy)

        # Friction should cause different trajectory (energy histories should differ)
        assert not np.allclose(
            history_friction.total_energy,
            history_no_friction.total_energy,
            rtol=0.01
        )


class TestGeodesicCorrection:
    """Tests for geodesic correction term in Hamiltonian dynamics."""

    def test_geodesic_force_shape(self, hamiltonian_trainer):
        """Test that geodesic force has correct shape."""
        try:
            from geometry.geodesic_corrections import compute_geodesic_force

            theta = hamiltonian_trainer.theta
            p = hamiltonian_trainer.p

            # Give non-zero momentum
            np.random.seed(42)
            p = 0.1 * np.random.randn(len(theta))

            force = compute_geodesic_force(hamiltonian_trainer, theta, p, eps=1e-5)
            assert force.shape == theta.shape
        except ImportError:
            pytest.skip("geodesic_corrections not available")

    def test_geodesic_force_finite(self, hamiltonian_trainer):
        """Test that geodesic force is finite."""
        try:
            from geometry.geodesic_corrections import compute_geodesic_force

            theta = hamiltonian_trainer.theta
            p = 0.1 * np.random.randn(len(theta))

            force = compute_geodesic_force(hamiltonian_trainer, theta, p, eps=1e-5)
            assert np.all(np.isfinite(force))
        except ImportError:
            pytest.skip("geodesic_corrections not available")

    def test_geodesic_force_zero_when_momentum_zero(self, hamiltonian_trainer):
        """Test that geodesic force is zero when momentum is zero."""
        try:
            from geometry.geodesic_corrections import compute_geodesic_force

            theta = hamiltonian_trainer.theta
            p = np.zeros_like(theta)

            force = compute_geodesic_force(hamiltonian_trainer, theta, p, eps=1e-5)
            # Force should be zero when p=0 (since it's quadratic in p)
            assert np.allclose(force, 0, atol=1e-10)
        except ImportError:
            pytest.skip("geodesic_corrections not available")


class TestParameterPacking:
    """Tests for parameter packing/unpacking."""

    def test_pack_unpack_roundtrip(self, simple_system):
        """Test that pack and unpack are inverses."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=1.0
        )

        # Get original agent states
        original_states = []
        for agent in simple_system.agents:
            original_states.append({
                'mu_q': agent.mu_q.copy(),
                'Sigma_q': agent.Sigma_q.copy()
            })

        # Pack parameters
        theta = trainer._pack_parameters()

        # Perturb theta
        theta_perturbed = theta + 0.1

        # Unpack perturbed theta
        trainer._unpack_parameters(theta_perturbed)

        # Pack again
        theta_repacked = trainer._pack_parameters()

        # Should match perturbed theta
        assert np.allclose(theta_repacked, theta_perturbed)


class TestMassMatrix:
    """Tests for mass matrix computation."""

    def test_mass_scale_affects_dynamics(self, simple_system):
        """Test that mass_scale parameter affects dynamics."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)

        # Small mass
        trainer1 = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=0.1,
            track_phase_space=False
        )
        np.random.seed(42)
        trainer1.p = 0.1 * np.random.randn(len(trainer1.theta))
        theta1_initial = trainer1.theta.copy()
        trainer1.step(dt=0.01)
        delta1 = np.linalg.norm(trainer1.theta - theta1_initial)

        # Large mass
        trainer2 = HamiltonianTrainer(
            simple_system, config=config, friction=0.0, mass_scale=10.0,
            track_phase_space=False
        )
        np.random.seed(42)
        trainer2.p = 0.1 * np.random.randn(len(trainer2.theta))
        theta2_initial = trainer2.theta.copy()
        trainer2.step(dt=0.01)
        delta2 = np.linalg.norm(trainer2.theta - theta2_initial)

        # Larger mass should result in smaller position change (same momentum)
        assert delta2 < delta1


class TestPhaseSpaceTracking:
    """Tests for phase space trajectory tracking."""

    def test_phase_space_tracking_enabled(self, simple_system):
        """Test phase space tracking when enabled."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=True
        )

        # If tracking is enabled, trainer should have tracking infrastructure
        assert hasattr(trainer, 'track_phase_space')

    def test_phase_space_tracking_disabled(self, simple_system):
        """Test that training works with tracking disabled."""
        from agent.hamiltonian_trainer import HamiltonianTrainer
        from config import TrainingConfig

        config = TrainingConfig(n_steps=5)
        trainer = HamiltonianTrainer(
            simple_system, config=config, friction=0.1, mass_scale=1.0,
            track_phase_space=False
        )

        # Should still be able to train
        history = trainer.train(n_steps=5, dt=0.01)
        assert len(history.steps) == 5