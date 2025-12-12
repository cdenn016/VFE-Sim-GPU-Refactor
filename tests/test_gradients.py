"""Gradient Computation Unit Tests"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestFreeEnergy:
    def test_free_energy_finite(self, simple_system):
        from gradients.free_energy_clean import compute_total_free_energy
        energies = compute_total_free_energy(simple_system)
        assert np.isfinite(energies.total)
        assert np.isfinite(energies.self_energy)

    def test_free_energy_nonnegative_self(self, simple_system):
        from gradients.free_energy_clean import compute_total_free_energy
        energies = compute_total_free_energy(simple_system)
        assert energies.self_energy >= 0

class TestNaturalGradients:
    def test_gradients_shape(self, simple_system):
        from gradients.gradient_engine import compute_natural_gradients
        grads = compute_natural_gradients(simple_system)
        assert len(grads) == simple_system.n_agents
        for agent, grad in zip(simple_system.agents, grads):
            assert grad.grad_mu_q.shape == agent.mu_q.shape

    def test_gradients_finite(self, simple_system):
        from gradients.gradient_engine import compute_natural_gradients
        grads = compute_natural_gradients(simple_system)
        for i, grad in enumerate(grads):
            assert np.all(np.isfinite(grad.grad_mu_q))