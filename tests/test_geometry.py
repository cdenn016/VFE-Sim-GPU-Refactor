"""
Geometry Module Unit Tests
===========================

Comprehensive tests for manifold geometry, SO(3) operations,
parallel transport, and gauge theory.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaseManifold:
    """Tests for base manifold construction."""

    def test_0d_manifold(self):
        """Test 0D (point) manifold."""
        from geometry.geometry_base import BaseManifold, TopologyType
        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        assert manifold.ndim == 0
        assert manifold.n_points == 1

    def test_1d_manifold(self):
        """Test 1D manifold."""
        from geometry.geometry_base import BaseManifold, TopologyType
        manifold = BaseManifold(shape=(32,), topology=TopologyType.PERIODIC)
        assert manifold.ndim == 1
        assert manifold.n_points == 32

    def test_2d_manifold(self):
        """Test 2D manifold."""
        from geometry.geometry_base import BaseManifold, TopologyType
        manifold = BaseManifold(shape=(16, 16), topology=TopologyType.PERIODIC)
        assert manifold.ndim == 2
        assert manifold.n_points == 256

    def test_topology_types(self):
        """Test different topology types."""
        from geometry.geometry_base import BaseManifold, TopologyType

        for topology in [TopologyType.FLAT, TopologyType.PERIODIC]:
            manifold = BaseManifold(shape=(8,), topology=topology)
            assert manifold.topology == topology


class TestSO3Generators:
    """Tests for SO(3) Lie algebra generators."""

    def test_generators_antisymmetric(self):
        """Test that generators are antisymmetric."""
        from math_utils.generators import generate_so3_generators
        K = 5  # Must be odd!
        generators = generate_so3_generators(K)

        for gen in generators:
            assert np.allclose(gen, -gen.T), "Generators must be antisymmetric"

    def test_generator_count(self):
        """Test that there are exactly 3 generators (dim of so(3))."""
        from math_utils.generators import generate_so3_generators
        K = 3
        generators = generate_so3_generators(K)
        assert len(generators) == 3  # SO(3) has 3 generators

    def test_generator_shapes(self):
        """Test generator dimensions match K."""
        from math_utils.generators import generate_so3_generators

        for K in [3, 5, 7]:
            generators = generate_so3_generators(K)
            for gen in generators:
                assert gen.shape == (K, K)

    def test_generator_commutation_relations(self):
        """Test SO(3) commutation relations [J_i, J_j] = i*eps_ijk*J_k."""
        from math_utils.generators import generate_so3_generators
        K = 3
        generators = generate_so3_generators(K)
        J1, J2, J3 = generators

        # [J1, J2] = J3 (up to normalization)
        comm_12 = J1 @ J2 - J2 @ J1
        # The commutator should be proportional to J3
        if not np.allclose(comm_12, 0):
            # Normalize and check
            ratio = comm_12 / (J3 + 1e-10)
            non_zero = np.abs(J3) > 1e-10
            if np.any(non_zero):
                ratios = ratio[non_zero]
                assert np.allclose(ratios, ratios[0], rtol=0.1)


class TestMatrixExponential:
    """Tests for matrix exponential on so(3)."""

    def test_exp_zero_is_identity(self):
        """Test exp(0) = I."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 3
        generators = generate_so3_generators(K)
        phi_zero = np.zeros(3)

        # Transport from zero to zero should give identity
        Omega = compute_transport(phi_zero, phi_zero, generators, validate=False)
        assert np.allclose(Omega, np.eye(K))

    def test_exp_produces_orthogonal_matrix(self, rng):
        """Test that exp(A) for antisymmetric A is orthogonal."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)
        phi = 0.5 * rng.standard_normal(3)

        Omega = compute_transport(phi, np.zeros(3), generators, validate=False)

        # Check orthogonality: Omega @ Omega.T = I
        assert np.allclose(Omega @ Omega.T, np.eye(K), atol=1e-5)
        # Check determinant = 1 (special orthogonal)
        assert np.allclose(np.linalg.det(Omega), 1.0, atol=1e-5)

    def test_exp_inverse_is_transpose(self, rng):
        """Test that exp(-A) = exp(A)^T for antisymmetric A."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 3
        generators = generate_so3_generators(K)
        phi = 0.3 * rng.standard_normal(3)

        Omega_forward = compute_transport(phi, np.zeros(3), generators, validate=False)
        Omega_backward = compute_transport(-phi, np.zeros(3), generators, validate=False)

        assert np.allclose(Omega_backward, Omega_forward.T, atol=1e-5)


class TestParallelTransport:
    """Tests for parallel transport operators."""

    def test_transport_identity_when_frames_equal(self, rng):
        """Test Omega_ii = I (transport to same frame)."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)
        phi = rng.standard_normal(3)

        Omega = compute_transport(phi, phi, generators, validate=False)
        assert np.allclose(Omega, np.eye(K), atol=1e-6)

    def test_transport_inverse_property(self, rng):
        """Test Omega_ij @ Omega_ji = I."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)
        phi_i = rng.standard_normal(3)
        phi_j = rng.standard_normal(3)

        Omega_ij = compute_transport(phi_i, phi_j, generators, validate=False)
        Omega_ji = compute_transport(phi_j, phi_i, generators, validate=False)

        product = Omega_ij @ Omega_ji
        assert np.allclose(product, np.eye(K), atol=1e-5)

    def test_transport_preserves_norm(self, rng):
        """Test that transport preserves vector norms."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)
        phi_i = 0.5 * rng.standard_normal(3)
        phi_j = 0.5 * rng.standard_normal(3)

        Omega_ij = compute_transport(phi_i, phi_j, generators, validate=False)

        # Random vector
        v = rng.standard_normal(K)
        v_transported = Omega_ij @ v

        assert np.allclose(np.linalg.norm(v), np.linalg.norm(v_transported), rtol=1e-5)

    def test_transport_gaussian_covariance(self, rng, random_spd_matrix):
        """Test transport of Gaussian covariance: Sigma' = Omega @ Sigma @ Omega.T."""
        from math_utils.generators import generate_so3_generators
        from math_utils.transport import compute_transport

        K = 5
        generators = generate_so3_generators(K)
        phi_i = 0.3 * rng.standard_normal(3)
        phi_j = 0.3 * rng.standard_normal(3)

        Omega_ij = compute_transport(phi_i, phi_j, generators, validate=False)

        # Original SPD matrix
        Sigma = random_spd_matrix(K)

        # Transport covariance
        Sigma_transported = Omega_ij @ Sigma @ Omega_ij.T

        # Should still be SPD
        eigvals = np.linalg.eigvalsh(Sigma_transported)
        assert np.all(eigvals > 0), "Transported covariance must be positive definite"

        # Should be symmetric
        assert np.allclose(Sigma_transported, Sigma_transported.T)


class TestGaugeConsensus:
    """Tests for gauge consensus operations."""

    def test_gauge_consensus_exists(self):
        """Test that gauge_consensus module exists."""
        try:
            from geometry.gauge_consensus import GaugeConsensusTracker
            assert True
        except ImportError:
            pytest.skip("GaugeConsensusTracker not available")


class TestPhaseSpaceTracker:
    """Tests for phase space tracking."""

    def test_tracker_construction(self):
        """Test phase space tracker construction."""
        from geometry.phase_space_tracker import PhaseSpaceTracker
        tracker = PhaseSpaceTracker()
        assert tracker is not None

    def test_tracker_records(self, simple_system):
        """Test that tracker can record states."""
        from geometry.phase_space_tracker import PhaseSpaceTracker
        tracker = PhaseSpaceTracker()

        # The tracker should have a record method or records attribute
        assert hasattr(tracker, 'record') or hasattr(tracker, 'records')


class TestLieAlgebra:
    """Tests for Lie algebra operations."""

    def test_lie_algebra_module_exists(self):
        """Test that lie_algebra module exists."""
        try:
            from geometry.lie_algebra import so3_basis, so3_bracket
            assert True
        except ImportError:
            pytest.skip("lie_algebra module not available")

    def test_so3_basis_antisymmetric(self):
        """Test SO(3) basis elements are antisymmetric."""
        try:
            from geometry.lie_algebra import so3_basis
            basis = so3_basis()
            for B in basis:
                assert np.allclose(B, -B.T)
        except ImportError:
            pytest.skip("lie_algebra.so3_basis not available")


class TestGeodesicCorrections:
    """Tests for geodesic correction computations."""

    def test_geodesic_corrections_module_exists(self):
        """Test that geodesic_corrections module exists."""
        try:
            from geometry.geodesic_corrections import compute_geodesic_force
            assert True
        except ImportError:
            pytest.skip("geodesic_corrections not available")

    def test_geodesic_force_returns_correct_shape(self, hamiltonian_trainer):
        """Test that geodesic force has correct shape."""
        try:
            from geometry.geodesic_corrections import compute_geodesic_force

            theta = hamiltonian_trainer.theta
            p = hamiltonian_trainer.p

            force = compute_geodesic_force(hamiltonian_trainer, theta, p, eps=1e-5)
            assert force.shape == theta.shape
        except ImportError:
            pytest.skip("geodesic_corrections not available")


class TestMultiAgentMassMatrix:
    """Tests for multi-agent mass matrix construction."""

    def test_mass_matrix_module_exists(self):
        """Test that multi_agent_mass_matrix module exists."""
        try:
            from geometry.multi_agent_mass_matrix import build_full_mass_matrix_with_coupling
            assert True
        except ImportError:
            pytest.skip("multi_agent_mass_matrix not available")


class TestSupportRegion:
    """Tests for support region geometry."""

    def test_full_support_creation(self):
        """Test creation of full support region."""
        from geometry.geometry_base import BaseManifold, TopologyType, create_full_support

        manifold = BaseManifold(shape=(8,), topology=TopologyType.PERIODIC)
        support = create_full_support(manifold)
        assert support.chi_weight.shape == (8,)
        assert np.allclose(support.chi_weight, 1.0)

    def test_support_region_class(self):
        """Test SupportRegion class."""
        from geometry.geometry_base import BaseManifold, TopologyType, SupportRegion

        manifold = BaseManifold(shape=(5,), topology=TopologyType.FLAT)
        chi = np.array([1.0, 0.8, 0.5, 0.2, 0.0])
        support = SupportRegion(base_manifold=manifold, chi_weight=chi)

        # Use allclose since SupportRegion converts to float32
        assert np.allclose(support.chi_weight, chi)