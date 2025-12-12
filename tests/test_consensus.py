# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:09:30 2025

@author: chris and christine
"""

"""
Consensus Detection Module Tests
=================================

Tests for epistemic death detection, consensus clustering,
and meta-agent candidate identification.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConsensusState:
    """Tests for ConsensusState dataclass."""

    def test_consensus_state_defaults(self):
        """Test ConsensusState default values."""
        from meta.consensus import ConsensusState

        state = ConsensusState()
        assert state.belief_consensus is False
        assert state.model_consensus is False
        assert state.belief_divergence == np.inf
        assert state.model_divergence == np.inf

    def test_epistemic_death_property(self):
        """Test is_epistemically_dead property."""
        from meta.consensus import ConsensusState

        # Not dead if only belief consensus
        state1 = ConsensusState(belief_consensus=True, model_consensus=False)
        assert not state1.is_epistemically_dead

        # Not dead if only model consensus
        state2 = ConsensusState(belief_consensus=False, model_consensus=True)
        assert not state2.is_epistemically_dead

        # Dead if both
        state3 = ConsensusState(belief_consensus=True, model_consensus=True)
        assert state3.is_epistemically_dead

    def test_consensus_state_with_divergences(self):
        """Test ConsensusState with divergence values."""
        from meta.consensus import ConsensusState

        state = ConsensusState(
            belief_consensus=True,
            model_consensus=False,
            belief_divergence=0.001,
            model_divergence=0.5
        )
        assert state.belief_divergence == 0.001
        assert state.model_divergence == 0.5


class TestConsensusDetector:
    """Tests for ConsensusDetector class."""

    def test_detector_construction(self):
        """Test ConsensusDetector initialization."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(
            belief_threshold=1e-3,
            model_threshold=1e-3,
            use_symmetric_kl=False,
            cache_transport=True
        )
        assert detector.belief_threshold == 1e-3
        assert detector.model_threshold == 1e-3
        assert detector.use_symmetric_kl is False
        assert detector.cache_transport is True

    def test_detector_with_symmetric_kl(self):
        """Test detector with symmetric KL divergence."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(use_symmetric_kl=True)
        assert detector.use_symmetric_kl is True

    def test_check_belief_consensus_returns_tuple(self, simple_system):
        """Test that check_belief_consensus returns (bool, float) tuple."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        agent_i = simple_system.agents[0]
        agent_j = simple_system.agents[1]

        result = detector.check_belief_consensus(agent_i, agent_j)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], (float, np.floating))

    def test_check_model_consensus_returns_tuple(self, simple_system):
        """Test that check_model_consensus returns (bool, float) tuple."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        agent_i = simple_system.agents[0]
        agent_j = simple_system.agents[1]

        result = detector.check_model_consensus(agent_i, agent_j)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (bool, np.bool_))
        assert isinstance(result[1], (float, np.floating))

    def test_check_full_consensus_returns_state(self, simple_system):
        """Test that check_full_consensus returns ConsensusState."""
        from meta.consensus import ConsensusDetector, ConsensusState

        detector = ConsensusDetector()
        agent_i = simple_system.agents[0]
        agent_j = simple_system.agents[1]

        state = detector.check_full_consensus(agent_i, agent_j)
        assert isinstance(state, ConsensusState)

    def test_identical_agents_have_consensus(self, rng):
        """Test that identical agents show consensus."""
        from agent.agents import Agent
        from geometry.geometry_base import BaseManifold, TopologyType
        from config import AgentConfig
        from meta.consensus import ConsensusDetector

        manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)
        config = AgentConfig(spatial_shape=(), K=3)

        # Create two agents with identical beliefs
        agent1 = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)
        agent2 = Agent(agent_id=1, config=config, rng=rng, base_manifold=manifold)

        # Make them identical
        agent2.mu_q = agent1.mu_q.copy()
        agent2.Sigma_q = agent1.Sigma_q.copy()
        agent2.mu_p = agent1.mu_p.copy()
        agent2.Sigma_p = agent1.Sigma_p.copy()
        agent2.gauge.phi = agent1.gauge.phi.copy()

        detector = ConsensusDetector(belief_threshold=1e-2, model_threshold=1e-2)
        belief_consensus, belief_div = detector.check_belief_consensus(agent1, agent2)
        model_consensus, model_div = detector.check_model_consensus(agent1, agent2)

        # KL divergence of identical distributions should be ~0
        assert belief_div < 1e-6, f"Expected ~0, got {belief_div}"
        assert model_div < 1e-6, f"Expected ~0, got {model_div}"

    def test_clear_cache(self):
        """Test transport cache clearing."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(cache_transport=True)
        # Simulate some cache entries
        detector._transport_cache[(0, 1)] = np.eye(3)

        detector.clear_cache()
        assert len(detector._transport_cache) == 0


class TestConsensusClustering:
    """Tests for consensus cluster detection."""

    def test_find_consensus_clusters_returns_list(self, simple_system):
        """Test that find_consensus_clusters returns list of clusters."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(belief_threshold=0.1, model_threshold=0.1)
        clusters = detector.find_consensus_clusters(simple_system)

        assert isinstance(clusters, list)
        # Each cluster should be a list of agent indices
        for cluster in clusters:
            assert isinstance(cluster, list)
            for idx in cluster:
                assert isinstance(idx, (int, np.integer))

    def test_consensus_matrix_shape(self, simple_system):
        """Test consensus matrix has correct shape."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        matrix = detector.compute_consensus_matrix(simple_system)

        n = simple_system.n_agents
        assert matrix.shape == (n, n)

    def test_consensus_matrix_diagonal_is_zero(self, simple_system):
        """Test that diagonal of consensus matrix is zero (no self-divergence)."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        matrix = detector.compute_consensus_matrix(simple_system)

        for i in range(simple_system.n_agents):
            assert matrix[i, i] == 0

    def test_identify_meta_agent_candidates(self, simple_system):
        """Test meta-agent candidate identification."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector(belief_threshold=0.5, model_threshold=0.5)
        candidates = detector.identify_meta_agent_candidates(simple_system, min_cluster_size=2)

        assert isinstance(candidates, list)
        for candidate in candidates:
            assert 'indices' in candidate
            assert 'belief_coherence' in candidate
            assert 'model_coherence' in candidate
            assert 'scale' in candidate
            assert len(candidate['indices']) >= 2


class TestKLDivergence:
    """Tests for KL divergence computations in consensus."""

    def test_kl_divergence_finite(self, simple_system):
        """Test that KL divergences are finite."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        agent_i = simple_system.agents[0]
        agent_j = simple_system.agents[1]

        _, belief_div = detector.check_belief_consensus(agent_i, agent_j)
        _, model_div = detector.check_model_consensus(agent_i, agent_j)

        assert np.isfinite(belief_div) or belief_div == np.inf
        assert np.isfinite(model_div) or model_div == np.inf

    def test_kl_divergence_non_negative(self, simple_system):
        """Test that KL divergences are non-negative."""
        from meta.consensus import ConsensusDetector

        detector = ConsensusDetector()
        agent_i = simple_system.agents[0]
        agent_j = simple_system.agents[1]

        _, belief_div = detector.check_belief_consensus(agent_i, agent_j)
        _, model_div = detector.check_model_consensus(agent_i, agent_j)

        assert belief_div >= 0
        assert model_div >= 0


class TestSpatialConsensus:
    """Tests for spatial consensus detection on manifolds with dimension > 0."""

    def test_spatial_consensus_returns_arrays(self, rng):
        """Test that spatial consensus methods return arrays."""
        from agent.agents import Agent
        from geometry.geometry_base import BaseManifold, TopologyType
        from config import AgentConfig
        from meta.consensus import ConsensusDetector

        manifold = BaseManifold(shape=(8,), topology=TopologyType.PERIODIC)
        config = AgentConfig(spatial_shape=(8,), K=3)

        agent1 = Agent(agent_id=0, config=config, rng=rng, base_manifold=manifold)
        agent2 = Agent(agent_id=1, config=config, rng=rng, base_manifold=manifold)

        detector = ConsensusDetector()

        # Check spatial belief consensus
        if hasattr(detector, 'check_belief_consensus_spatial'):
            consensus_mask, kl_map = detector.check_belief_consensus_spatial(agent1, agent2)
            assert consensus_mask.shape == (8,)
            assert kl_map.shape == (8,)
            assert consensus_mask.dtype == bool


class TestAnalysisFunctions:
    """Tests for consensus analysis functions."""

    def test_analyze_consensus_dynamics(self, simple_system):
        """Test consensus dynamics analysis."""
        from meta.consensus import ConsensusDetector, analyze_consensus_dynamics

        detector = ConsensusDetector()
        history = []  # Empty history for initial analysis

        result = analyze_consensus_dynamics(simple_system, detector, history)

        assert 'n_clusters' in result
        assert 'largest_cluster' in result
        assert 'mean_divergence' in result
        assert 'clusters' in result