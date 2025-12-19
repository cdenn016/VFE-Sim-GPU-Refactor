"""
Experimental Transformer Components
===================================

This module contains experimental features that are not yet part of the core
VFE transformer implementation.

Currently includes:
- hamiltonian_ffn: Symplectic Hamiltonian dynamics on belief space
"""

from transformer.experimental.hamiltonian_ffn import (
    HamiltonianFFN,
    MassConfig,
    PhaseSpaceState,
    LeapfrogIntegrator,
)

__all__ = [
    'HamiltonianFFN',
    'MassConfig',
    'PhaseSpaceState',
    'LeapfrogIntegrator',
]
