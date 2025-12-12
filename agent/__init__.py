# -*- coding: utf-8 -*-
"""
Agent Module
============

Contains agent implementations for both NumPy (CPU) and PyTorch (GPU).

NumPy (CPU):
    - Agent: Standard agent with smooth sections
    - MultiAgentSystem: Multi-agent system with energy/gradient computation

PyTorch (GPU):
    - TensorAgent: GPU-accelerated agent with autograd
    - TensorSystem: Batched multi-agent system
    - TensorTrainer: GPU training with automatic differentiation
"""

# NumPy-based agents (original)
from .agents import Agent
from .system import MultiAgentSystem
from .trainer import Trainer, TrainingConfig, TrainingHistory
from .hamiltonian_trainer import HamiltonianTrainer, HamiltonianHistory
from .masking import SupportRegionSmooth, SupportPatternsSmooth, MaskConfig

# PyTorch-based agents (GPU)
from .tensor_agent import TensorAgent
from .tensor_system import TensorSystem
from .tensor_trainer import TensorTrainer, TensorTrainingConfig, TensorTrainingHistory

__all__ = [
    # NumPy
    'Agent',
    'MultiAgentSystem',
    'Trainer',
    'TrainingConfig',
    'TrainingHistory',
    'HamiltonianTrainer',
    'HamiltonianHistory',
    'SupportRegionSmooth',
    'SupportPatternsSmooth',
    'MaskConfig',
    # PyTorch
    'TensorAgent',
    'TensorSystem',
    'TensorTrainer',
    'TensorTrainingConfig',
    'TensorTrainingHistory',
]
