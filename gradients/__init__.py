# -*- coding: utf-8 -*-
"""
Gradients Module
================

Contains gradient computation implementations for both NumPy (CPU) and PyTorch (GPU).

NumPy (CPU):
    - gradient_engine: Hand-derived natural gradients
    - gradient_terms: KL gradient formulas
    - free_energy_clean: Energy computation

PyTorch (GPU):
    - torch_energy: Differentiable energy functions with autograd
    - torch_gradients: Additional PyTorch gradient utilities
"""

# NumPy-based gradients (original, hand-derived)
from .gradient_engine import compute_natural_gradients
from .gradient_terms import (
    grad_self_wrt_q,
    grad_self_wrt_p,
    grad_kl_source,
    grad_kl_target,
    grad_kl_wrt_transport,
    cholesky_gradient,
)
from .free_energy_clean import compute_total_free_energy
from .softmax_grads import (
    compute_softmax_weights,
    compute_softmax_derivative_fields,
    compute_softmax_coupling_gradients,
)
from .update_engine import UpdateEngine

# PyTorch-based (GPU with autograd)
from .torch_energy import (
    FreeEnergy,
    kl_divergence_gaussian,
    transport_gaussian,
    compute_transport_operator,
    batched_pairwise_kl,
)

__all__ = [
    # NumPy (CPU)
    'compute_natural_gradients',
    'grad_self_wrt_q',
    'grad_self_wrt_p',
    'grad_kl_source',
    'grad_kl_target',
    'grad_kl_wrt_transport',
    'cholesky_gradient',
    'compute_total_free_energy',
    'compute_softmax_weights',
    'compute_softmax_derivative_fields',
    'compute_softmax_coupling_gradients',
    'UpdateEngine',
    # PyTorch (GPU)
    'FreeEnergy',
    'kl_divergence_gaussian',
    'transport_gaussian',
    'compute_transport_operator',
    'batched_pairwise_kl',
]
