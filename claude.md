# VFE Simulation Suite - GPU Refactoring Plan (PyTorch/CUDA)

## Executive Summary

This document outlines a comprehensive plan to refactor the VFE (Variational Free Energy) simulation suite from CPU-bound NumPy to GPU-accelerated PyTorch. The RTX 5090 has 32GB VRAM and 21,760 CUDA cores - we should be able to achieve **50-100x speedup** for large multi-agent simulations.

**Key Insight**: The current implementation hand-derives all gradients (~1300 lines in `gradient_engine.py` + `gradient_terms.py`). With PyTorch autograd, we can:
1. Eliminate manual gradient derivation entirely
2. Get automatic differentiation through complex operations (transport, softmax, KL)
3. Easily extend to new energy terms without re-deriving gradients

---

## Current Architecture Analysis

### Bottlenecks Identified

| Component | Current | Issue | Impact |
|-----------|---------|-------|--------|
| Agent State | NumPy arrays | CPU-bound, no autograd | **CRITICAL** |
| Gradient Computation | Hand-derived | ~1300 LOC, error-prone | **CRITICAL** |
| Parallel Execution | `joblib` | GIL-limited, CPU only | **HIGH** |
| Pairwise Operations | Nested loops | O(N²) sequential | **HIGH** |
| Spatial Integration | Per-point loops | Cache-inefficient | **MEDIUM** |
| Cholesky/Inverse | `np.linalg` | No GPU acceleration | **MEDIUM** |

### File Dependencies (Simulation Flow)

```
simulation_config.py    → Configuration dataclass
         ↓
simulation_runner.py    → Orchestration
         ↓
    ┌────┴────┐
    ↓         ↓
agent/     gradients/
├─agents.py            ├─gradient_engine.py     ← HAND-DERIVED GRADIENTS
├─system.py            ├─gradient_terms.py      ← KL gradient formulas
├─trainer.py           ├─softmax_grads.py       ← ∂β/∂θ derivations
└─hamiltonian_trainer.py └─free_energy_clean.py ← Energy computation
         ↓
    math_utils/
    ├─transport.py      ← Ω = exp(φ_i)exp(-φ_j)
    ├─push_pull.py      ← Gaussian transport
    ├─sigma.py          ← Covariance operations
    └─torch_backend.py  ← Partial GPU support (extend this)
```

### Current Gradient Derivation (REPLACE WITH AUTOGRAD)

```python
# gradient_terms.py - Currently 400+ lines of manual gradients
def grad_self_wrt_q(mu_q, Sigma_q, mu_p, Sigma_p):
    """∂KL(q||p)/∂(μ_q, Σ_q) - MANUAL DERIVATION"""
    Sigma_p_inv = safe_inv(Sigma_p)
    Sigma_q_inv = safe_inv(Sigma_q)

    grad_mu = Sigma_p_inv @ (mu_q - mu_p)
    grad_Sigma = 0.5 * (Sigma_p_inv - Sigma_q_inv)

    return grad_mu, grad_Sigma

# WITH AUTOGRAD - just 3 lines:
def compute_kl_with_grad(mu_q, Sigma_q, mu_p, Sigma_p):
    kl = kl_divergence(mu_q, Sigma_q, mu_p, Sigma_p)  # Forward pass
    kl.backward()  # Autograd computes all gradients
    return mu_q.grad, Sigma_q.grad
```

---

## Refactoring Strategy

### Phase 1: Core Tensor Infrastructure (Week 1)

**Goal**: Replace NumPy agent state with PyTorch tensors that support autograd.

#### 1.1 Create `TensorAgent` class

```python
# agent/tensor_agent.py (NEW)
import torch
import torch.nn as nn

class TensorAgent(nn.Module):
    """
    Agent with PyTorch tensor state for GPU acceleration + autograd.

    State tensors (all require_grad=True):
        mu_q: (K,) or (*S, K) belief mean
        L_q: (K, K) or (*S, K, K) belief Cholesky factor (lower triangular)
        mu_p: (K,) or (*S, K) prior mean
        L_p: (K, K) or (*S, K, K) prior Cholesky factor
        phi: (3,) or (*S, 3) gauge field (so(3) Lie algebra)

    Why Cholesky (L) instead of Sigma?
        1. Guaranteed positive definite: Σ = LL^T always SPD
        2. Unconstrained optimization: L can be any lower triangular
        3. Faster: O(K²) params vs O(K²) for full Sigma
    """

    def __init__(self, K: int, spatial_shape: tuple = (), device='cuda', dtype=torch.float32):
        super().__init__()
        self.K = K
        self.spatial_shape = spatial_shape
        self.device = device
        self.dtype = dtype

        # Determine shapes
        if spatial_shape == ():
            # 0D: Point agent (particle)
            mu_shape = (K,)
            L_shape = (K, K)
            phi_shape = (3,)
        else:
            # ND: Field agent
            mu_shape = (*spatial_shape, K)
            L_shape = (*spatial_shape, K, K)
            phi_shape = (*spatial_shape, 3)

        # === TRAINABLE PARAMETERS ===
        # Belief distribution q = N(mu_q, L_q @ L_q.T)
        self.mu_q = nn.Parameter(torch.zeros(mu_shape, device=device, dtype=dtype))
        self.L_q = nn.Parameter(torch.eye(K, device=device, dtype=dtype).expand(*L_shape).clone())

        # Prior distribution p = N(mu_p, L_p @ L_p.T)
        self.mu_p = nn.Parameter(torch.zeros(mu_shape, device=device, dtype=dtype))
        self.L_p = nn.Parameter(torch.eye(K, device=device, dtype=dtype).expand(*L_shape).clone())

        # Gauge field φ ∈ so(3)
        self.phi = nn.Parameter(torch.zeros(phi_shape, device=device, dtype=dtype))

    @property
    def Sigma_q(self) -> torch.Tensor:
        """Belief covariance Σ_q = L_q L_q^T (computed, not stored)."""
        return self.L_q @ self.L_q.transpose(-1, -2)

    @property
    def Sigma_p(self) -> torch.Tensor:
        """Prior covariance Σ_p = L_p L_p^T (computed, not stored)."""
        return self.L_p @ self.L_p.transpose(-1, -2)

    def initialize(self, mu_scale=1.0, sigma_scale=1.0, phi_scale=0.1, rng=None):
        """Initialize with random state."""
        if rng is None:
            rng = torch.Generator(device=self.device)

        with torch.no_grad():
            self.mu_q.normal_(0, mu_scale, generator=rng)
            self.mu_p.normal_(0, mu_scale, generator=rng)

            # Initialize L as scaled identity + small perturbation
            self.L_q.copy_(sigma_scale * torch.eye(self.K, device=self.device, dtype=self.dtype))
            self.L_p.copy_(sigma_scale * torch.eye(self.K, device=self.device, dtype=self.dtype))

            # Small random gauge field
            self.phi.uniform_(-phi_scale, phi_scale, generator=rng)
```

#### 1.2 Create `TensorSystem` for batched multi-agent operations

```python
# agent/tensor_system.py (NEW)
class TensorSystem(nn.Module):
    """
    Multi-agent system with batched GPU operations.

    All agents' states are stored in batched tensors for efficient GPU execution:
        mu_q_all: (N, K) or (N, *S, K) - all agent belief means
        L_q_all: (N, K, K) or (N, *S, K, K) - all agent belief Cholesky factors
        ... etc

    Key insight: N×N pairwise operations become single batched matmuls!
    """

    def __init__(self, n_agents: int, K: int, spatial_shape: tuple = (),
                 generators: torch.Tensor = None, device='cuda'):
        super().__init__()
        self.n_agents = n_agents
        self.K = K
        self.spatial_shape = spatial_shape
        self.device = device

        # SO(3) generators for transport (3, K, K)
        if generators is None:
            from math_utils.generators import generate_so3_generators
            generators = torch.tensor(generate_so3_generators(K), device=device, dtype=torch.float32)
        self.register_buffer('generators', generators)

        # === BATCHED STATE TENSORS ===
        self.agents = nn.ModuleList([
            TensorAgent(K, spatial_shape, device) for _ in range(n_agents)
        ])

    def get_batched_state(self) -> dict:
        """Stack all agent states into batched tensors."""
        return {
            'mu_q': torch.stack([a.mu_q for a in self.agents]),      # (N, K)
            'Sigma_q': torch.stack([a.Sigma_q for a in self.agents]), # (N, K, K)
            'mu_p': torch.stack([a.mu_p for a in self.agents]),      # (N, K)
            'Sigma_p': torch.stack([a.Sigma_p for a in self.agents]), # (N, K, K)
            'phi': torch.stack([a.phi for a in self.agents]),        # (N, 3)
        }

    def compute_all_pairwise_transport(self) -> torch.Tensor:
        """
        Compute all N×N transport operators Ω_ij = exp(φ_i) exp(-φ_j).

        Returns:
            Omega: (N, N, K, K) transport operators
        """
        phi = torch.stack([a.phi for a in self.agents])  # (N, 3)
        return batched_transport(phi, phi, self.generators)  # (N, N, K, K)
```

### Phase 2: Autograd Energy Functions (Week 2)

**Goal**: Implement free energy as differentiable PyTorch functions.

#### 2.1 Differentiable KL Divergence

```python
# gradients/torch_energy.py (NEW)
import torch
import torch.nn.functional as F

def kl_divergence_gaussian(
    mu_q: torch.Tensor,    # (..., K)
    Sigma_q: torch.Tensor, # (..., K, K)
    mu_p: torch.Tensor,    # (..., K)
    Sigma_p: torch.Tensor, # (..., K, K)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    KL(q || p) for multivariate Gaussians - FULLY DIFFERENTIABLE.

    Autograd handles ∂KL/∂μ_q, ∂KL/∂Σ_q, ∂KL/∂μ_p, ∂KL/∂Σ_p automatically!

    KL = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - K + log|Σ_p|/|Σ_q|)
    """
    K = mu_q.shape[-1]

    # Regularize for numerical stability
    eye = torch.eye(K, device=mu_q.device, dtype=mu_q.dtype)
    Sigma_q = Sigma_q + eps * eye
    Sigma_p = Sigma_p + eps * eye

    # Cholesky decomposition (differentiable!)
    L_q = torch.linalg.cholesky(Sigma_q)
    L_p = torch.linalg.cholesky(Sigma_p)

    # Log determinants: log|Σ| = 2 * Σ log(L_ii)
    logdet_q = 2 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)
    logdet_p = 2 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)

    # Trace term: tr(Σ_p^{-1} Σ_q)
    Sigma_p_inv = torch.cholesky_inverse(L_p)
    trace_term = torch.sum(Sigma_p_inv * Sigma_q, dim=(-2, -1))

    # Quadratic term: (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q)
    delta = mu_p - mu_q
    # Solve L_p y = delta, then ||y||² = delta^T Σ_p^{-1} delta
    y = torch.linalg.solve_triangular(L_p, delta.unsqueeze(-1), upper=False)
    quad_term = torch.sum(y ** 2, dim=(-2, -1))

    # KL divergence
    kl = 0.5 * (trace_term + quad_term - K + logdet_p - logdet_q)

    return torch.clamp(kl, min=0.0)


def transport_gaussian(
    mu: torch.Tensor,      # (..., K)
    Sigma: torch.Tensor,   # (..., K, K)
    Omega: torch.Tensor    # (..., K, K)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transport Gaussian by operator: (Ω μ, Ω Σ Ω^T) - DIFFERENTIABLE.

    Autograd computes gradients through transport automatically!
    """
    mu_t = torch.einsum('...ij,...j->...i', Omega, mu)
    Sigma_t = Omega @ Sigma @ Omega.transpose(-2, -1)
    # Symmetrize for numerical stability
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-2, -1))
    return mu_t, Sigma_t


def compute_transport_operator(
    phi_i: torch.Tensor,     # (..., 3)
    phi_j: torch.Tensor,     # (..., 3)
    generators: torch.Tensor # (3, K, K)
) -> torch.Tensor:
    """
    Compute Ω_ij = exp(φ_i · J) exp(-φ_j · J) - DIFFERENTIABLE.

    Uses torch.linalg.matrix_exp which supports autograd!
    """
    # Contract gauge fields with generators: X = φ · J
    X_i = torch.einsum('...a,aij->...ij', phi_i, generators)  # (..., K, K)
    X_j = torch.einsum('...a,aij->...ij', phi_j, generators)  # (..., K, K)

    # Matrix exponentials (differentiable!)
    exp_i = torch.linalg.matrix_exp(X_i)
    exp_neg_j = torch.linalg.matrix_exp(-X_j)

    # Compose: Ω = exp(φ_i · J) @ exp(-φ_j · J)
    Omega = exp_i @ exp_neg_j

    return Omega
```

#### 2.2 Complete Free Energy Function

```python
# gradients/torch_energy.py (continued)

class FreeEnergy(nn.Module):
    """
    Differentiable free energy functional.

    F = α·Σᵢ KL(qᵢ||pᵢ)                           [Self-coupling]
      + λ_β·Σᵢⱼ βᵢⱼ·KL(qᵢ||Ωᵢⱼ[qⱼ])               [Belief alignment]
      + λ_γ·Σᵢⱼ γᵢⱼ·KL(pᵢ||Ωᵢⱼ[pⱼ])               [Prior alignment]

    Call .backward() on the output to get all gradients via autograd!
    """

    def __init__(self, config: 'SimulationConfig'):
        super().__init__()
        self.lambda_self = config.lambda_self
        self.lambda_belief = config.lambda_belief_align
        self.lambda_prior = config.lambda_prior_align
        self.kappa_beta = config.kappa_beta
        self.kappa_gamma = config.kappa_gamma

    def forward(self, system: 'TensorSystem') -> torch.Tensor:
        """Compute total free energy (differentiable)."""
        state = system.get_batched_state()
        N = system.n_agents

        # === 1. Self-coupling: Σᵢ KL(qᵢ||pᵢ) ===
        E_self = kl_divergence_gaussian(
            state['mu_q'], state['Sigma_q'],
            state['mu_p'], state['Sigma_p']
        ).sum()

        # === 2. Belief alignment: Σᵢⱼ βᵢⱼ·KL(qᵢ||Ωᵢⱼ[qⱼ]) ===
        if self.lambda_belief > 0:
            # Compute all pairwise transport operators
            Omega = system.compute_all_pairwise_transport()  # (N, N, K, K)

            # Transport all beliefs
            mu_q = state['mu_q']       # (N, K)
            Sigma_q = state['Sigma_q'] # (N, K, K)

            # Expand for broadcasting: (N, 1, K) vs (1, N, K)
            mu_i = mu_q.unsqueeze(1)         # (N, 1, K)
            Sigma_i = Sigma_q.unsqueeze(1)   # (N, 1, K, K)
            mu_j = mu_q.unsqueeze(0)         # (1, N, K)
            Sigma_j = Sigma_q.unsqueeze(0)   # (1, N, K, K)

            # Transport j to i's frame: Ωᵢⱼ[qⱼ]
            mu_j_t = torch.einsum('ijkl,jl->ijk', Omega, mu_q)  # (N, N, K)
            Sigma_j_t = Omega @ Sigma_j @ Omega.transpose(-2, -1)  # (N, N, K, K)

            # Compute all N×N KL divergences
            KL_belief = kl_divergence_gaussian(
                mu_i.expand(N, N, -1), Sigma_i.expand(N, N, -1, -1),
                mu_j_t, Sigma_j_t
            )  # (N, N)

            # Softmax attention weights: βᵢⱼ = softmax(-κ·KLᵢⱼ)
            beta = F.softmax(-self.kappa_beta * KL_belief, dim=-1)  # (N, N)

            # Weighted sum
            E_belief = (beta * KL_belief).sum()
        else:
            E_belief = torch.tensor(0.0, device=system.device)

        # === 3. Prior alignment (similar structure) ===
        if self.lambda_prior > 0:
            # ... similar to belief alignment but with mu_p, Sigma_p
            E_prior = self._compute_prior_alignment(system, state, Omega)
        else:
            E_prior = torch.tensor(0.0, device=system.device)

        # === Total Free Energy ===
        F_total = (self.lambda_self * E_self +
                   self.lambda_belief * E_belief +
                   self.lambda_prior * E_prior)

        return F_total
```

### Phase 3: GPU-Accelerated Training Loop (Week 3)

**Goal**: Replace CPU training loop with GPU-native version.

#### 3.1 Tensor Trainer

```python
# agent/tensor_trainer.py (NEW)
import torch
from torch.optim import Adam, SGD

class TensorTrainer:
    """
    GPU-accelerated training with PyTorch optimizers.

    Key advantages:
    1. Autograd computes all gradients (no hand-derivation!)
    2. Built-in optimizers (Adam, SGD, etc.)
    3. torch.compile for kernel fusion
    4. Mixed precision training (FP16) for 2x speedup
    """

    def __init__(
        self,
        system: TensorSystem,
        config: 'SimulationConfig',
        use_compile: bool = True,
        mixed_precision: bool = False
    ):
        self.system = system
        self.config = config
        self.device = system.device

        # Free energy module
        self.free_energy = FreeEnergy(config).to(self.device)

        # Compile for kernel fusion (PyTorch 2.0+)
        if use_compile:
            self.free_energy = torch.compile(
                self.free_energy,
                mode='reduce-overhead'
            )

        # Optimizer (adaptive learning rates per parameter group)
        self.optimizer = Adam([
            {'params': [a.mu_q for a in system.agents], 'lr': config.lr_mu_q},
            {'params': [a.L_q for a in system.agents], 'lr': config.lr_sigma_q},
            {'params': [a.mu_p for a in system.agents], 'lr': config.lr_mu_p},
            {'params': [a.L_p for a in system.agents], 'lr': config.lr_sigma_p},
            {'params': [a.phi for a in system.agents], 'lr': config.lr_phi},
        ])

        # Mixed precision (optional, 2x speedup on tensor cores)
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')

    def step(self) -> float:
        """Single training step with autograd."""
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.amp.autocast('cuda'):
                F = self.free_energy(self.system)
            self.scaler.scale(F).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            F = self.free_energy(self.system)
            F.backward()  # Autograd computes ALL gradients!
            self.optimizer.step()

        return F.item()

    def train(self, n_steps: int, log_every: int = 10) -> list:
        """Full training loop."""
        history = []

        for step in range(n_steps):
            F = self.step()
            history.append(F)

            if step % log_every == 0:
                print(f"Step {step:4d} | F = {F:.4f}")

        return history
```

#### 3.2 Hamiltonian Dynamics (Symplectic Integration)

```python
# agent/tensor_hamiltonian.py (NEW)
class TensorHamiltonianTrainer:
    """
    Hamiltonian dynamics with GPU-accelerated symplectic integration.

    Phase space: (θ, p) where θ = (μ_q, L_q, μ_p, L_p, φ)

    H = T(p) + V(θ)
      = (1/2M) Σᵢ ||pᵢ||² + F(θ)

    Symplectic integrator (Störmer-Verlet):
        p_{n+1/2} = p_n - (dt/2) ∇V(θ_n)
        θ_{n+1} = θ_n + (dt/M) p_{n+1/2}
        p_{n+1} = p_{n+1/2} - (dt/2) ∇V(θ_{n+1})
    """

    def __init__(self, system: TensorSystem, config: 'SimulationConfig'):
        self.system = system
        self.free_energy = FreeEnergy(config).to(system.device)

        # Initialize momenta (conjugate to all parameters)
        self.momenta = {}
        for i, agent in enumerate(system.agents):
            self.momenta[i] = {
                'mu_q': torch.zeros_like(agent.mu_q),
                'L_q': torch.zeros_like(agent.L_q),
                'phi': torch.zeros_like(agent.phi),
            }

        self.mass = config.hamiltonian_mass_scale
        self.friction = config.hamiltonian_friction

    def step(self, dt: float) -> dict:
        """Symplectic Verlet step."""
        # Compute ∇V(θ) via autograd
        F = self.free_energy(self.system)
        F.backward()

        # Half-step momentum
        for i, agent in enumerate(self.system.agents):
            self.momenta[i]['mu_q'] -= 0.5 * dt * agent.mu_q.grad
            self.momenta[i]['L_q'] -= 0.5 * dt * agent.L_q.grad
            self.momenta[i]['phi'] -= 0.5 * dt * agent.phi.grad

        # Full-step position
        with torch.no_grad():
            for i, agent in enumerate(self.system.agents):
                agent.mu_q += (dt / self.mass) * self.momenta[i]['mu_q']
                agent.L_q += (dt / self.mass) * self.momenta[i]['L_q']
                agent.phi += (dt / self.mass) * self.momenta[i]['phi']

        # Zero grads and recompute
        self.system.zero_grad()
        F = self.free_energy(self.system)
        F.backward()

        # Half-step momentum (with optional friction)
        friction_factor = torch.exp(-self.friction * dt)
        for i, agent in enumerate(self.system.agents):
            self.momenta[i]['mu_q'] -= 0.5 * dt * agent.mu_q.grad
            self.momenta[i]['mu_q'] *= friction_factor
            # ... similar for L_q, phi

        # Compute kinetic energy
        T = sum(
            0.5 / self.mass * (
                torch.sum(m['mu_q']**2) +
                torch.sum(m['L_q']**2) +
                torch.sum(m['phi']**2)
            )
            for m in self.momenta.values()
        )

        return {'F': F.item(), 'T': T.item(), 'H': F.item() + T.item()}
```

### Phase 4: Migration Path (Week 4)

**Goal**: Incremental migration without breaking existing code.

#### 4.1 Conversion Utilities

```python
# math_utils/migration.py (NEW)
def numpy_agent_to_tensor(agent: 'Agent', device='cuda') -> TensorAgent:
    """Convert existing NumPy agent to TensorAgent."""
    t_agent = TensorAgent(agent.K, agent.base_manifold.shape, device)

    with torch.no_grad():
        t_agent.mu_q.copy_(torch.tensor(agent.mu_q, device=device))
        t_agent.mu_p.copy_(torch.tensor(agent.mu_p, device=device))
        t_agent.phi.copy_(torch.tensor(agent.gauge.phi, device=device))

        # Convert Sigma to Cholesky L
        L_q = np.linalg.cholesky(agent.Sigma_q + 1e-6 * np.eye(agent.K))
        L_p = np.linalg.cholesky(agent.Sigma_p + 1e-6 * np.eye(agent.K))
        t_agent.L_q.copy_(torch.tensor(L_q, device=device))
        t_agent.L_p.copy_(torch.tensor(L_p, device=device))

    return t_agent


def tensor_agent_to_numpy(t_agent: TensorAgent) -> dict:
    """Convert TensorAgent state back to NumPy for compatibility."""
    return {
        'mu_q': t_agent.mu_q.detach().cpu().numpy(),
        'Sigma_q': t_agent.Sigma_q.detach().cpu().numpy(),
        'mu_p': t_agent.mu_p.detach().cpu().numpy(),
        'Sigma_p': t_agent.Sigma_p.detach().cpu().numpy(),
        'phi': t_agent.phi.detach().cpu().numpy(),
    }
```

#### 4.2 Hybrid Mode (Gradual Migration)

```python
# simulation_runner.py - Add GPU mode dispatch
def run_training(system, cfg: SimulationConfig, output_dir: Path):
    """Unified training interface with GPU acceleration."""

    if cfg.use_gpu and cfg.gpu_backend == 'pytorch':
        # NEW: Full GPU path with autograd
        return _run_gpu_training(system, cfg, output_dir)
    else:
        # Existing CPU path (unchanged)
        return _run_cpu_training(system, cfg, output_dir)


def _run_gpu_training(system, cfg, output_dir):
    """GPU-accelerated training with PyTorch."""
    from agent.tensor_system import TensorSystem
    from agent.tensor_trainer import TensorTrainer

    # Convert to GPU tensors
    print("[GPU] Converting system to TensorSystem...")
    tensor_system = TensorSystem.from_numpy_system(system, device='cuda')

    # Create trainer with autograd
    trainer = TensorTrainer(
        tensor_system, cfg,
        use_compile=cfg.torch_compile,
        mixed_precision=(cfg.torch_dtype == 'float16')
    )

    # Train
    print(f"[GPU] Training for {cfg.n_steps} steps...")
    history = trainer.train(cfg.n_steps, log_every=cfg.log_every)

    # Convert back for analysis
    print("[GPU] Converting results back to NumPy...")
    for i, agent in enumerate(system.agents):
        state = tensor_agent_to_numpy(tensor_system.agents[i])
        agent.mu_q = state['mu_q']
        agent.Sigma_q = state['Sigma_q']
        # ... etc

    return history
```

---

## Performance Optimization

### Batched Operations (Critical for N² Scaling)

```python
# math_utils/batched_ops.py (NEW)
@torch.compile(mode='max-autotune')
def batched_pairwise_kl(
    mu: torch.Tensor,      # (N, K)
    Sigma: torch.Tensor,   # (N, K, K)
    Omega: torch.Tensor,   # (N, N, K, K)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute all N×N pairwise KL divergences in one kernel.

    KL[i,j] = KL(qᵢ || Ωᵢⱼ[qⱼ])

    Performance: ~50-100x faster than nested Python loops!
    """
    N, K = mu.shape
    device = mu.device
    dtype = mu.dtype

    # Transport all distributions
    # mu_j_t[i,j] = Omega[i,j] @ mu[j]
    mu_t = torch.einsum('ijkl,jl->ijk', Omega, mu)  # (N, N, K)

    # Sigma_j_t[i,j] = Omega[i,j] @ Sigma[j] @ Omega[i,j].T
    Sigma_t = Omega @ Sigma.unsqueeze(0) @ Omega.transpose(-2, -1)  # (N, N, K, K)

    # Broadcast source distributions
    mu_i = mu.unsqueeze(1).expand(N, N, K)           # (N, N, K)
    Sigma_i = Sigma.unsqueeze(1).expand(N, N, K, K)  # (N, N, K, K)

    # Vectorized KL computation
    eye = torch.eye(K, device=device, dtype=dtype)
    Sigma_t_reg = Sigma_t + eps * eye
    Sigma_i_reg = Sigma_i + eps * eye

    # Batched Cholesky
    L_t = torch.linalg.cholesky(Sigma_t_reg)
    L_i = torch.linalg.cholesky(Sigma_i_reg)

    # Log determinants
    logdet_t = 2 * torch.diagonal(L_t, dim1=-2, dim2=-1).log().sum(-1)
    logdet_i = 2 * torch.diagonal(L_i, dim1=-2, dim2=-1).log().sum(-1)

    # Trace term
    Sigma_t_inv = torch.cholesky_inverse(L_t)
    trace = (Sigma_t_inv * Sigma_i_reg).sum((-2, -1))

    # Quadratic term
    delta = mu_t - mu_i
    y = torch.linalg.solve_triangular(L_t, delta.unsqueeze(-1), upper=False)
    quad = (y ** 2).sum((-2, -1))

    # KL
    kl = 0.5 * (trace + quad - K + logdet_t - logdet_i)

    return kl.clamp(min=0.0)
```

### Memory Optimization

```python
# Gradient checkpointing for large systems
class FreeEnergyCheckpointed(nn.Module):
    """Memory-efficient free energy with gradient checkpointing."""

    def forward(self, system):
        # Checkpoint expensive operations to trade compute for memory
        from torch.utils.checkpoint import checkpoint

        E_self = self.compute_self_energy(system)
        E_belief = checkpoint(
            self.compute_belief_alignment,
            system,
            use_reentrant=False
        )

        return E_self + E_belief
```

---

## Testing Strategy

### 1. Numerical Gradient Verification

```python
# tests/test_torch_gradients.py
def test_kl_gradient_vs_numerical():
    """Verify autograd matches finite differences."""
    mu_q = torch.randn(5, requires_grad=True)
    Sigma_q = torch.eye(5, requires_grad=True)
    mu_p = torch.randn(5, requires_grad=True)
    Sigma_p = torch.eye(5, requires_grad=True)

    # Autograd
    kl = kl_divergence_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)
    kl.backward()

    # Numerical gradient
    eps = 1e-5
    for i in range(5):
        mu_q_plus = mu_q.clone()
        mu_q_plus[i] += eps
        mu_q_minus = mu_q.clone()
        mu_q_minus[i] -= eps

        kl_plus = kl_divergence_gaussian(mu_q_plus.detach(), Sigma_q.detach(), mu_p.detach(), Sigma_p.detach())
        kl_minus = kl_divergence_gaussian(mu_q_minus.detach(), Sigma_q.detach(), mu_p.detach(), Sigma_p.detach())

        numerical_grad = (kl_plus - kl_minus) / (2 * eps)

        assert torch.allclose(mu_q.grad[i], numerical_grad, rtol=1e-3)
```

### 2. CPU vs GPU Consistency

```python
def test_cpu_gpu_consistency():
    """Ensure GPU results match CPU baseline."""
    system_cpu = TensorSystem(n_agents=5, K=3, device='cpu')
    system_gpu = TensorSystem(n_agents=5, K=3, device='cuda')

    # Copy state
    for a_cpu, a_gpu in zip(system_cpu.agents, system_gpu.agents):
        a_gpu.load_state_dict(a_cpu.state_dict())

    F_cpu = FreeEnergy(config)(system_cpu)
    F_gpu = FreeEnergy(config)(system_gpu)

    assert torch.allclose(F_cpu, F_gpu.cpu(), rtol=1e-4)
```

---

## File Structure After Refactoring

```
VFE-Sim-GPU-Refactor/
├── agent/
│   ├── agents.py              # Existing NumPy agent (keep for compatibility)
│   ├── tensor_agent.py        # NEW: PyTorch tensor agent
│   ├── tensor_system.py       # NEW: Batched multi-agent system
│   ├── tensor_trainer.py      # NEW: GPU training with autograd
│   ├── tensor_hamiltonian.py  # NEW: Symplectic GPU integration
│   └── ...
├── gradients/
│   ├── gradient_engine.py     # Keep (CPU fallback)
│   ├── torch_energy.py        # NEW: Differentiable energy functions
│   ├── torch_gradients.py     # Extend existing
│   └── ...
├── math_utils/
│   ├── torch_backend.py       # Extend with batched ops
│   ├── batched_ops.py         # NEW: Compiled batched operations
│   ├── migration.py           # NEW: NumPy ↔ Tensor conversion
│   └── ...
├── simulation_config.py       # Add GPU config options
├── simulation_runner.py       # Add GPU dispatch
└── tests/
    ├── test_torch_gradients.py # NEW: Autograd verification
    └── ...
```

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Gradient Computation** | Hand-derived (~1300 LOC) | Autograd (automatic) |
| **Parallelism** | CPU joblib | GPU CUDA cores |
| **Memory Model** | NumPy arrays | PyTorch tensors on GPU |
| **Optimization** | Manual SGD | torch.optim (Adam, etc.) |
| **Pairwise Ops** | Python loops | Batched kernels |
| **Expected Speedup** | 1x (baseline) | 50-100x |

**Key Benefits**:
1. **Eliminate 1300+ lines of hand-derived gradients** - autograd does it all
2. **RTX 5090 utilization** - 32GB VRAM, 21,760 CUDA cores
3. **torch.compile** - kernel fusion for additional 2-3x speedup
4. **Mixed precision** - FP16 tensor cores for 2x throughput
5. **Future-proof** - easy to add new energy terms without gradient derivation

---

## Next Steps

1. **Phase 1**: Implement `TensorAgent` and `TensorSystem` classes
2. **Phase 2**: Implement differentiable energy functions
3. **Phase 3**: Create `TensorTrainer` with autograd
4. **Phase 4**: Integration tests and benchmarking
5. **Phase 5**: Migrate hierarchical emergence to GPU

---

## Transformer VFE Implementation - Bug Fixes & Architecture Notes

### Critical Bugs Fixed (Dec 2025)

| Commit | Bug | Impact | Fix |
|--------|-----|--------|-----|
| `0bef864` | OOV tokens in val but not train | Val sees unknown tokens | Pass train vocab_mapping to val dataset |
| `7e9f0ab` | Padding not masked in loss | Loss computed on pad tokens | Add `pad_token_id` param, use as `ignore_index` |
| `85ee4df` | Fallback URLs downloaded PTB instead of WikiText-2 | Wrong dataset distribution | Update URLs to WikiText-2 sources |
| `e3b7376` | **CRITICAL**: `.detach()` on VFE outputs | **All gradients broken** | Remove `.detach()` from `VFE_dynamic` returns |

### Architecture: FEP, Not EM

The VFE transformer does **NOT** use traditional E-M steps. It implements **Free Energy Principle (FEP)** variational inference:

```
Algorithm: Iterative VFE Minimization with Fixed Priors
─────────────────────────────────────────────────────────
For each forward pass:
  1. Initialize beliefs from embeddings: μ = embed(tokens), σ = σ_init
  2. For n_iterations VFE steps:
     a. Recompute attention β from current beliefs (dynamic-β)
     b. Compute VFE gradients: ∂F/∂μ, ∂F/∂σ
     c. Apply Fisher preconditioning (natural gradient)
     d. Update beliefs: μ ← μ - lr · ∇_nat F
  3. Return final beliefs μ_final, σ_final
  4. Backprop through μ_final updates embeddings (learning)
```

**Key insight**: This is principled FEP with two timescales:
- **Fast (perception)**: VFE iterations minimize F w.r.t. beliefs q(z)
- **Slow (learning)**: Backprop minimizes F w.r.t. generative model θ (embeddings)

The M-step option (`m_step_interval > 0`) is **disabled by default** and is experimental online prior adaptation, not core FEP.

### Known Issues (Not Yet Fixed)

| Issue | Location | Status |
|-------|----------|--------|
| Position encoding disabled | `model.py:335` (commented out) | Needs decision |
| Double `token_embed()` call | `model.py:433,566` | Potential redundancy |
| Observation gradient uses `no_grad()` | `variational_ffn.py` | By design (observation is fixed) |

### Additional Bug Fixed

| Commit | Bug | Impact | Fix |
|--------|-----|--------|-----|
| (pending) | KL in VFE gradient used only Mahalanobis term | Softmax coupling gradient was incorrect | Added full KL with trace + logdet terms |

The VFE gradient computation in `compute_vfe_gradients_gpu()` was computing KL values for the softmax coupling term using only the Mahalanobis (quadratic) term:
```python
# BEFORE (incomplete):
kl_values = 0.5 * δμᵀ Σ⁻¹ δμ

# AFTER (full KL):
kl_values = 0.5 * (tr(Σ_p⁻¹ Σ_q) + δμᵀ Σ_p⁻¹ δμ - K + log|Σ_p| - log|Σ_q|)
```

### Files Modified

- `transformer/data.py` - Vocab mapping, fallback URLs, embedded sample cleanup
- `transformer/train.py` - `pad_token_id` parameter for loss masking
- `transformer/standard_transformer.py` - `pad_token_id` in forward
- `transformer/hamiltonian_ffn.py` - `pad_token_id` in potential energy
- `transformer/variational_ffn.py` - **CRITICAL**: Removed `.detach()` from VFE outputs, fixed full KL computation

### Recommended Next Actions

1. **Run training experiments** to verify fixes improve learning
2. **Re-enable position encoding** if sequence order matters
3. **Profile VFE iteration count** - may be able to reduce `n_iterations` for speed
4. **Add gradient norm logging** to monitor VFE dynamics during training

---

## Pure FEP Transformer (Dec 2025)

A new module `transformer/pure_fep_transformer.py` implements a transformer that learns **entirely through VFE minimization**, without backpropagation or external optimizers (Adam, SGD, etc.).

### Core Dynamics

Both beliefs and priors evolve via gradient descent on the Variational Free Energy:

```
dq/dt = -η_q · Σ_q · ∂F/∂μ_q     (fast timescale - perception)
dp/dt = -η_p · ∂F/∂μ_p           (slow timescale - learning)
```

Where the full VFE is:

```
F = α·KL(q||p)                              [Self-coupling]
  + λ_β·Σ_ij β_ij·KL(q_i||Ω_ij·q_j)         [Belief alignment]
  + λ_γ·Σ_ij γ_ij·KL(p_i||Ω_ij·p_j)         [Prior alignment]
  + Σ_d decay^d·KL(p||h^d)                   [Ouroboros Tower]
  + E[-log p(y|z)]                           [Observation likelihood]
```

### Prior Gradient Components

The prior gradient `∂F/∂μ_p` includes three terms:

1. **Self-coupling**: `α·(μ_p - μ_q)/σ_p²` — pulls priors toward beliefs
2. **Prior alignment**: `λ_γ·Σ_j(μ_p - Ω_ij·μ_p[j])/σ_j` — inter-position consistency
3. **Hyperprior (Ouroboros)**: `Σ_d decay^d·(μ_p - μ_h^d)/σ_h^d` — alignment with ancestors

### Hierarchical Prior Flow

Priors also receive **top-down flow** from parent layer beliefs:

```
p_child^(ζ) ← EMA(p_child^(ζ), Ω · q_parent^(ζ+1))
```

This is NOT gradient descent — it's direct assignment of parent beliefs (transported via gauge) as child priors.

### Architecture

```
PureFEPTransformer
├── PriorBank (unified embedding + output via token priors)
├── GaugePositionEncoder (position in φ ∈ so(3))
├── PureFEPLayer[0] (scale ζ=0)
│   ├── Beliefs q_i = N(μ_q, Σ_q)
│   ├── Priors p_i = N(μ_p, Σ_p)  ← from parent + gradient descent
│   ├── Hyperpriors h_i^d         ← Ouroboros Tower (grandparent, great-grandparent, ...)
│   └── Gauge frames φ_i
├── PureFEPLayer[1] (scale ζ=1)
│   └── ...
└── Dynamic layer spawning/merging based on VFE pressure
```

### Advanced Features

| Feature | Config Flag | Description |
|---------|-------------|-------------|
| **Prior Coupling** | `prior_coupling_enabled` | λ_γ term for prior-prior KL alignment |
| **Gradient Prior Updates** | `gradient_prior_updates` | dp/dt = -∂F/∂p instead of EMA |
| **Ouroboros Tower** | `enable_ouroboros_tower` | Non-Markovian hyperpriors from ALL ancestors |
| **Dynamic Layers** | `dynamic_layers_enabled` | Spawn/merge layers based on VFE gradient |
| **Exact Covariance Transport** | `exact_covariance_transport` | Σ_t = Ω·Σ·Ω^T (vs approximate) |
| **Multi-Irrep** | `use_multi_irrep` | Block-diagonal SO(3) generators |

### Ouroboros Tower (Non-Markovian Memory)

Instead of just parent → child prior flow (Markovian), collect hyperpriors from ALL ancestors:

```
p_i^(ζ)     ← q^(ζ+1)      parent (immediate prior)
h_i^(ζ,0)   ← q^(ζ+2)      grandparent (1st hyperprior)
h_i^(ζ,1)   ← q^(ζ+3)      great-grandparent (2nd hyperprior)
```

Each hyperprior contributes with decaying weight: `F += Σ_d decay^d · KL(p || h^d)`

This creates **long-range memory** where top-layer abstract beliefs directly influence bottom layers.

### Dynamic Layer Emergence

Layers can spawn or merge based on VFE pressure:

- **SPAWN**: When VFE gradient norm > `layer_spawn_threshold`
  - New layer inserted with priors interpolated from neighbors
- **MERGE**: When adjacent layers have >0.99 belief similarity
  - Redundant layers combined to reduce computation

### Usage

```python
from transformer.pure_fep_transformer import PureFEPConfig, PureFEPTransformer

config = PureFEPConfig(
    embed_dim=127,              # Must be ODD for SO(3) irreps
    num_layers=3,
    vocab_size=10000,
    # Dynamics
    mu_lr=0.1,                  # Belief learning rate
    prior_lr=0.01,              # Prior learning rate (slower!)
    # Advanced features
    gradient_prior_updates=True,     # Full dp/dt = -∂F/∂p
    prior_coupling_enabled=True,     # Prior alignment term
    enable_ouroboros_tower=True,     # Non-Markovian hyperpriors
    tower_max_depth=3,
    tower_decay=0.3,
    dynamic_layers_enabled=True,     # Adaptive architecture
    max_layers=8,
)

model = PureFEPTransformer(config)
metrics = model.train_step(input_ids, targets, n_vfe_steps=20)
```

### Embedding Modes

| Mode | Description |
|------|-------------|
| `prior_bank` | **Principled**: Token priors serve as both embedding AND output |
| `learned` | Standard nn.Embedding (ad hoc but fast) |
| `hybrid` | Learned embedding + PriorBank output |

### Output Modes

| Mode | Description |
|------|-------------|
| `kl_to_prior` | **Principled**: p(y\|q) ∝ exp(-KL(q\|\|π_y)/τ) |
| `linear` | Standard W·μ projection (ad hoc) |

### Position Modes

| Mode | Description |
|------|-------------|
| `gauge_frame` | **Principled**: Position encoded in φ ∈ so(3) — affects transport! |
| `sinusoidal_mu` | Standard sinusoidal added to μ |

### The Full Learning Loop

```
Forward Pass (perception):
  1. Initialize beliefs from token priors: q ← π_token
  2. For n_vfe_steps:
     - Compute attention β_ij = softmax(-KL_ij/κ)
     - Compute VFE gradients ∂F/∂μ_q, ∂F/∂σ_q
     - Natural gradient descent: μ ← μ - η·Σ·∇F
  3. Output via KL to token priors: logits = -KL(q||π_v)/τ

Backward Pass (learning):
  1. Collect hyperpriors from ancestors (Ouroboros Tower)
  2. Top-down prior flow: p_child ← Ω·q_parent
  3. Gradient-based prior update: p ← p - η_p·∂F/∂p
  4. Check dynamic emergence conditions (spawn/merge layers)
```

This implements **predictive coding** in the FEP sense with proper two-timescale dynamics!

---

## References

- PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
- torch.compile: https://pytorch.org/docs/stable/torch.compiler.html
- Symplectic Integrators: Hairer, Lubich, Wanner - "Geometric Numerical Integration"
- Information Geometry: Amari - "Information Geometry and Its Applications"
- Free Energy Principle: Friston - "The free-energy principle: a unified brain theory?"
