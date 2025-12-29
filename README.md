# VFE-Sim GPU Refactor

A GPU-accelerated research framework implementing variational free energy (VFE) minimization for multi-agent systems and a novel gauge-theoretic transformer architecture.

## Overview

This project unifies three theoretical frameworks:
- **Variational Inference** - Free Energy Principle from theoretical neuroscience
- **Renormalization Group Theory** - Scale hierarchy and meta-agent emergence
- **Information Bottleneck Principle** - Optimal compression through gauge invariance

### Key Components

1. **Multi-Agent Simulation System** - Agents modeled as smooth sections of statistical manifolds, evolving via VFE minimization with SO(3) gauge symmetry

2. **Gauge-Theoretic Transformer** - A novel neural network architecture for language modeling that uses gauge theory and variational free energy instead of traditional backpropagation

3. **Differential Geometry Engine** - GPU-accelerated implementations of parallel transport, Fisher metrics, geodesic corrections, and Lie algebra operations

## Installation

**Requirements**: Python 3.7+ with CUDA-capable NVIDIA GPU (recommended)

```bash
git clone https://github.com/cdenn016/VFE-Sim-GPU-Refactor.git
cd VFE-Sim-GPU-Refactor
pip install -r requirements.txt
```

### Core Dependencies

- PyTorch (>=2.0.0) with CUDA support
- NumPy / SciPy
- Numba (JIT compilation)
- Matplotlib / Seaborn / Plotly (visualization)
- NetworkX (graph operations)

## Usage

### Multi-Agent Simulation

```bash
# Default simulation
python simulation_runner.py

# With specific presets
python simulation_runner.py --preset emergence   # Meta-agent emergence demo
python simulation_runner.py --preset ouroboros   # Ouroboros Tower (non-Markovian memory)
python simulation_runner.py --preset hamiltonian # Underdamped dynamics
```

### Transformer Training

```bash
# Standard training with VFE
python transformer/train.py

# Pure FEP training (no backpropagation)
python transformer/train_pure_fep.py

# Baseline comparison
python transformer/train_standard_baseline.py
```

### Visualization

```bash
# Attention pattern analysis
python visualize_attention_with_context.py --mode text --text "your sample text"
python visualize_attention_with_context.py --mode validation --dataset wikitext-2

# Belief space visualization
python visualize_belief_space.py

# Gauge semantics analysis
python analyze_gauge_semantics.py
```

## Project Structure

```
VFE-Sim-GPU-Refactor/
├── agent/              # Multi-agent system (beliefs, priors, gauge frames)
├── transformer/        # Gauge-theoretic transformer architecture
├── geometry/           # Differential geometry (manifolds, connections, Lie groups)
├── gradients/          # VFE gradient computation engine
├── math_utils/         # Mathematical primitives (SO(N) generators, transport)
├── meta/               # Meta-agent emergence and RG flow analysis
├── experiments/        # Research experiments
├── analysis/           # Data analysis pipeline
├── tests/              # Test suite
├── docs/               # Technical documentation
├── Transformer Paper/  # Research manuscript
├── simulation_runner.py    # Main simulation orchestrator
├── config.py               # System configuration
└── simulation_config.py    # Experiment presets
```

## Key Features

### Multi-Agent System
- **Belief-Prior Dynamics**: Gaussian beliefs q(z) and priors p(z) evolving via VFE minimization
- **SO(3) Gauge Symmetry**: Rotation-invariant computation with parallel transport
- **Meta-Agent Emergence**: Automatic detection of higher-scale structures via RG flow
- **Hamiltonian Dynamics**: Support for underdamped (conservative) and overdamped (dissipative) dynamics

### Gauge-Theoretic Transformer
- **Pure VFE Learning**: Two-timescale learning without backpropagation
  - Fast: Belief evolution (perception)
  - Slow: Prior evolution (learning)
- **KL-Based Attention**: `β_ij = softmax(-KL(q_i || Ω_ij[q_j])/κ)` replacing traditional softmax
- **SO(N) Gauge Groups**: Flexible symmetry structures (SO(3), SO(20), etc.)
- **Multi-Irrep Decomposition**: SO(3) irreducible representations (scalars, vectors, tensors)
- **Ouroboros Tower**: Non-Markovian hyperpriors from all ancestor layers

### Mathematical Features
- Fisher metric preconditioning (natural gradients)
- Geodesic corrections for Riemannian optimization
- Covariance field tracking with matrix-valued uncertainties
- GPU-friendly transport operator caching

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -v -k "transformer"
pytest tests/ -v -k "hamiltonian"

# With short traceback
pytest tests/ --tb=short
```

## Documentation

See the `docs/` directory for technical documentation:
- `PURE_FEP_TRANSFORMER_OVERVIEW.md` - Transformer architecture overview
- `PURE_FEP_TRANSFORMER_REVIEW.md` - Technical review and analysis
- `classical_models_as_limits.md` - Connection to classical models

## Research Findings

Recent experimental results:
- SO(20) gauge groups outperform SO(3) in language modeling (PPL 166 vs 341)
- VFE is mathematically equivalent to the Information Bottleneck principle
- Dynamic β implements input-dependent compression via KL-based attention
- Gauge invariance provides geometric compression as a symmetry-based prior
- RG fixed points represent optimal Information Bottleneck representations

## Configuration

Key configuration files:
- `config.py` - Agent, system, and training parameters
- `simulation_config.py` - Experiment presets and simulation settings

## License

This project is for research purposes.
