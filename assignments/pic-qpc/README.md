# Tree-PIC → QPC Implementation

**Programming Exam Submission**  
**Student: Asjad Afnan**  
**Course: Advanced Machine Learning**

## Overview

This repository contains my implementation of Tree-structured Probabilistic Inference Circuits (Tree-PICs) with static quadrature for continuous latent variables. The implementation includes both analytic Linear-Gaussian conditionals and neural Energy-Based Models.

## Implementation Details

### Core Components

- **Tree Structure**: Hierarchical latent variable models with smoothness and decomposability properties
- **Quadrature Methods**: Gauss-Legendre, Trapezoid, and Uniform integration for continuous variables
- **Conditional Distributions**: 
  - Linear-Gaussian (analytic, closed-form)
  - Neural Energy-Based (learnable, with Fourier features)
- **Circuit Compilation**: Bottom-up Tree-PIC to QPC compilation algorithm

### Key Features

- Stable numerical computations with log-space operations
- Comprehensive testing (64 tests passing)
- Mathematical validation against ground truth
- Reproducible examples for synthetic and UCI datasets

## Quick Start

```python
import torch
from pic import LatentTree, TreePIC, QPC, LinearGaussian, GaussianLeaf, Quadrature

# Define tree structure
parents = {"root": None, "z1": "root", "x1": "z1", "x2": "z1"}
scopes = {"root": {"x1", "x2"}, "z1": {"x1", "x2"}, "x1": {"x1"}, "x2": {"x2"}}

tree = LatentTree.from_parents(parents, scopes)

# Create conditionals and leaves
A = torch.tensor([[1.0]])
b = torch.tensor([0.0])
Sigma = torch.tensor([[1.0]])

conditionals = {"z1": LinearGaussian("z1", A, b, Sigma)}
leaves = {
    "x1": GaussianLeaf("x1", mu=0.0, sigma=1.0),
    "x2": GaussianLeaf("x2", mu=0.0, sigma=1.0)
}

# Compile to QPC
tree_pic = TreePIC(tree, conditionals, leaves)
quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 32)
qpc = tree_pic.compile_to_qpc(quadrature)

# Inference
x = torch.randn(10, 2)
log_probs = qpc.log_prob(x)
```

## Examples

```bash
# Run tests
python -m pytest tests/ -v

# Synthetic data training
python examples/train_synth.py

# UCI dataset training
python examples/train_uci.py

# Analytic validation
python examples/analytic_sanity.py
```

## Project Structure

```
pic/
├── structures.py      # Tree structure and validation
├── quadrature.py      # Numerical integration methods
├── nodes.py          # Circuit node abstractions
├── leaves.py         # Leaf node distributions
├── conditionals.py   # Conditional distributions
├── compile.py        # Tree-PIC to QPC compilation
├── evaluate.py       # Inference operations
└── utils.py          # Utility functions

tests/                # Test suite
examples/             # Usage examples
docs/                 # Documentation
```

## Dependencies

- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.0.0

## Installation

```bash
pip install torch numpy scipy matplotlib seaborn pandas scikit-learn
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- `docs/design.md` - Implementation architecture
- `docs/math.md` - Mathematical background
- `docs/experiments.md` - Experimental results

## Implementation Notes

This implementation focuses on:
1. **Numerical stability** - All computations done in log-space
2. **Modular design** - Clean abstractions for extensibility
3. **Comprehensive testing** - Unit tests, integration tests, and edge cases
4. **Mathematical correctness** - Validation against known analytic solutions

The code demonstrates advanced probabilistic modeling concepts including hierarchical latent variable models, numerical integration methods, and the integration of neural networks with probabilistic inference.
