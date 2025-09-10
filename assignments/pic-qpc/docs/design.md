# Design Guide

This document describes the architecture and implementation details of the Tree-PIC → QPC library.

## Architecture Overview

The library follows a two-stage compilation approach:

1. **TreePIC (Symbolic)**: High-level representation of the probabilistic model
2. **QPC (Materialized)**: Compiled circuit ready for inference

```
TreePIC (Symbolic) → Compilation → QPC (Materialized) → Inference
```

## Core Components

### 1. Tree Structures (`pic.structures`)

The tree structure defines the hierarchical relationship between variables and enforces key invariants:

#### Smoothness Property
Child scopes partition the parent scope:
```
∀ node n: ∪_{child c} scope(c) = scope(n)
```

#### Decomposability Property
No variable appears in multiple children:
```
∀ children c1, c2: scope(c1) ∩ scope(c2) = ∅
```

#### Implementation Details
- **TreeSpec**: Validates tree structure invariants
- **LatentTree**: Provides utility methods for tree traversal
- **Validation**: Automatic checking of smoothness and decomposability

### 2. Quadrature (`pic.quadrature`)

Numerical integration with stable operations:

#### Supported Methods
- **Gauss-Legendre**: High accuracy for smooth functions
- **Trapezoid**: Simple, robust for most cases
- **Uniform**: Midpoint rule, good for uniform distributions

#### Stable Operations
```python
def log_sum_exp_stable(log_values, weights=None, dim=-1):
    # Numerical stable log-sum-exp with optional weights
    max_val = torch.max(log_values, dim=dim, keepdim=True)[0]
    stable_log_values = log_values - max_val
    exp_values = torch.exp(stable_log_values)
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    log_sum = torch.log(sum_exp) + max_val
    return log_sum
```

#### Complexity
- **Time**: O(N) for N quadrature points
- **Space**: O(N) for storing grid and weights
- **Accuracy**: O(N^(-k)) where k depends on quadrature method

### 3. Circuit Nodes (`pic.nodes`)

Abstract circuit representation with caching:

#### Node Types
- **SumNode**: Mixture of components (log-sum-exp)
- **ProductNode**: Factorized distribution (sum of log-probs)
- **LeafNode**: Base distributions (Gaussian, Bernoulli)
- **IntegralNode**: Marginalization over latent variables

#### Caching Strategy
```python
class CircuitNode:
    def __init__(self):
        self._cached_log_prob = None
        self._cache_valid = False
    
    def clear_cache(self):
        self._cached_log_prob = None
        self._cache_valid = False
```

### 4. Conditional Distributions (`pic.conditionals`)

Two families of conditional distributions:

#### Linear-Gaussian
```python
p(z_i | z_parent) = N(z_i; A * z_parent + b, Σ)
```
- **Pros**: Analytic normalization, fast inference
- **Cons**: Limited expressiveness
- **Complexity**: O(D²) for D-dimensional variables

#### Neural Energy-Based
```python
p(z_i | z_parent) ∝ exp(-E(z_i, z_parent)) / Z(z_parent)
```
- **Pros**: High expressiveness, learnable
- **Cons**: Requires quadrature for normalization
- **Complexity**: O(N * D) for N quadrature points, D dimensions

#### Fourier Features
For neural energy models, we use Fourier feature embeddings:
```python
features = [sin(Bx), cos(Bx)]
```
This improves training stability and convergence.

### 5. Compilation (`pic.compile`)

Bottom-up compilation from TreePIC to QPC:

#### Algorithm
1. **Leaf nodes**: Direct mapping to circuit nodes
2. **Internal nodes**: Bottom-up construction
3. **Single child**: Wrap with IntegralNode for marginalization
4. **Multiple children**: Create ProductNode for factorization

#### Complexity
- **Time**: O(|V|) where |V| is number of nodes
- **Space**: O(|V|) for storing circuit structure

## Tensor Shapes and Broadcasting

### Input Format
- **Single sample**: `(n_features,)`
- **Batch**: `(batch_size, n_features)`
- **With quadrature**: `(batch_size, n_quadrature, n_features)`

### Broadcasting Rules
1. Quadrature dimension is added as needed
2. Parent context is expanded to match quadrature points
3. Log probabilities are computed element-wise
4. Integration reduces quadrature dimension

### Example
```python
# Input: (batch_size, n_features)
# Quadrature: (n_points,)
# Parent context: (batch_size, n_points, parent_dim)
# Child values: (batch_size, n_points, child_dim)
# Output: (batch_size,)
```

## Numerical Stability

### Log-Sum-Exp Stability
```python
def stable_log_sum_exp(log_values, dim=-1):
    max_val = torch.max(log_values, dim=dim, keepdim=True)[0]
    stable_log_values = log_values - max_val
    exp_values = torch.exp(stable_log_values)
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    log_sum = torch.log(sum_exp) + max_val
    return log_sum
```

### Gradient Stability
- **Gradient clipping**: Prevents exploding gradients
- **Weight decay**: Regularization for neural components
- **Safe operations**: log(1 + x) instead of log(x + 1)

### Boundary Handling
- **Extreme values**: Clipping to prevent overflow
- **Small values**: Epsilon addition for log operations
- **NaN detection**: Automatic checks and warnings

## Error Sources and Mitigation

### 1. Quadrature Error
**Source**: Finite number of quadrature points
**Mitigation**: 
- Use Gauss-Legendre for smooth functions
- Adaptive quadrature for complex regions
- Error estimation and refinement

### 2. Numerical Overflow
**Source**: Large energy values in neural models
**Mitigation**:
- Gradient clipping
- Energy normalization
- Stable log-sum-exp

### 3. Underflow
**Source**: Very small probabilities
**Mitigation**:
- Log-space computations
- Safe exponential operations
- Minimum probability thresholds

### 4. Grid Choice
**Source**: Inappropriate integration bounds
**Mitigation**:
- Adaptive grid based on data
- Multiple quadrature methods
- Grid refinement strategies

## Performance Considerations

### Memory Usage
- **Quadrature points**: O(N) per latent variable
- **Batch processing**: O(batch_size * N * D)
- **Caching**: O(batch_size * N) for intermediate results

### Computational Complexity
- **Forward pass**: O(batch_size * N * D * L) where L is circuit depth
- **Backward pass**: O(batch_size * N * D * L) with gradient computation
- **Memory**: O(batch_size * N * D) for activations

### Optimization Strategies
1. **Batch processing**: Vectorized operations
2. **Caching**: Reuse computed values
3. **Gradient accumulation**: For large batches
4. **Mixed precision**: FP16 for memory efficiency

## Device Support

### CPU Operations
- All operations work on CPU
- NumPy backend for quadrature weights
- Threading for batch processing

### GPU Operations
- PyTorch tensors automatically use GPU
- Quadrature weights moved to device
- Memory-efficient batch processing

### Multi-GPU
- DataParallel for model replication
- DistributedDataParallel for large models
- Gradient synchronization

## Serialization

### Model State
```python
state_dict = {
    "tree_spec": tree.spec,
    "conditionals": conditionals,
    "leaves": leaves,
    "parameters": parameters
}
```

### Quadrature State
```python
quadrature_state = {
    "grid": quadrature.grid,
    "weights": quadrature.weights,
    "method": quadrature.method
}
```

### Loading/Saving
- **Format**: PyTorch .pt files
- **Versioning**: Automatic compatibility checks
- **Migration**: Backward compatibility support

## Testing Strategy

### Unit Tests
- **Structures**: Tree validation and traversal
- **Quadrature**: Integration accuracy and stability
- **Conditionals**: Log probability and normalization
- **Nodes**: Forward and backward passes

### Integration Tests
- **End-to-end**: Full training pipeline
- **Reproducibility**: Fixed seed validation
- **Performance**: Memory and timing benchmarks

### Property-Based Tests
- **Random trees**: Structure invariants
- **Convergence**: Error vs quadrature points
- **Stability**: Gradient and numerical checks

## Future Extensions

### Planned Features
1. **Adaptive quadrature**: Automatic grid refinement
2. **Structured conditionals**: More complex dependencies
3. **Sampling**: Efficient sampling algorithms
4. **Distributed training**: Multi-GPU support

### Research Directions
1. **Hierarchical quadrature**: Multi-resolution integration
2. **Neural quadrature**: Learned integration points
3. **Structured inference**: Exploiting tree structure
4. **Online learning**: Incremental model updates
