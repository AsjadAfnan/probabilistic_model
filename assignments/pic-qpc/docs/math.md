# Mathematical Background

This document provides the theoretical foundations and mathematical derivations for the Tree-PIC → QPC library.

## Probabilistic Inference Circuits (PICs)

### Definition

A Probabilistic Inference Circuit (PIC) is a directed acyclic graph (DAG) where:
- **Nodes** represent probability distributions
- **Edges** represent conditional dependencies
- **Leaves** represent observed variables
- **Internal nodes** represent latent variables

### Tree-PIC Structure

A Tree-PIC is a PIC with a tree structure where:
- Each node has at most one parent
- The graph is connected and acyclic
- Variables are organized in a hierarchical structure

## Mathematical Formulation

### Joint Distribution

For a Tree-PIC with variables $V = \{v_1, \ldots, v_n\}$, the joint distribution is:

$$p(x, z) = \prod_{i=1}^n p(v_i | \text{pa}(v_i))$$

where $\text{pa}(v_i)$ denotes the parents of variable $v_i$.

### Marginalization

To compute the marginal distribution over observed variables $x$:

$$p(x) = \int p(x, z) \, dz$$

This requires integration over all latent variables $z$.

## Static Quadrature Approach

### From Integration to Summation

Instead of continuous integration, we use static quadrature:

$$\int p(x, z) \, dz \approx \sum_{k=1}^N w_k p(x, z_k)$$

where:
- $\{z_k\}_{k=1}^N$ are quadrature points
- $\{w_k\}_{k=1}^N$ are quadrature weights
- $N$ is the number of quadrature points

### Log-Space Computation

For numerical stability, we work in log-space:

$$\log p(x) = \log \sum_{k=1}^N w_k p(x, z_k)$$

Using the log-sum-exp trick:

$$\log p(x) = \max_k \log(w_k p(x, z_k)) + \log \sum_{k=1}^N \exp(\log(w_k p(x, z_k)) - \max_k \log(w_k p(x, z_k)))$$

## Conditional Distributions

### Linear-Gaussian Conditionals

For linear-Gaussian conditionals:

$$p(z_i | z_{\text{pa}}) = \mathcal{N}(z_i; A \cdot z_{\text{pa}} + b, \Sigma)$$

where:
- $A$ is the linear transformation matrix
- $b$ is the bias vector
- $\Sigma$ is the covariance matrix

#### Analytic Normalization

The normalization constant is:

$$Z(z_{\text{pa}}) = \int \mathcal{N}(z_i; A \cdot z_{\text{pa}} + b, \Sigma) \, dz_i = 1$$

This is independent of $z_{\text{pa}}$, making it computationally efficient.

### Neural Energy-Based Conditionals

For neural energy-based conditionals:

$$p(z_i | z_{\text{pa}}) = \frac{\exp(-E(z_i, z_{\text{pa}}))}{Z(z_{\text{pa}})}$$

where:
- $E(z_i, z_{\text{pa}})$ is the energy function (neural network)
- $Z(z_{\text{pa}})$ is the normalization constant

#### Normalization via Quadrature

The normalization constant is computed using quadrature:

$$Z(z_{\text{pa}}) = \int \exp(-E(z_i, z_{\text{pa}})) \, dz_i \approx \sum_{k=1}^N w_k \exp(-E(z_k, z_{\text{pa}}))$$

In log-space:

$$\log Z(z_{\text{pa}}) = \log \sum_{k=1}^N w_k \exp(-E(z_k, z_{\text{pa}}))$$

## Quadrature Methods

### Gauss-Legendre Quadrature

Gauss-Legendre quadrature provides high accuracy for smooth functions:

$$\int_{-1}^1 f(x) \, dx \approx \sum_{k=1}^N w_k f(x_k)$$

where $\{x_k, w_k\}$ are the Gauss-Legendre points and weights.

#### Transformation to Arbitrary Interval

For integration over $[a, b]$:

$$\int_a^b f(x) \, dx = \frac{b-a}{2} \int_{-1}^1 f\left(\frac{b-a}{2} t + \frac{a+b}{2}\right) \, dt$$

### Trapezoid Rule

Simple but robust quadrature:

$$\int_a^b f(x) \, dx \approx \frac{h}{2} \left(f(a) + 2\sum_{k=1}^{N-1} f(x_k) + f(b)\right)$$

where $h = (b-a)/(N-1)$ and $x_k = a + kh$.

### Uniform Quadrature

Midpoint rule for uniform sampling:

$$\int_a^b f(x) \, dx \approx h \sum_{k=1}^N f\left(a + \left(k-\frac{1}{2}\right)h\right)$$

where $h = (b-a)/N$.

## Fourier Features

### Motivation

Neural networks struggle to learn high-frequency functions. Fourier features help by:

1. **Frequency encoding**: Explicitly encoding spatial frequencies
2. **Training stability**: Improving gradient flow
3. **Convergence**: Faster convergence to high-frequency patterns

### Implementation

Given input $x \in \mathbb{R}^d$, Fourier features are:

$$\phi(x) = [\sin(Bx), \cos(Bx)]$$

where $B \in \mathbb{R}^{d \times m/2}$ is a random projection matrix.

#### Random Projection

The matrix $B$ is typically sampled as:

$$B_{ij} \sim \mathcal{N}(0, \sigma^2)$$

where $\sigma$ controls the frequency spectrum.

## Circuit Compilation

### Bottom-Up Construction

The compilation process follows a bottom-up approach:

1. **Leaf nodes**: Direct mapping to circuit nodes
2. **Internal nodes**: Recursive construction
3. **Single child**: Wrap with IntegralNode
4. **Multiple children**: Create ProductNode

### Mathematical Justification

For a node $n$ with children $\{c_1, \ldots, c_k\}$:

#### Single Child
$$p(n) = \int p(c_1) \, dz_{c_1}$$

#### Multiple Children
$$p(n) = \prod_{i=1}^k p(c_i)$$

This preserves the tree structure and enables efficient inference.

## Error Analysis

### Quadrature Error

The quadrature error depends on the method:

#### Gauss-Legendre
For smooth functions, the error is:
$$|E_N| \leq \frac{(b-a)^{2N+1} (N!)^4}{(2N+1)((2N)!)^3} \max_{x \in [a,b]} |f^{(2N)}(x)|$$

#### Trapezoid Rule
For twice differentiable functions:
$$|E_N| \leq \frac{(b-a)^3}{12N^2} \max_{x \in [a,b]} |f''(x)|$$

### Numerical Stability

#### Log-Sum-Exp Stability

The log-sum-exp operation is numerically stable:

$$\log \sum_{i=1}^n \exp(x_i) = \max_i x_i + \log \sum_{i=1}^n \exp(x_i - \max_i x_i)$$

This prevents overflow and underflow.

#### Gradient Stability

For neural energy models, gradients are:

$$\nabla_\theta \log p(x) = \nabla_\theta \log \sum_k w_k p(x, z_k)$$

Using the chain rule:

$$\nabla_\theta \log p(x) = \frac{\sum_k w_k \nabla_\theta p(x, z_k)}{\sum_k w_k p(x, z_k)}$$

## Convergence Analysis

### Monotone Convergence

As the number of quadrature points increases, the error decreases monotonically:

**Theorem**: For smooth functions and appropriate quadrature methods:
$$|E_{N_1}| \geq |E_{N_2}| \quad \text{if} \quad N_1 < N_2$$

### Rate of Convergence

The convergence rate depends on the quadrature method:

- **Gauss-Legendre**: Exponential convergence for analytic functions
- **Trapezoid**: Quadratic convergence for smooth functions
- **Uniform**: Linear convergence for continuous functions

## Complexity Analysis

### Time Complexity

#### Forward Pass
- **Single sample**: $O(N \cdot D \cdot L)$
- **Batch processing**: $O(B \cdot N \cdot D \cdot L)$

where:
- $N$ = number of quadrature points
- $D$ = dimension of latent variables
- $L$ = depth of the circuit
- $B$ = batch size

#### Backward Pass
- **Gradient computation**: $O(B \cdot N \cdot D \cdot L)$
- **Parameter updates**: $O(P)$ where $P$ is number of parameters

### Space Complexity

#### Memory Usage
- **Quadrature storage**: $O(N \cdot D)$
- **Intermediate results**: $O(B \cdot N \cdot D)$
- **Gradients**: $O(P)$

## Deterministic MAP Inference

### Most Probable Explanation

For finding the most probable explanation (MAP):

$$\arg\max_z p(z | x) = \arg\max_z p(x, z)$$

This can be solved using gradient-based optimization:

$$\nabla_z \log p(x, z) = \nabla_z \sum_i \log p(v_i | \text{pa}(v_i))$$

### Optimization Algorithm

1. **Initialize**: $z^{(0)} = \text{random}$
2. **Iterate**: $z^{(t+1)} = z^{(t)} + \alpha \nabla_z \log p(x, z^{(t)})$
3. **Converge**: When $\|\nabla_z \log p(x, z^{(t)})\| < \epsilon$

### Convergence Conditions

For unimodal distributions, MAP inference converges to the global optimum. For multimodal distributions, multiple restarts may be needed.

## Marginalization and Conditioning

### Marginalization

To compute marginal distributions:

$$p(x_S) = \int p(x) \, dx_{\bar{S}}$$

where $S$ is the subset of variables to keep and $\bar{S}$ is the complement.

### Conditioning

To condition on observations:

$$p(z | x_{\text{obs}}) = \frac{p(x_{\text{obs}}, z)}{p(x_{\text{obs}})}$$

This requires computing the marginal $p(x_{\text{obs}})$.

## Relationship to Other Methods

### Variational Inference

Tree-PIC can be viewed as a form of variational inference where:
- **Variational family**: Tree-structured distributions
- **Inference method**: Quadrature-based integration
- **Optimization**: Maximum likelihood or variational Bayes

### Sum-Product Networks (SPNs)

Tree-PIC is related to SPNs but with:
- **Continuous variables**: Instead of discrete
- **Quadrature integration**: Instead of exact summation
- **Neural conditionals**: Instead of tabular parameters

### Neural ODEs

The quadrature approach is similar to neural ODEs:
- **Discretization**: Quadrature points vs. ODE solver steps
- **Adaptive methods**: Future work on adaptive quadrature
- **Gradient flow**: Continuous optimization in both cases

## Future Mathematical Directions

### Adaptive Quadrature

Future work could implement adaptive quadrature:

1. **Error estimation**: Compute quadrature error
2. **Grid refinement**: Add points in high-error regions
3. **Convergence**: Stop when error is below threshold

### Hierarchical Quadrature

Multi-resolution quadrature for efficiency:

1. **Coarse grid**: Initial approximation
2. **Fine grid**: Refinement in important regions
3. **Adaptive selection**: Choose resolution based on importance

### Neural Quadrature

Learned quadrature points:

1. **Learned weights**: Optimize quadrature weights
2. **Learned points**: Optimize quadrature locations
3. **Task-specific**: Adapt to specific inference tasks
