#!/usr/bin/env python3
"""Analytic sanity check for Tree-PIC → QPC library.

This script demonstrates the library with analytic examples and visualizations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pic import (
    LatentTree, Quadrature, TreePIC, QPC,
    LinearGaussian, NeuralEnergyConditional,
    GaussianLeaf
)
from pic.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    """Run the analytic sanity check."""
    print("Tree-PIC → QPC: Analytic Sanity Check")
    print("=" * 50)
    
    # 1. Simple Linear-Gaussian Example
    print("\n1. Creating simple linear-Gaussian model...")
    
    # Define tree structure
    parents = {"root": None, "z1": "root", "x1": "z1", "x2": "z1"}
    scopes = {
        "root": {"z1", "x1", "x2"},
        "z1": {"z1", "x1", "x2"},
        "x1": {"x1"},
        "x2": {"x2"}
    }
    
    tree = LatentTree.from_parents(parents, scopes)
    print(f"Tree structure: {tree.spec.nodes}")
    print(f"Leaf nodes: {tree.spec.leaf_nodes}")
    
    # Create conditionals and leaves
    A = torch.tensor([[1.0]])
    b = torch.tensor([0.0])
    Sigma = torch.tensor([[1.0]])
    
    conditionals = {
        "z1": LinearGaussian("z1", A, b, Sigma)
    }
    
    leaves = {
        "x1": GaussianLeaf("x1", mu=0.0, sigma=1.0),
        "x2": GaussianLeaf("x2", mu=0.0, sigma=1.0)
    }
    
    tree_pic = TreePIC(tree, conditionals, leaves)
    print("TreePIC created successfully!")
    
    # 2. Compile to QPC
    print("\n2. Compiling to QPC...")
    quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 32)
    qpc = tree_pic.compile_to_qpc(quadrature)
    
    print(f"Quadrature points: {len(quadrature)}")
    print(f"Integration range: [{quadrature.grid.min():.2f}, {quadrature.grid.max():.2f}]")
    
    # 3. Test with data
    print("\n3. Testing with synthetic data...")
    x_test = torch.randn(100, 2)
    
    # Compute log probabilities using QPC
    log_probs_qpc = qpc.log_prob(x_test)
    
    # Compute analytic ground truth
    from torch.distributions import MultivariateNormal
    mu_gt = torch.zeros(2)
    cov_gt = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    dist_gt = MultivariateNormal(mu_gt, cov_gt)
    log_probs_gt = dist_gt.log_prob(x_test)
    
    print(f"QPC NLL: {-log_probs_qpc.mean():.4f}")
    print(f"Ground truth NLL: {-log_probs_gt.mean():.4f}")
    print(f"Mean absolute error: {torch.abs(log_probs_qpc - log_probs_gt).mean():.6f}")
    
    # 4. Convergence study
    print("\n4. Convergence study...")
    n_points_list = [8, 16, 32, 64, 128]
    errors = []
    
    for n_points in n_points_list:
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, n_points)
        qpc = tree_pic.compile_to_qpc(quadrature)
        log_probs_qpc = qpc.log_prob(x_test)
        error = torch.abs(log_probs_qpc - log_probs_gt).mean().item()
        errors.append(error)
        print(f"N={n_points:3d}: Error = {error:.6f}")
    
    # 5. Plot results
    print("\n5. Creating visualizations...")
    
    # Convergence plot
    plt.figure(figsize=(10, 6))
    plt.loglog(n_points_list, errors, 'o-', lw=2, markersize=8)
    plt.xlabel('Number of Quadrature Points')
    plt.ylabel('Mean Absolute Error')
    plt.title('Convergence of Quadrature Approximation')
    plt.grid(True, alpha=0.3)
    
    # Add theoretical convergence line
    x_theory = np.array(n_points_list)
    y_theory = 1.0 / x_theory**2
    y_theory = y_theory * errors[0] * n_points_list[0]**2
    plt.loglog(x_theory, y_theory, 'r--', lw=2, label='O(N^(-2))')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Numerical stability test
    print("\n6. Testing numerical stability...")
    extreme_values = torch.tensor([
        [0.0, 0.0],
        [10.0, 10.0],
        [-10.0, -10.0],
        [100.0, 100.0],
        [-100.0, -100.0]
    ])
    
    quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 32)
    qpc = tree_pic.compile_to_qpc(quadrature)
    
    print("Input values\t\tLog Probability\t\tFinite?")
    print("-" * 60)
    
    all_finite = True
    for x in extreme_values:
        log_prob = qpc.log_prob(x.unsqueeze(0))
        is_finite = torch.isfinite(log_prob).item()
        all_finite = all_finite and is_finite
        print(f"{x[0]:8.1f}, {x[1]:8.1f}\t\t{log_prob.item():12.6f}\t\t{is_finite}")
    
    print(f"\nAll log probabilities are finite: {'✓' if all_finite else '✗'}")
    
    print("\nAnalytic sanity check completed successfully!")

if __name__ == "__main__":
    main()
