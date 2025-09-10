"""Quadrature methods for numerical integration in probabilistic inference circuits."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class Quadrature:
    """Quadrature grid and weights for numerical integration.
    
    This class provides quadrature points and weights for numerical integration
    in probabilistic inference circuits. It supports various quadrature methods
    and provides stable log-sum-exp operations.
    """
    
    def __init__(
        self,
        grid: Tensor,
        weights: Tensor,
        method: str = "custom"
    ) -> None:
        """Initialize quadrature with grid points and weights.
        
        Args:
            grid: Quadrature points of shape (n_points,)
            weights: Quadrature weights of shape (n_points,)
            method: Name of the quadrature method used
        """
        if grid.shape != weights.shape:
            raise ValueError(f"Grid shape {grid.shape} != weights shape {weights.shape}")
        
        self.grid = grid
        self.weights = weights
        self.method = method
        self.n_points = len(grid)
    
    @classmethod
    def trapezoid(
        cls,
        a: float,
        b: float,
        n_points: int,
        device: Optional[torch.device] = None
    ) -> Quadrature:
        """Create trapezoid quadrature rule.
        
        Args:
            a: Lower bound of integration interval
            b: Upper bound of integration interval
            n_points: Number of quadrature points
            device: Device to place tensors on
            
        Returns:
            Quadrature object with trapezoid rule
        """
        if n_points < 2:
            raise ValueError("Trapezoid rule requires at least 2 points")
        
        grid = torch.linspace(a, b, n_points, device=device)
        weights = torch.full((n_points,), (b - a) / (n_points - 1), device=device)
        weights[0] = weights[-1] = weights[0] / 2  # Endpoints get half weight
        
        return cls(grid, weights, method="trapezoid")
    
    @classmethod
    def gauss_legendre(
        cls,
        a: float,
        b: float,
        n_points: int,
        device: Optional[torch.device] = None
    ) -> Quadrature:
        """Create Gauss-Legendre quadrature rule.
        
        Args:
            a: Lower bound of integration interval
            b: Upper bound of integration interval
            n_points: Number of quadrature points
            device: Device to place tensors on
            
        Returns:
            Quadrature object with Gauss-Legendre rule
        """
        if n_points < 1:
            raise ValueError("Gauss-Legendre requires at least 1 point")
        
        # Get Gauss-Legendre points and weights on [-1, 1]
        import scipy.special as scipy_special
        
        points, weights_scipy = scipy_special.roots_legendre(n_points)
        
        # Transform to [a, b]
        grid = torch.tensor(
            (b - a) / 2 * points + (a + b) / 2,
            dtype=torch.float32,
            device=device
        )
        weights = torch.tensor(
            (b - a) / 2 * weights_scipy,
            dtype=torch.float32,
            device=device
        )
        
        return cls(grid, weights, method="gauss_legendre")
    
    @classmethod
    def uniform(
        cls,
        a: float,
        b: float,
        n_points: int,
        device: Optional[torch.device] = None
    ) -> Quadrature:
        """Create uniform quadrature rule (midpoint rule).
        
        Args:
            a: Lower bound of integration interval
            b: Upper bound of integration interval
            n_points: Number of quadrature points
            device: Device to place tensors on
            
        Returns:
            Quadrature object with uniform rule
        """
        if n_points < 1:
            raise ValueError("Uniform rule requires at least 1 point")
        
        # Midpoints of uniform intervals
        dx = (b - a) / n_points
        grid = torch.linspace(a + dx/2, b - dx/2, n_points, device=device)
        weights = torch.full((n_points,), dx, device=device)
        
        return cls(grid, weights, method="uniform")
    
    def log_sum_exp(
        self,
        log_values: Tensor,
        dim: int = -1
    ) -> Tensor:
        """Compute stable log-sum-exp with quadrature weights.
        
        Args:
            log_values: Log values of shape (..., n_points, ...)
            dim: Dimension along which to sum (should be the quadrature dimension)
            
        Returns:
            Log-sum-exp result with weights incorporated
        """
        # Ensure weights are on the same device and have compatible shape
        weights = self.weights.to(log_values.device)
        
        # Add log weights to log values
        log_weights = torch.log(weights)
        
        # Reshape log_weights to broadcast with log_values
        weight_shape = [1] * log_values.dim()
        weight_shape[dim] = -1
        log_weights = log_weights.view(weight_shape)
        
        # Add log weights
        log_weighted_values = log_values + log_weights
        
        # Compute stable log-sum-exp
        max_val = torch.max(log_weighted_values, dim=dim, keepdim=True)[0]
        stable_log_values = log_weighted_values - max_val
        exp_values = torch.exp(stable_log_values)
        sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
        log_sum = torch.log(sum_exp) + max_val
        
        return log_sum.squeeze(dim)
    
    def integrate(
        self,
        values: Tensor,
        dim: int = -1
    ) -> Tensor:
        """Integrate values using quadrature weights.
        
        Args:
            values: Values to integrate of shape (..., n_points, ...)
            dim: Dimension along which to integrate (should be the quadrature dimension)
            
        Returns:
            Integral result
        """
        weights = self.weights.to(values.device)
        
        # Reshape weights to broadcast with values
        weight_shape = [1] * values.dim()
        weight_shape[dim] = -1
        weights = weights.view(weight_shape)
        
        return torch.sum(values * weights, dim=dim)
    
    def to(self, device: torch.device) -> Quadrature:
        """Move quadrature to specified device.
        
        Args:
            device: Target device
            
        Returns:
            New Quadrature object on the target device
        """
        return Quadrature(
            self.grid.to(device),
            self.weights.to(device),
            self.method
        )
    
    def __len__(self) -> int:
        """Return number of quadrature points."""
        return self.n_points
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Quadrature(method={self.method}, n_points={self.n_points})"


def log_sum_exp_stable(
    log_values: Tensor,
    weights: Optional[Tensor] = None,
    dim: int = -1
) -> Tensor:
    """Compute stable log-sum-exp with optional weights.
    
    Args:
        log_values: Log values
        weights: Optional weights (if None, uniform weights are used)
        dim: Dimension along which to sum
        
    Returns:
        Log-sum-exp result
    """
    if weights is not None:
        # Add log weights
        log_weights = torch.log(weights)
        
        # Reshape log_weights to broadcast with log_values
        weight_shape = [1] * log_values.dim()
        weight_shape[dim] = -1
        log_weights = log_weights.view(weight_shape)
        
        log_values = log_values + log_weights
    
    # Compute stable log-sum-exp
    max_val = torch.max(log_values, dim=dim, keepdim=True)[0]
    stable_log_values = log_values - max_val
    exp_values = torch.exp(stable_log_values)
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    log_sum = torch.log(sum_exp) + max_val
    
    return log_sum.squeeze(dim)


def create_quadrature(
    method: Literal["trapezoid", "gauss_legendre", "uniform"],
    a: float,
    b: float,
    n_points: int,
    device: Optional[torch.device] = None
) -> Quadrature:
    """Factory function to create quadrature objects.
    
    Args:
        method: Quadrature method to use
        a: Lower bound of integration interval
        b: Upper bound of integration interval
        n_points: Number of quadrature points
        device: Device to place tensors on
        
    Returns:
        Quadrature object
    """
    if method == "trapezoid":
        return Quadrature.trapezoid(a, b, n_points, device)
    elif method == "gauss_legendre":
        return Quadrature.gauss_legendre(a, b, n_points, device)
    elif method == "uniform":
        return Quadrature.uniform(a, b, n_points, device)
    else:
        raise ValueError(f"Unknown quadrature method: {method}")
