"""Conditional distribution implementations for latent variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from .quadrature import Quadrature


class Conditional(ABC):
    """Abstract base class for conditional distributions p(z_i | z_parent)."""
    
    def __init__(self, name: str) -> None:
        """Initialize conditional distribution.
        
        Args:
            name: Conditional identifier
        """
        self.name = name
    
    @abstractmethod
    def log_prob(self, z: Tensor, z_parent: Tensor) -> Tensor:
        """Compute log probability p(z_i | z_parent).
        
        Args:
            z: Child variable values
            z_parent: Parent variable values
            
        Returns:
            Log probability tensor
        """
        pass
    
    @abstractmethod
    def log_norm(self, z_parent: Tensor, quadrature: Quadrature) -> Tensor:
        """Compute log normalization constant.
        
        Args:
            z_parent: Parent variable values
            quadrature: Quadrature for integration
            
        Returns:
            Log normalization tensor
        """
        pass


class LinearGaussian(Conditional):
    """Linear-Gaussian conditional distribution with analytic normalization.
    
    p(z_i | z_parent) = N(z_i; A * z_parent + b, Σ)
    """
    
    def __init__(
        self,
        name: str,
        A: Tensor,
        b: Tensor,
        Sigma: Tensor
    ) -> None:
        """Initialize linear-Gaussian conditional.
        
        Args:
            name: Conditional identifier
            A: Linear transformation matrix
            b: Bias vector
            Sigma: Covariance matrix
        """
        super().__init__(name)
        self.A = A
        self.b = b
        self.Sigma = Sigma
        
        # Validate dimensions
        if A.dim() != 2:
            raise ValueError(f"A must be 2D matrix, got shape {A.shape}")
        if b.dim() != 1:
            raise ValueError(f"b must be 1D vector, got shape {b.shape}")
        if Sigma.dim() != 2:
            raise ValueError(f"Sigma must be 2D matrix, got shape {Sigma.shape}")
        
        # Check compatibility
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"A rows {A.shape[0]} != b length {b.shape[0]}")
        if A.shape[0] != Sigma.shape[0] or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError(f"Sigma must be square, got shape {Sigma.shape}")
    
    def log_prob(self, z: Tensor, z_parent: Tensor) -> Tensor:
        """Compute log probability of linear-Gaussian conditional.
        
        Args:
            z: Child variable values
            z_parent: Parent variable values
            
        Returns:
            Log probability tensor
        """
        # Compute mean: μ = A * z_parent + b
        mean = torch.matmul(z_parent, self.A.t()) + self.b
        
        # Create normal distribution
        dist = Normal(loc=mean, scale=torch.sqrt(torch.diag(self.Sigma)))
        
        return dist.log_prob(z)
    
    def log_norm(self, z_parent: Tensor, quadrature: Quadrature) -> Tensor:
        """Compute log normalization constant (analytic).
        
        For linear-Gaussian, the normalization is constant and independent of z_parent.
        
        Args:
            z_parent: Parent variable values (not used for linear-Gaussian)
            quadrature: Quadrature (not used for linear-Gaussian)
            
        Returns:
            Log normalization tensor (constant)
        """
        # For linear-Gaussian, normalization is constant
        # log Z = -0.5 * log(2π) - 0.5 * log|Σ|
        log_det_Sigma = torch.logdet(self.Sigma)
        log_norm = -0.5 * torch.log(2 * torch.pi * torch.ones_like(log_det_Sigma)) - 0.5 * log_det_Sigma
        
        # Return constant for all z_parent values
        if z_parent.dim() > 0:
            return log_norm.expand(z_parent.shape[0])
        else:
            return log_norm


class FourierFeatures(nn.Module):
    """Fourier feature embedding for neural energy-based models."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma: float = 1.0
    ) -> None:
        """Initialize Fourier feature embedding.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            sigma: Frequency scaling parameter
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        # Random projection matrix
        self.register_buffer(
            'B',
            torch.randn(input_dim, output_dim // 2) * sigma
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier feature embedding.
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Fourier features of shape (..., output_dim)
        """
        # Project input
        proj = torch.matmul(x, self.B)
        
        # Apply sin and cos
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        
        # Concatenate
        features = torch.cat([sin_features, cos_features], dim=-1)
        
        return features


class NeuralEnergyConditional(Conditional):
    """Neural energy-based conditional distribution.
    
    p(z_i | z_parent) ∝ exp(-E(z_i, z_parent)) / Z(z_parent)
    where E is a neural network and Z is the normalization constant.
    """
    
    def __init__(
        self,
        name: str,
        parent_dim: int,
        child_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_fourier_features: bool = True,
        fourier_sigma: float = 1.0
    ) -> None:
        """Initialize neural energy-based conditional.
        
        Args:
            name: Conditional identifier
            parent_dim: Dimension of parent variable
            child_dim: Dimension of child variable
            hidden_dim: Hidden dimension of neural network
            n_layers: Number of layers in neural network
            use_fourier_features: Whether to use Fourier feature embedding
            fourier_sigma: Frequency scaling for Fourier features
        """
        super().__init__(name)
        self.parent_dim = parent_dim
        self.child_dim = child_dim
        self.hidden_dim = hidden_dim
        self.use_fourier_features = use_fourier_features
        
        # Input dimension
        input_dim = parent_dim + child_dim
        if use_fourier_features:
            self.fourier = FourierFeatures(input_dim, hidden_dim, fourier_sigma)
            nn_input_dim = hidden_dim
        else:
            self.fourier = None
            nn_input_dim = input_dim
        
        # Neural network for energy function
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(nn_input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (energy is scalar)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.energy_net = nn.Sequential(*layers)
    
    def energy(self, z: Tensor, z_parent: Tensor) -> Tensor:
        """Compute energy E(z, z_parent).
        
        Args:
            z: Child variable values
            z_parent: Parent variable values
            
        Returns:
            Energy tensor
        """
        # Concatenate inputs
        x = torch.cat([z_parent, z], dim=-1)
        
        # Apply Fourier features if enabled
        if self.use_fourier_features:
            x = self.fourier(x)
        
        # Compute energy
        energy = self.energy_net(x).squeeze(-1)
        
        return energy
    
    def log_prob(self, z: Tensor, z_parent: Tensor) -> Tensor:
        """Compute unnormalized log probability.
        
        Args:
            z: Child variable values
            z_parent: Parent variable values
            
        Returns:
            Unnormalized log probability tensor
        """
        # Unnormalized log probability: -E(z, z_parent)
        energy = self.energy(z, z_parent)
        return -energy
    
    def log_norm(self, z_parent: Tensor, quadrature: Quadrature) -> Tensor:
        """Compute log normalization constant via quadrature.
        
        Args:
            z_parent: Parent variable values
            quadrature: Quadrature for integration
            
        Returns:
            Log normalization tensor
        """
        # Get quadrature points
        z_points = quadrature.grid
        
        # Reshape for broadcasting
        if z_parent.dim() > 1:
            # z_parent: (batch, parent_dim)
            # z_points: (n_points,)
            # We want: (batch, n_points, child_dim)
            batch_size = z_parent.shape[0]
            n_points = len(z_points)
            
            # Expand z_parent to (batch, n_points, parent_dim)
            z_parent_expanded = z_parent.unsqueeze(1).expand(batch_size, n_points, -1)
            
            # Expand z_points to (batch, n_points, child_dim)
            z_points_expanded = z_points.unsqueeze(0).expand(batch_size, n_points)
            if self.child_dim > 1:
                z_points_expanded = z_points_expanded.unsqueeze(-1).expand(-1, -1, self.child_dim)
            else:
                z_points_expanded = z_points_expanded.unsqueeze(-1)  # (batch, n_points, 1)
        else:
            # Single parent value (z_parent is 1D: (parent_dim,))
            z_parent_expanded = z_parent.unsqueeze(0).expand(len(z_points), -1)  # (n_points, parent_dim)
            # z_points is (n_points,), need to make it (n_points, child_dim)
            z_points_expanded = z_points.unsqueeze(-1)  # (n_points, 1)
            if self.child_dim > 1:
                z_points_expanded = z_points_expanded.expand(-1, self.child_dim)
        
        # Compute unnormalized log probabilities
        log_probs = self.log_prob(z_points_expanded, z_parent_expanded)
        
        # Integrate using quadrature
        log_norm = quadrature.log_sum_exp(log_probs, dim=-1)
        
        return log_norm
    
    def normalized_log_prob(self, z: Tensor, z_parent: Tensor, quadrature: Quadrature) -> Tensor:
        """Compute normalized log probability.
        
        Args:
            z: Child variable values
            z_parent: Parent variable values
            quadrature: Quadrature for normalization
            
        Returns:
            Normalized log probability tensor
        """
        # Unnormalized log probability
        log_prob_unnorm = self.log_prob(z, z_parent)
        
        # Log normalization constant
        log_norm = self.log_norm(z_parent, quadrature)
        
        # Normalized log probability
        log_prob_norm = log_prob_unnorm - log_norm
        
        return log_prob_norm
