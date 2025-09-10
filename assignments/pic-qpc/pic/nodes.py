"""Core circuit node abstractions for probabilistic inference circuits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


class CircuitNode(ABC):
    """Abstract base class for all circuit nodes."""
    
    def __init__(self, name: str) -> None:
        """Initialize node with a name.
        
        Args:
            name: Node identifier
        """
        self.name = name
        self._cached_log_prob: Optional[Tensor] = None
        self._cache_valid = False
    
    @abstractmethod
    def forward(self, x: Tensor, quadrature: Optional[Quadrature] = None) -> Tensor:
        """Forward pass through the node.
        
        Args:
            x: Input tensor
            quadrature: Optional quadrature for integration
            
        Returns:
            Log probability tensor
        """
        pass
    
    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._cached_log_prob = None
        self._cache_valid = False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"


class SumNode(CircuitNode):
    """Sum node representing mixture of components."""
    
    def __init__(
        self,
        name: str,
        children: List[CircuitNode],
        weights: Optional[Tensor] = None
    ) -> None:
        """Initialize sum node.
        
        Args:
            name: Node identifier
            children: List of child nodes
            weights: Log weights for the mixture (if None, uniform weights)
        """
        super().__init__(name)
        self.children = children
        self.weights = weights
        
        if weights is not None:
            # Normalize weights
            log_weights = torch.log_softmax(weights, dim=0)
            self.log_weights = log_weights
        else:
            # Uniform weights
            n_children = len(children)
            self.log_weights = torch.full(
                (n_children,), -torch.log(torch.tensor(n_children, dtype=torch.float32))
            )
    
    def forward(self, x: Tensor, quadrature: Optional[Quadrature] = None) -> Tensor:
        """Forward pass: weighted sum of child log probabilities.
        
        Args:
            x: Input tensor
            quadrature: Not used for sum nodes
            
        Returns:
            Log probability tensor
        """
        if self._cache_valid and self._cached_log_prob is not None:
            return self._cached_log_prob
        
        # Get log probabilities from all children
        child_log_probs = []
        for child in self.children:
            child_log_prob = child.forward(x, quadrature)
            child_log_probs.append(child_log_prob)
        
        # Stack along new dimension
        stacked_log_probs = torch.stack(child_log_probs, dim=-1)
        
        # Add log weights
        log_weights = self.log_weights.to(stacked_log_probs.device)
        weighted_log_probs = stacked_log_probs + log_weights
        
        # Log-sum-exp to get mixture log probability
        from .quadrature import log_sum_exp_stable
        log_prob = log_sum_exp_stable(weighted_log_probs, dim=-1)
        
        # Cache result
        self._cached_log_prob = log_prob
        self._cache_valid = True
        
        return log_prob
    
    def clear_cache(self) -> None:
        """Clear cache for this node and all children."""
        super().clear_cache()
        for child in self.children:
            child.clear_cache()


class ProductNode(CircuitNode):
    """Product node representing factorized distribution."""
    
    def __init__(self, name: str, children: List[CircuitNode]) -> None:
        """Initialize product node.
        
        Args:
            name: Node identifier
            children: List of child nodes
        """
        super().__init__(name)
        self.children = children
    
    def forward(self, x: Tensor, quadrature: Optional[Quadrature] = None) -> Tensor:
        """Forward pass: sum of child log probabilities.
        
        Args:
            x: Input tensor
            quadrature: Not used for product nodes
            
        Returns:
            Log probability tensor
        """
        if self._cache_valid and self._cached_log_prob is not None:
            return self._cached_log_prob
        
        # Sum log probabilities from all children
        log_prob = torch.zeros_like(x[..., 0])  # Take first dimension for shape
        
        for child in self.children:
            child_log_prob = child.forward(x, quadrature)
            log_prob = log_prob + child_log_prob
        
        # Cache result
        self._cached_log_prob = log_prob
        self._cache_valid = True
        
        return log_prob
    
    def clear_cache(self) -> None:
        """Clear cache for this node and all children."""
        super().clear_cache()
        for child in self.children:
            child.clear_cache()


class LeafNode(CircuitNode):
    """Abstract base class for leaf nodes."""
    
    def __init__(self, name: str) -> None:
        """Initialize leaf node.
        
        Args:
            name: Node identifier
        """
        super().__init__(name)
    
    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability for given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Log probability tensor
        """
        pass
    
    def forward(self, x: Tensor, quadrature: Optional[Quadrature] = None) -> Tensor:
        """Forward pass: compute log probability.
        
        Args:
            x: Input tensor
            quadrature: Not used for leaf nodes
            
        Returns:
            Log probability tensor
        """
        if self._cache_valid and self._cached_log_prob is not None:
            return self._cached_log_prob
        
        log_prob = self.log_prob(x)
        
        # Cache result
        self._cached_log_prob = log_prob
        self._cache_valid = True
        
        return log_prob


class IntegralNode(CircuitNode):
    """Integral node for marginalization over latent variables."""
    
    def __init__(
        self,
        name: str,
        child: CircuitNode,
        quadrature: Quadrature
    ) -> None:
        """Initialize integral node.
        
        Args:
            name: Node identifier
            child: Child node to integrate over
            quadrature: Quadrature rule for integration
        """
        super().__init__(name)
        self.child = child
        self.quadrature = quadrature
    
    def forward(self, x: Tensor, quadrature: Optional[Quadrature] = None) -> Tensor:
        """Forward pass: integrate child over quadrature points.
        
        Args:
            x: Input tensor
            quadrature: Override quadrature (if None, use node's quadrature)
            
        Returns:
            Log probability tensor after integration
        """
        if self._cache_valid and self._cached_log_prob is not None:
            return self._cached_log_prob
        
        # Use node's quadrature if none provided
        if quadrature is None:
            quadrature = self.quadrature
        
        # Get quadrature points
        z_points = quadrature.grid
        
        # Reshape for broadcasting
        if x.dim() > 0:
            # Add quadrature dimension
            z_points = z_points.view(1, -1).expand(x.shape[0], -1)
        
        # Compute log probabilities at quadrature points
        # This requires the child to handle the quadrature dimension properly
        log_probs = self.child.forward(z_points, quadrature)
        
        # Integrate using quadrature weights
        log_prob = quadrature.log_sum_exp(log_probs, dim=-1)
        
        # Cache result
        self._cached_log_prob = log_prob
        self._cache_valid = True
        
        return log_prob
    
    def clear_cache(self) -> None:
        """Clear cache for this node and child."""
        super().clear_cache()
        self.child.clear_cache()


# Import here to avoid circular imports
from .quadrature import Quadrature
