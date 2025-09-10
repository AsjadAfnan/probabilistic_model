"""Leaf node implementations for different distribution types."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli, Normal

from .nodes import LeafNode


class GaussianLeaf(LeafNode):
    """Gaussian leaf node with learnable mean and standard deviation."""
    
    def __init__(
        self,
        name: str,
        mu_fn: Optional[Callable[[Tensor], Tensor]] = None,
        sigma_fn: Optional[Callable[[Tensor], Tensor]] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        min_sigma: float = 1e-6
    ) -> None:
        """Initialize Gaussian leaf node.
        
        Args:
            name: Node identifier
            mu_fn: Function to compute mean from parent values
            sigma_fn: Function to compute standard deviation from parent values
            mu: Fixed mean value (if mu_fn is None)
            sigma: Fixed standard deviation value (if sigma_fn is None)
            min_sigma: Minimum standard deviation for numerical stability
        """
        super().__init__(name)
        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn
        self.mu = mu
        self.sigma = sigma
        self.min_sigma = min_sigma
        
        # Validate parameters
        if mu_fn is None and mu is None:
            raise ValueError("Either mu_fn or mu must be provided")
        if sigma_fn is None and sigma is None:
            raise ValueError("Either sigma_fn or sigma must be provided")
    
    def _get_mu(self, x: Tensor) -> Tensor:
        """Get mean value(s).
        
        Args:
            x: Input tensor (may be parent values)
            
        Returns:
            Mean tensor
        """
        if self.mu_fn is not None:
            return self.mu_fn(x)
        else:
            return torch.full_like(x, self.mu)
    
    def _get_sigma(self, x: Tensor) -> Tensor:
        """Get standard deviation value(s).
        
        Args:
            x: Input tensor (may be parent values)
            
        Returns:
            Standard deviation tensor
        """
        if self.sigma_fn is not None:
            sigma = self.sigma_fn(x)
        else:
            sigma = torch.full_like(x, self.sigma)
        
        # Ensure minimum standard deviation for numerical stability
        return torch.clamp(sigma, min=self.min_sigma)
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of Gaussian distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            Log probability tensor
        """
        # For Gaussian leaves, x represents the observed values
        # We need to get mu and sigma from parent context if functions are provided
        # For now, assume x contains both observed values and parent context
        # This will be refined in the conditional implementation
        
        # Extract observed values (assuming last dimension)
        if x.dim() > 1:
            observed = x[..., -1]
            parent_context = x[..., :-1] if x.shape[-1] > 1 else x
        else:
            observed = x
            parent_context = x
        
        mu = self._get_mu(parent_context)
        sigma = self._get_sigma(parent_context)
        
        # Create normal distribution
        dist = Normal(loc=mu, scale=sigma)
        
        return dist.log_prob(observed)


class BernoulliLeaf(LeafNode):
    """Bernoulli leaf node with learnable logits."""
    
    def __init__(
        self,
        name: str,
        logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
        logits: Optional[float] = None
    ) -> None:
        """Initialize Bernoulli leaf node.
        
        Args:
            name: Node identifier
            logits_fn: Function to compute logits from parent values
            logits: Fixed logits value (if logits_fn is None)
        """
        super().__init__(name)
        self.logits_fn = logits_fn
        self.logits = logits
        
        # Validate parameters
        if logits_fn is None and logits is None:
            raise ValueError("Either logits_fn or logits must be provided")
    
    def _get_logits(self, x: Tensor) -> Tensor:
        """Get logits value(s).
        
        Args:
            x: Input tensor (may be parent values)
            
        Returns:
            Logits tensor
        """
        if self.logits_fn is not None:
            return self.logits_fn(x)
        else:
            return torch.full_like(x, self.logits)
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of Bernoulli distribution.
        
        Args:
            x: Input tensor (binary values)
            
        Returns:
            Log probability tensor
        """
        # Extract observed values and parent context
        if x.dim() > 1:
            observed = x[..., -1]
            parent_context = x[..., :-1] if x.shape[-1] > 1 else x
        else:
            observed = x
            parent_context = x
        
        logits = self._get_logits(parent_context)
        
        # Create Bernoulli distribution
        dist = Bernoulli(logits=logits)
        
        return dist.log_prob(observed)


class ConditionalGaussianLeaf(GaussianLeaf):
    """Gaussian leaf node that depends on parent latent variables."""
    
    def __init__(
        self,
        name: str,
        mu_fn: Callable[[Tensor], Tensor],
        sigma_fn: Callable[[Tensor], Tensor],
        min_sigma: float = 1e-6
    ) -> None:
        """Initialize conditional Gaussian leaf node.
        
        Args:
            name: Node identifier
            mu_fn: Function to compute mean from parent values
            sigma_fn: Function to compute standard deviation from parent values
            min_sigma: Minimum standard deviation for numerical stability
        """
        super().__init__(
            name=name,
            mu_fn=mu_fn,
            sigma_fn=sigma_fn,
            min_sigma=min_sigma
        )
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability given parent context.
        
        Args:
            x: Input tensor with parent context and observed values
            
        Returns:
            Log probability tensor
        """
        # x should contain [parent_context, observed_value]
        if x.dim() == 1:
            # Single sample: [parent_context, observed]
            parent_context = x[:-1]
            observed = x[-1:]
        else:
            # Batch: [batch, parent_context, observed]
            parent_context = x[..., :-1]
            observed = x[..., -1:]
        
        mu = self._get_mu(parent_context)
        sigma = self._get_sigma(parent_context)
        
        # Create normal distribution
        dist = Normal(loc=mu, scale=sigma)
        
        return dist.log_prob(observed)


class ConditionalBernoulliLeaf(BernoulliLeaf):
    """Bernoulli leaf node that depends on parent latent variables."""
    
    def __init__(
        self,
        name: str,
        logits_fn: Callable[[Tensor], Tensor]
    ) -> None:
        """Initialize conditional Bernoulli leaf node.
        
        Args:
            name: Node identifier
            logits_fn: Function to compute logits from parent values
        """
        super().__init__(name=name, logits_fn=logits_fn)
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability given parent context.
        
        Args:
            x: Input tensor with parent context and observed values
            
        Returns:
            Log probability tensor
        """
        # x should contain [parent_context, observed_value]
        if x.dim() == 1:
            # Single sample: [parent_context, observed]
            parent_context = x[:-1]
            observed = x[-1:]
        else:
            # Batch: [batch, parent_context, observed]
            parent_context = x[..., :-1]
            observed = x[..., -1:]
        
        logits = self._get_logits(parent_context)
        
        # Create Bernoulli distribution
        dist = Bernoulli(logits=logits)
        
        return dist.log_prob(observed)
