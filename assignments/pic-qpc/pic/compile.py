"""Compilation from Tree-PIC to QPC with materialization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .structures import LatentTree
from .quadrature import Quadrature
from .nodes import CircuitNode, SumNode, ProductNode, IntegralNode
from .leaves import GaussianLeaf, BernoulliLeaf
from .conditionals import Conditional, LinearGaussian, NeuralEnergyConditional


class TreePIC:
    """Symbolic Tree-PIC representation."""
    
    def __init__(
        self,
        tree: LatentTree,
        conditionals: Dict[str, Conditional],
        leaves: Dict[str, Union[GaussianLeaf, BernoulliLeaf]]
    ) -> None:
        """Initialize Tree-PIC with tree structure and components.
        
        Args:
            tree: Latent tree structure
            conditionals: Dictionary mapping node names to conditional distributions
            leaves: Dictionary mapping leaf node names to leaf distributions
        """
        self.tree = tree
        self.conditionals = conditionals
        self.leaves = leaves
        
        # Validate that all nodes have corresponding components
        self._validate_components()
    
    def _validate_components(self) -> None:
        """Validate that all nodes have corresponding components."""
        # Check that all non-root nodes have conditionals
        for node in self.tree.spec.nodes:
            if not self.tree.is_root(node):
                if node not in self.conditionals:
                    raise ValueError(f"Missing conditional for node {node}")
        
        # Check that all leaf nodes have leaf distributions
        for leaf in self.tree.spec.leaf_nodes:
            if leaf not in self.leaves:
                raise ValueError(f"Missing leaf distribution for node {leaf}")
    
    def compile_to_qpc(self, quadrature: Quadrature) -> QPC:
        """Compile Tree-PIC to QPC with materialization.
        
        Args:
            quadrature: Quadrature rule for integration
            
        Returns:
            Compiled QPC object
        """
        # Build circuit nodes bottom-up
        circuit_nodes = {}
        
        # Start with leaf nodes
        for leaf_name in self.tree.spec.leaf_nodes:
            leaf_dist = self.leaves[leaf_name]
            circuit_nodes[leaf_name] = leaf_dist
        
        # Build internal nodes bottom-up
        for node in self._get_bottom_up_order():
            if self.tree.is_leaf(node):
                continue
            
            children = self.tree.get_children(node)
            child_nodes = [circuit_nodes[child] for child in children]
            
            if len(children) == 1:
                # Single child - just wrap with conditional
                child_node = child_nodes[0]
                conditional = self.conditionals[children[0]]
                
                # Create integral node for marginalization
                circuit_nodes[node] = IntegralNode(
                    name=node,
                    child=child_node,
                    quadrature=quadrature
                )
            else:
                # Multiple children - create product node
                circuit_nodes[node] = ProductNode(
                    name=node,
                    children=child_nodes
                )
        
        # Get root node
        root_node = circuit_nodes[self.tree.spec.root_node]
        
        return QPC(root_node, quadrature)
    
    def _get_bottom_up_order(self) -> List[str]:
        """Get nodes in bottom-up order for compilation."""
        # Use topological sort (reverse of top-down order)
        visited = set()
        order = []
        
        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            
            # Visit children first
            for child in self.tree.get_children(node):
                visit(child)
            
            order.append(node)
        
        # Start from root
        visit(self.tree.spec.root_node)
        
        return order


class QPC:
    """Quadrature Probabilistic Circuit - materialized form of Tree-PIC."""
    
    def __init__(self, root_node: CircuitNode, quadrature: Quadrature) -> None:
        """Initialize QPC with root node and quadrature.
        
        Args:
            root_node: Root circuit node
            quadrature: Quadrature rule for integration
        """
        self.root_node = root_node
        self.quadrature = quadrature
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of observations.
        
        Args:
            x: Observation tensor
            
        Returns:
            Log probability tensor
        """
        # Clear any cached computations
        self.root_node.clear_cache()
        
        # Forward pass through the circuit
        return self.root_node.forward(x, self.quadrature)
    
    def marginal_log_prob(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute marginal log probability for masked variables.
        
        Args:
            x: Observation tensor
            mask: Boolean mask indicating which variables to marginalize
            
        Returns:
            Marginal log probability tensor
        """
        # For now, this is a placeholder implementation
        # In a full implementation, this would handle marginalization
        # by integrating over masked variables using quadrature
        
        if mask is None:
            return self.log_prob(x)
        
        # Simple implementation: set masked values to zero and integrate
        # This is a simplified version - full implementation would be more complex
        x_masked = x.clone()
        x_masked[mask] = 0.0
        
        return self.log_prob(x_masked)
    
    def condition(self, observations: Dict[str, Tensor]) -> QPC:
        """Condition the circuit on observations.
        
        Args:
            observations: Dictionary mapping variable names to observed values
            
        Returns:
            Conditioned QPC
        """
        # This is a placeholder implementation
        # In a full implementation, this would create a new circuit
        # with the observed variables fixed
        
        # For now, return self (no conditioning)
        return self
    
    def sample(self, n: int, device: Optional[torch.device] = None) -> Tensor:
        """Sample from the circuit (optional implementation).
        
        Args:
            n: Number of samples
            device: Device to place samples on
            
        Returns:
            Sample tensor
        """
        # This is a placeholder implementation
        # In a full implementation, this would implement sampling
        # using the circuit structure
        
        raise NotImplementedError("Sampling not yet implemented")
    
    def to(self, device: torch.device) -> QPC:
        """Move QPC to specified device.
        
        Args:
            device: Target device
            
        Returns:
            QPC on target device
        """
        # Move quadrature to device
        quadrature_device = self.quadrature.to(device)
        
        # Note: Moving the circuit nodes to device would require
        # implementing to() methods for all node types
        # For now, return a new QPC with moved quadrature
        
        return QPC(self.root_node, quadrature_device)
    
    def parameters(self) -> List[Tensor]:
        """Get all parameters of the circuit.
        
        Returns:
            List of parameter tensors
        """
        # This would collect parameters from all nodes
        # For now, return empty list
        return []
    
    def state_dict(self) -> Dict[str, Tensor]:
        """Get state dictionary for serialization.
        
        Returns:
            State dictionary
        """
        # This would collect state from all nodes
        # For now, return empty dict
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        # This would load state into all nodes
        # For now, do nothing
        pass


def compile_bayesian_network_to_tree_pic(
    parents: Dict[str, Optional[str]],
    scopes: Dict[str, set],
    conditionals: Dict[str, Conditional],
    leaves: Dict[str, Union[GaussianLeaf, BernoulliLeaf]]
) -> TreePIC:
    """Compile Bayesian network to Tree-PIC.
    
    Args:
        parents: Dictionary mapping each node to its parent (None for root)
        scopes: Dictionary mapping each node to its scope
        conditionals: Dictionary mapping node names to conditional distributions
        leaves: Dictionary mapping leaf node names to leaf distributions
        
    Returns:
        TreePIC object
    """
    # Create latent tree
    tree = LatentTree.from_parents(parents, scopes)
    
    # Create TreePIC
    return TreePIC(tree, conditionals, leaves)


def compile_ltm_to_tree_pic(
    ltm_spec: Dict,
    conditionals: Dict[str, Conditional],
    leaves: Dict[str, Union[GaussianLeaf, BernoulliLeaf]]
) -> TreePIC:
    """Compile Latent Tree Model to Tree-PIC.
    
    Args:
        ltm_spec: LTM specification dictionary
        conditionals: Dictionary mapping node names to conditional distributions
        leaves: Dictionary mapping leaf node names to leaf distributions
        
    Returns:
        TreePIC object
    """
    # Extract tree structure from LTM spec
    parents = ltm_spec["parents"]
    scopes = ltm_spec["scopes"]
    
    return compile_bayesian_network_to_tree_pic(parents, scopes, conditionals, leaves)
