"""Tree structure definitions and validation for latent variable models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor


@dataclass
class TreeSpec:
    """Specification for a latent tree structure.
    
    Attributes:
        nodes: List of node identifiers
        parents: Dictionary mapping each node to its parent (None for root)
        children: Dictionary mapping each node to its children
        scopes: Dictionary mapping each node to its scope (set of variables)
        leaf_nodes: Set of leaf node identifiers
        root_node: Root node identifier
    """
    
    nodes: List[str]
    parents: Dict[str, Optional[str]]
    children: Dict[str, List[str]]
    scopes: Dict[str, Set[str]]
    leaf_nodes: Set[str]
    root_node: str
    
    def __post_init__(self) -> None:
        """Validate tree structure invariants."""
        self._validate_tree_structure()
        self._validate_smoothness()
        self._validate_decomposability()
    
    def _validate_tree_structure(self) -> None:
        """Validate basic tree structure properties."""
        # All nodes should have a parent except root
        for node in self.nodes:
            if node == self.root_node:
                if self.parents[node] is not None:
                    raise ValueError(f"Root node {node} has parent {self.parents[node]}")
            else:
                if self.parents[node] is None:
                    raise ValueError(f"Non-root node {node} has no parent")
        
        # Children should be consistent with parents
        for node, parent in self.parents.items():
            if parent is not None:
                if node not in self.children[parent]:
                    raise ValueError(f"Node {node} not in children of {parent}")
        
        # Leaf nodes should have no children
        for leaf in self.leaf_nodes:
            if self.children[leaf]:
                raise ValueError(f"Leaf node {leaf} has children: {self.children[leaf]}")
    
    def _validate_smoothness(self) -> None:
        """Validate smoothness property: child scopes partition parent scope."""
        for node in self.nodes:
            if node in self.leaf_nodes:
                continue
            
            parent_scope = self.scopes[node]
            child_scopes = [self.scopes[child] for child in self.children[node]]
            
            # All child scopes should be subsets of parent scope
            for child_scope in child_scopes:
                if not child_scope.issubset(parent_scope):
                    raise ValueError(
                        f"Child scope {child_scope} not subset of parent scope {parent_scope}"
                    )
            
            # Child scopes should be disjoint
            for i, scope1 in enumerate(child_scopes):
                for j, scope2 in enumerate(child_scopes[i+1:], i+1):
                    if scope1 & scope2:
                        raise ValueError(
                            f"Child scopes {scope1} and {scope2} overlap"
                        )
            
            # Union of child scopes should equal parent scope
            union_scope = set().union(*child_scopes)
            if union_scope != parent_scope:
                raise ValueError(
                    f"Union of child scopes {union_scope} != parent scope {parent_scope}"
                )
    
    def _validate_decomposability(self) -> None:
        """Validate decomposability property: no variable appears in multiple children."""
        for node in self.nodes:
            if node in self.leaf_nodes:
                continue
            
            child_scopes = [self.scopes[child] for child in self.children[node]]
            
            # Check that no variable appears in multiple children
            for i, scope1 in enumerate(child_scopes):
                for j, scope2 in enumerate(child_scopes[i+1:], i+1):
                    if scope1 & scope2:
                        raise ValueError(
                            f"Variable overlap between children: {scope1} and {scope2}"
                        )


class LatentTree:
    """A latent tree structure with validation and utility methods."""
    
    def __init__(self, spec: TreeSpec) -> None:
        """Initialize with a validated tree specification.
        
        Args:
            spec: Tree specification with validated structure
        """
        self.spec = spec
    
    @classmethod
    def from_parents(
        cls, 
        parents: Dict[str, Optional[str]], 
        scopes: Dict[str, Set[str]]
    ) -> LatentTree:
        """Create a latent tree from parent relationships and scopes.
        
        Args:
            parents: Dictionary mapping each node to its parent (None for root)
            scopes: Dictionary mapping each node to its scope
            
        Returns:
            LatentTree with validated structure
        """
        nodes = list(parents.keys())
        
        # Build children dictionary
        children: Dict[str, List[str]] = {node: [] for node in nodes}
        root_node = None
        
        for node, parent in parents.items():
            if parent is None:
                root_node = node
            else:
                children[parent].append(node)
        
        if root_node is None:
            raise ValueError("No root node found (no node with parent=None)")
        
        # Find leaf nodes
        leaf_nodes = {node for node in nodes if not children[node]}
        
        spec = TreeSpec(
            nodes=nodes,
            parents=parents,
            children=children,
            scopes=scopes,
            leaf_nodes=leaf_nodes,
            root_node=root_node
        )
        
        return cls(spec)
    
    def get_ancestors(self, node: str) -> List[str]:
        """Get all ancestors of a node (excluding the node itself).
        
        Args:
            node: Node identifier
            
        Returns:
            List of ancestor nodes from root to immediate parent
        """
        ancestors = []
        current = self.spec.parents[node]
        while current is not None:
            ancestors.append(current)
            current = self.spec.parents[current]
        return ancestors[::-1]  # Reverse to get root-to-parent order
    
    def get_descendants(self, node: str) -> List[str]:
        """Get all descendants of a node (excluding the node itself).
        
        Args:
            node: Node identifier
            
        Returns:
            List of descendant nodes
        """
        descendants = []
        stack = self.spec.children[node].copy()
        
        while stack:
            current = stack.pop()
            descendants.append(current)
            stack.extend(self.spec.children[current])
        
        return descendants
    
    def get_path_to_root(self, node: str) -> List[str]:
        """Get the path from a node to the root.
        
        Args:
            node: Starting node
            
        Returns:
            List of nodes from node to root (inclusive)
        """
        path = [node]
        current = self.spec.parents[node]
        while current is not None:
            path.append(current)
            current = self.spec.parents[current]
        return path
    
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if one node is an ancestor of another.
        
        Args:
            ancestor: Potential ancestor node
            descendant: Potential descendant node
            
        Returns:
            True if ancestor is an ancestor of descendant
        """
        current = self.spec.parents[descendant]
        while current is not None:
            if current == ancestor:
                return True
            current = self.spec.parents[current]
        return False
    
    def get_scope(self, node: str) -> Set[str]:
        """Get the scope of a node.
        
        Args:
            node: Node identifier
            
        Returns:
            Set of variables in the node's scope
        """
        return self.spec.scopes[node]
    
    def get_children(self, node: str) -> List[str]:
        """Get the children of a node.
        
        Args:
            node: Node identifier
            
        Returns:
            List of child nodes
        """
        return self.spec.children[node]
    
    def get_parent(self, node: str) -> Optional[str]:
        """Get the parent of a node.
        
        Args:
            node: Node identifier
            
        Returns:
            Parent node identifier, or None if root
        """
        return self.spec.parents[node]
    
    def is_leaf(self, node: str) -> bool:
        """Check if a node is a leaf.
        
        Args:
            node: Node identifier
            
        Returns:
            True if the node is a leaf
        """
        return node in self.spec.leaf_nodes
    
    def is_root(self, node: str) -> bool:
        """Check if a node is the root.
        
        Args:
            node: Node identifier
            
        Returns:
            True if the node is the root
        """
        return node == self.spec.root_node
