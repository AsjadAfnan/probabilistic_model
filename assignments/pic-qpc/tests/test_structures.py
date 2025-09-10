"""Tests for tree structure definitions and validation."""

import pytest
import torch

from pic.structures import LatentTree, TreeSpec


class TestTreeSpec:
    """Test TreeSpec validation and properties."""
    
    def test_valid_tree_structure(self):
        """Test valid tree structure creation."""
        nodes = ["root", "child1", "child2"]
        parents = {"root": None, "child1": "root", "child2": "root"}
        children = {"root": ["child1", "child2"], "child1": [], "child2": []}
        scopes = {
            "root": {"x1", "x2", "x3"},
            "child1": {"x1", "x2"},
            "child2": {"x3"}
        }
        leaf_nodes = {"child1", "child2"}
        root_node = "root"
        
        spec = TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)
        
        assert spec.nodes == nodes
        assert spec.parents == parents
        assert spec.children == children
        assert spec.scopes == scopes
        assert spec.leaf_nodes == leaf_nodes
        assert spec.root_node == root_node
    
    def test_invalid_root_parent(self):
        """Test that root node cannot have a parent."""
        nodes = ["root", "child1"]
        parents = {"root": "child1", "child1": None}  # Invalid: root has parent
        children = {"root": [], "child1": ["root"]}
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        leaf_nodes = {"root"}
        root_node = "root"
        
        with pytest.raises(ValueError, match="Root node.*has parent"):
            TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)
    
    def test_invalid_non_root_no_parent(self):
        """Test that non-root nodes must have parents."""
        nodes = ["root", "child1"]
        parents = {"root": None, "child1": None}  # Invalid: child1 has no parent
        children = {"root": ["child1"], "child1": []}
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        leaf_nodes = {"child1"}
        root_node = "root"
        
        with pytest.raises(ValueError, match="Non-root node.*has no parent"):
            TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)
    
    def test_invalid_leaf_with_children(self):
        """Test that leaf nodes cannot have children."""
        nodes = ["root", "child1"]
        parents = {"root": None, "child1": "root"}
        children = {"root": ["child1"], "child1": ["root"]}  # Invalid: leaf has child
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        leaf_nodes = {"child1"}
        root_node = "root"
        
        with pytest.raises(ValueError, match="Leaf node.*has children"):
            TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)
    
    def test_invalid_smoothness(self):
        """Test smoothness validation."""
        nodes = ["root", "child1", "child2"]
        parents = {"root": None, "child1": "root", "child2": "root"}
        children = {"root": ["child1", "child2"], "child1": [], "child2": []}
        scopes = {
            "root": {"x1", "x2"},
            "child1": {"x1", "x2"},  # Invalid: overlaps with child2
            "child2": {"x2"}
        }
        leaf_nodes = {"child1", "child2"}
        root_node = "root"
        
        with pytest.raises(ValueError, match="Child scopes.*overlap"):
            TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)
    
    def test_invalid_decomposability(self):
        """Test decomposability validation."""
        nodes = ["root", "child1", "child2"]
        parents = {"root": None, "child1": "root", "child2": "root"}
        children = {"root": ["child1", "child2"], "child1": [], "child2": []}
        scopes = {
            "root": {"x1", "x2", "x3"},
            "child1": {"x1", "x2"},
            "child2": {"x2", "x3"}  # Invalid: x2 appears in both children
        }
        leaf_nodes = {"child1", "child2"}
        root_node = "root"
        
        with pytest.raises(ValueError, match="Child scopes.*overlap"):
            TreeSpec(nodes, parents, children, scopes, leaf_nodes, root_node)


class TestLatentTree:
    """Test LatentTree functionality."""
    
    def test_from_parents_valid(self):
        """Test creating LatentTree from parent relationships."""
        parents = {"root": None, "child1": "root", "child2": "root"}
        scopes = {
            "root": {"x1", "x2", "x3"},
            "child1": {"x1", "x2"},
            "child2": {"x3"}
        }
        
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.spec.root_node == "root"
        assert tree.spec.leaf_nodes == {"child1", "child2"}
        assert tree.spec.parents == parents
        assert tree.spec.scopes == scopes
    
    def test_from_parents_no_root(self):
        """Test error when no root node is specified."""
        parents = {"node1": "node2", "node2": "node1"}  # No root
        scopes = {"node1": {"x1"}, "node2": {"x2"}}
        
        with pytest.raises(ValueError, match="No root node found"):
            LatentTree.from_parents(parents, scopes)
    
    def test_get_ancestors(self):
        """Test getting ancestors of a node."""
        parents = {"root": None, "child1": "root", "grandchild": "child1"}
        scopes = {"root": {"x1"}, "child1": {"x1"}, "grandchild": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        ancestors = tree.get_ancestors("grandchild")
        assert ancestors == ["root", "child1"]
        
        ancestors = tree.get_ancestors("child1")
        assert ancestors == ["root"]
        
        ancestors = tree.get_ancestors("root")
        assert ancestors == []
    
    def test_get_descendants(self):
        """Test getting descendants of a node."""
        parents = {"root": None, "child1": "root", "child2": "root", "grandchild": "child1"}
        scopes = {"root": {"x1", "x2"}, "child1": {"x1"}, "child2": {"x2"}, "grandchild": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        descendants = tree.get_descendants("root")
        assert set(descendants) == {"child1", "child2", "grandchild"}
        
        descendants = tree.get_descendants("child1")
        assert descendants == ["grandchild"]
        
        descendants = tree.get_descendants("child2")
        assert descendants == []
    
    def test_get_path_to_root(self):
        """Test getting path from node to root."""
        parents = {"root": None, "child1": "root", "grandchild": "child1"}
        scopes = {"root": {"x1"}, "child1": {"x1"}, "grandchild": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        path = tree.get_path_to_root("grandchild")
        assert path == ["grandchild", "child1", "root"]
        
        path = tree.get_path_to_root("child1")
        assert path == ["child1", "root"]
        
        path = tree.get_path_to_root("root")
        assert path == ["root"]
    
    def test_is_ancestor(self):
        """Test ancestor relationship checking."""
        parents = {"root": None, "child1": "root", "grandchild": "child1"}
        scopes = {"root": {"x1"}, "child1": {"x1"}, "grandchild": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.is_ancestor("root", "child1")
        assert tree.is_ancestor("root", "grandchild")
        assert tree.is_ancestor("child1", "grandchild")
        assert not tree.is_ancestor("child1", "root")
        assert not tree.is_ancestor("grandchild", "root")
    
    def test_get_scope(self):
        """Test getting node scope."""
        parents = {"root": None, "child1": "root", "child2": "root"}
        scopes = {"root": {"x1", "x2"}, "child1": {"x1"}, "child2": {"x2"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.get_scope("root") == {"x1", "x2"}
        assert tree.get_scope("child1") == {"x1"}
    
    def test_get_children(self):
        """Test getting node children."""
        parents = {"root": None, "child1": "root", "child2": "root"}
        scopes = {"root": {"x1", "x2"}, "child1": {"x1"}, "child2": {"x2"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        children = tree.get_children("root")
        assert set(children) == {"child1", "child2"}
        
        children = tree.get_children("child1")
        assert children == []
    
    def test_get_parent(self):
        """Test getting node parent."""
        parents = {"root": None, "child1": "root"}
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.get_parent("root") is None
        assert tree.get_parent("child1") == "root"
    
    def test_is_leaf(self):
        """Test leaf node checking."""
        parents = {"root": None, "child1": "root"}
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        assert not tree.is_leaf("root")
        assert tree.is_leaf("child1")
    
    def test_is_root(self):
        """Test root node checking."""
        parents = {"root": None, "child1": "root"}
        scopes = {"root": {"x1"}, "child1": {"x1"}}
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.is_root("root")
        assert not tree.is_root("child1")


class TestComplexTreeStructures:
    """Test more complex tree structures."""
    
    def test_deep_tree(self):
        """Test deep tree structure."""
        parents = {
            "root": None,
            "level1_1": "root",
            "level1_2": "root",
            "level2_1": "level1_1",
            "level2_2": "level1_1",
            "level3_1": "level2_1",
            "level3_2": "level2_1"
        }
        scopes = {
            "root": {"x1", "x2", "x3", "x4", "x5", "x6"},
            "level1_1": {"x1", "x2", "x3"},
            "level1_2": {"x4", "x5", "x6"},
            "level2_1": {"x1", "x2"},
            "level2_2": {"x3"},
            "level3_1": {"x1"},
            "level3_2": {"x2"}
        }
        
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.spec.root_node == "root"
        assert tree.spec.leaf_nodes == {"level1_2", "level2_2", "level3_1", "level3_2"}
        
        # Test paths
        path = tree.get_path_to_root("level3_1")
        assert path == ["level3_1", "level2_1", "level1_1", "root"]
        
        # Test ancestors
        ancestors = tree.get_ancestors("level3_1")
        assert ancestors == ["root", "level1_1", "level2_1"]
        
        # Test descendants
        descendants = tree.get_descendants("level1_1")
        assert set(descendants) == {"level2_1", "level2_2", "level3_1", "level3_2"}
    
    def test_binary_tree(self):
        """Test binary tree structure."""
        parents = {
            "root": None,
            "left": "root",
            "right": "root",
            "left_left": "left",
            "left_right": "left",
            "right_left": "right",
            "right_right": "right"
        }
        scopes = {
            "root": {"x1", "x2", "x3", "x4", "x5", "x6", "x7"},
            "left": {"x1", "x2", "x3"},
            "right": {"x4", "x5", "x6", "x7"},
            "left_left": {"x1"},
            "left_right": {"x2", "x3"},
            "right_left": {"x4", "x5"},
            "right_right": {"x6", "x7"}
        }
        
        tree = LatentTree.from_parents(parents, scopes)
        
        assert tree.spec.leaf_nodes == {"left_left", "left_right", "right_left", "right_right"}
        
        # Test that all paths to root are correct
        for leaf in tree.spec.leaf_nodes:
            path = tree.get_path_to_root(leaf)
            assert path[-1] == "root"
            assert len(path) >= 2
