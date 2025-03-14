from tree_sitter import Node

from gnn-baselines.sitter.base import TreeSitter
from gnn-baselines.sitter.build import TREE_SITTER_LIB


def parse(s: str) -> Node:
    ts = TreeSitter(TREE_SITTER_LIB)
    return ts.parse(s).root_node


def traverse_leaves_dfs(root: Node):
    for child in root.children:
        if len(child.children) == 0:
            yield child
        else:
            yield from traverse_leaves_dfs(child)
