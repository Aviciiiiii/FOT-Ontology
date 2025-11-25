"""
Trie data structure for efficient entity matching in text cleaning.

This module implements a Trie (prefix tree) for fast lookup of Field of Technology (FOT)
entities during the text cleaning process. The Trie supports exact matching of multi-word
phrases and is case-insensitive.
"""

from __future__ import annotations
from typing import List, Optional


class TrieNode:
    """A node in the Trie data structure."""

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.fot = None


class Trie:
    """Trie data structure for efficient entity phrase matching."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, words: List[str], fot: str) -> None:
        """Insert an entity phrase into the Trie.

        Args:
            words: List of words that make up the entity phrase
            fot: The original FOT entity string
        """
        node = self.root
        for word in words:
            word = word.lower()
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.is_end = True
        node.fot = fot

    def search_exact(self, words: List[str]) -> Optional[str]:
        """Search for an exact match of the word sequence.

        Args:
            words: List of words to search for

        Returns:
            The matching FOT entity string if found, None otherwise
        """
        node = self.root
        for word in words:
            word = word.lower()
            if word not in node.children:
                return None
            node = node.children[word]
        return node.fot if node.is_end else None


def build_fot_trie(fot_list: List[str]) -> Trie:
    """Build a Trie from a list of FOT entities.

    Args:
        fot_list: List of FOT entity strings

    Returns:
        A Trie containing all the entities
    """
    trie = Trie()
    for fot in fot_list:
        if fot and isinstance(fot, str):  # Skip empty or invalid entries
            trie.insert(fot.lower().split(), fot)
    return trie