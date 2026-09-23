"""B+ tree index with a pluggable node store.

The tree logic is independent of where nodes live:

* :class:`MemNodeStore` keeps nodes as plain dicts (used by unit tests and
  the index micro-benchmark, where the *algorithm* is compared head-to-head
  against sorted arrays and the learned index);
* :class:`PoolNodeStore` serialises every node into a 4 KB page managed by
  the buffer pool, which is what the storage engine actually uses.

Node model
----------
::

    leaf     = {"t": 1, "k": [keys], "v": [record_ids], "d": [tombstones], "n": next_leaf}
    internal = {"t": 2, "k": [separator keys], "c": [child page ids]}

Invariants
----------
* keys are unique and kept sorted;
* a leaf holds at most ``order`` keys; an internal node at most ``order``
  children (so ``order - 1`` separator keys) — exceeding this triggers a
  split;
* a leaf entry may be tombstoned (``d[i]``).  Tombstones are compacted
  in place when a leaf is more than half dead; **cross-node underflow
  rebalancing is deliberately not implemented** (documented limitation).

Deleting a key therefore never shrinks the tree — space is only reclaimed
by rebuilding the index.  This keeps the code robust at the cost of a
documented deviation from a fully balanced B+ tree.
"""

from __future__ import annotations

import bisect
import json
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

from .buffer_pool import BufferPool
from .page import PAGE_SIZE
from .table import PAGE_TYPE_BTREE_LEAF, PAGE_TYPE_BTREE_INTERNAL, ModifyHook


class DuplicateKeyError(ValueError):
    pass


# ----------------------------------------------------------------------
# node stores
# ----------------------------------------------------------------------
class NodeStore(ABC):
    @abstractmethod
    def get(self, page_id: int) -> dict:
        ...

    @abstractmethod
    def put(self, page_id: int, node: dict) -> None:
        ...

    @abstractmethod
    def new(self) -> int:
        ...

    @abstractmethod
    def free(self, page_id: int) -> None:
        ...


class MemNodeStore(NodeStore):
    """Nodes live in a plain dict — no serialisation, no I/O."""

    def __init__(self) -> None:
        self.nodes: Dict[int, dict] = {}
        self._next = 0

    def get(self, page_id: int) -> dict:
        return self.nodes[page_id]

    def put(self, page_id: int, node: dict) -> None:
        self.nodes[page_id] = node

    def new(self) -> int:
        node_id = self._next
        self._next += 1
        return node_id

    def free(self, page_id: int) -> None:
        self.nodes.pop(page_id, None)

    def __len__(self) -> int:
        return len(self.nodes)


class PoolNodeStore(NodeStore):
    """Nodes serialised as JSON inside 4 KB buffer-pool pages.

    Page layout: ``[u8 page_type][u32 json_len][json bytes]``.  JSON is used
    for clarity; a production engine would use compact binary node formats.
    """

    def __init__(self, pool: BufferPool, rel: str, on_modify: Optional[ModifyHook] = None) -> None:
        self.pool = pool
        self.rel = rel
        self.on_modify = on_modify

    def get(self, page_id: int) -> dict:
        page = self.pool.fetch_page(self.rel, page_id)
        try:
            (length,) = int.from_bytes(page.buffer[1:5], "little"),
            return json.loads(bytes(page.buffer[5 : 5 + length]))
        finally:
            self.pool.unpin(page)

    def put(self, page_id: int, node: dict) -> None:
        data = json.dumps(node).encode("utf-8")
        if len(data) + 5 > PAGE_SIZE:
            raise OverflowError("B+ tree node exceeds one page")
        page = self.pool.fetch_page(self.rel, page_id)
        try:
            page.buffer[0] = PAGE_TYPE_BTREE_LEAF if node["t"] == 1 else PAGE_TYPE_BTREE_INTERNAL
            page.buffer[1:5] = len(data).to_bytes(4, "little")
            page.buffer[5 : 5 + len(data)] = data
            page.dirty = True
        finally:
            self.pool.unpin(page)
        if self.on_modify:
            self.on_modify(self.rel, page_id)

    def new(self) -> int:
        page = self.pool.allocate_page(self.rel)
        node_id = page.page_id
        self.pool.unpin(page)
        return node_id

    def free(self, page_id: int) -> None:
        self.pool.delete_page(self.rel, page_id)


# ----------------------------------------------------------------------
# the tree
# ----------------------------------------------------------------------
class BTree:
    """A unique-key B+ tree.  ``order`` is the maximum fanout (keys per
    leaf / children per internal node)."""

    def __init__(self, order: int, store: NodeStore, root_page: int | None = None) -> None:
        if order < 3:
            raise ValueError("B+ tree order must be >= 3")
        self.order = order
        self.store = store
        self.root = root_page
        if self.root is None:
            self.root = self._new_leaf()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def insert(self, key: int, value: int) -> None:
        """Insert ``(key, value)``; raises :class:`DuplicateKeyError` when
        the key is already live (tombstoned keys are resurrected)."""
        promote = self._insert(self.root, key, value)
        if promote is not None:
            sep, right = promote
            old_root = self.root
            new_root = self._new_internal([sep], [old_root, right])
            self.root = new_root

    def get(self, key: int) -> Optional[int]:
        """Point lookup; returns the record id or None."""
        node = self.store.get(self._descend(key))
        keys = node["k"]
        i = bisect.bisect_left(keys, key)
        if i < len(keys) and keys[i] == key and not node["d"][i]:
            return node["v"][i]
        return None

    def contains(self, key: int) -> bool:
        return self.get(key) is not None

    def delete(self, key: int) -> bool:
        """Tombstone a key; returns False if it was not live.  No underflow
        rebalancing (documented limitation)."""
        node = self.store.get(self._descend(key))
        keys = node["k"]
        i = bisect.bisect_left(keys, key)
        if i >= len(keys) or keys[i] != key or node["d"][i]:
            return False
        node["d"][i] = True
        if sum(node["d"]) > len(keys) // 2:
            self._compact_leaf(node)
        self.store.put(self._descend(key), node)
        return True

    def range_scan(self, lo: int | None = None, hi: int | None = None) -> List[Tuple[int, int]]:
        """Inclusive range scan over ``[lo, hi]`` (None = unbounded).  A
        ``seen`` set guards against the leaf sentinel (``n = 0``) colliding
        with leaf id 0 in a single-leaf tree."""
        out: List[Tuple[int, int]] = []
        leaf_id = self._leftmost() if lo is None else self._descend(lo)
        seen = set()
        while leaf_id is not None and leaf_id not in seen:
            seen.add(leaf_id)
            node = self.store.get(leaf_id)
            keys, values, deleted, nxt = node["k"], node["v"], node["d"], node["n"]
            start = 0 if lo is None else bisect.bisect_left(keys, lo)
            for i in range(start, len(keys)):
                if hi is not None and keys[i] > hi:
                    return out
                if not deleted[i]:
                    out.append((keys[i], values[i]))
            leaf_id = nxt
        return out

    def entries(self) -> Iterator[Tuple[int, int]]:
        """All live ``(key, value)`` pairs in key order."""
        yield from self.range_scan(None, None)

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())

    def height(self) -> int:
        """Number of levels from root to a leaf (1 = root is a leaf)."""
        node = self.store.get(self.root)
        h = 1
        while node["t"] == 2:
            node = self.store.get(node["c"][0])
            h += 1
        return h

    def node_count(self) -> int:
        seen = set()
        stack = [self.root]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.store.get(node_id)
            if node["t"] == 2:
                stack.extend(node["c"])
        return len(seen)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _descend(self, key: int) -> int:
        """Follow the path to the leaf that would contain ``key``."""
        node_id = self.root
        node = self.store.get(node_id)
        while node["t"] == 2:
            i = bisect.bisect_right(node["k"], key)
            node_id = node["c"][i]
            node = self.store.get(node_id)
        return node_id

    def _leftmost(self) -> int:
        node_id = self.root
        node = self.store.get(node_id)
        while node["t"] == 2:
            node_id = node["c"][0]
            node = self.store.get(node_id)
        return node_id

    def _new_leaf(self) -> int:
        node_id = self.store.new()
        self.store.put(node_id, {"t": 1, "k": [], "v": [], "d": [], "n": 0})
        return node_id

    def _new_internal(self, keys: List[int], children: List[int]) -> int:
        node_id = self.store.new()
        self.store.put(node_id, {"t": 2, "k": keys, "c": children})
        return node_id

    def _insert(self, node_id: int, key: int, value: int) -> Optional[Tuple[int, int]]:
        """Insert below ``node_id``.  Returns ``(separator, right_child)``
        if this node split, else None.  ``separator`` must be promoted to
        the parent."""
        node = self.store.get(node_id)
        if node["t"] == 1:  # leaf
            keys, values, deleted = node["k"], node["v"], node["d"]
            i = bisect.bisect_left(keys, key)
            if i < len(keys) and keys[i] == key:
                if deleted[i]:
                    deleted[i] = False
                    values[i] = value
                    node["v"], node["d"] = values, deleted
                    self.store.put(node_id, node)
                    return None
                raise DuplicateKeyError(f"key {key} already exists")
            keys.insert(i, key)
            values.insert(i, value)
            deleted.insert(i, False)
            if len(keys) <= self.order:
                self.store.put(node_id, node)
                return None
            return self._split_leaf(node_id, node)
        # internal node
        i = bisect.bisect_right(node["k"], key)
        promote = self._insert(node["c"][i], key, value)
        if promote is None:
            return None
        sep, right = promote
        keys, children = node["k"], node["c"]
        j = bisect.bisect_left(keys, sep)
        keys.insert(j, sep)
        children.insert(j + 1, right)
        if len(children) <= self.order:
            self.store.put(node_id, node)
            return None
        return self._split_internal(node_id, node)

    def _split_leaf(self, node_id: int, node: dict) -> Tuple[int, int]:
        keys, values, deleted = node["k"], node["v"], node["d"]
        mid = len(keys) // 2
        right_id = self._new_leaf()
        right = {"t": 1, "k": keys[mid:], "v": values[mid:], "d": deleted[mid:], "n": node["n"]}
        node["k"], node["v"], node["d"] = keys[:mid], values[:mid], deleted[:mid]
        node["n"] = right_id
        self.store.put(node_id, node)
        self.store.put(right_id, right)
        return (right["k"][0], right_id)

    def _split_internal(self, node_id: int, node: dict) -> Tuple[int, int]:
        keys, children = node["k"], node["c"]
        mid = len(keys) // 2
        sep = keys[mid]
        right_id = self._new_internal(keys[mid + 1 :], children[mid + 1 :])
        node["k"], node["c"] = keys[:mid], children[: mid + 1]
        self.store.put(node_id, node)
        return (sep, right_id)

    @staticmethod
    def _compact_leaf(node: dict) -> None:
        keep = [i for i, d in enumerate(node["d"]) if not d]
        node["k"] = [node["k"][i] for i in keep]
        node["v"] = [node["v"][i] for i in keep]
        node["d"] = [False] * len(keep)
