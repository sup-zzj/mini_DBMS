"""Buffer pool unit tests: capacity-bounded eviction, dirty-page write-back,
pinning (no-steal), LRU/clock/random replacement order and hit statistics.
Disk I/O is simulated with in-memory DiskManagers for speed and determinism."""
import pytest

from engine.buffer_pool import BufferPool, BufferPoolFullError
from engine.page import PAGE_SIZE, DiskManager


def make_pool(capacity=4, policy="clock", seed=0):
    pool = BufferPool(capacity, PAGE_SIZE, policy, seed)
    disk = DiskManager("mem://test", in_memory=True)
    pool.register_relation("r", disk)
    return pool, disk


def test_capacity_bounded_eviction():
    pool, disk = make_pool(capacity=4)
    for i in range(10):
        page = pool.allocate_page("r")
        pool.unpin(page)
    assert pool.resident <= 4
    # all pages are still readable from disk
    for i in range(10):
        page = pool.fetch_page("r", i)
        pool.unpin(page)
    assert pool.resident == 4


def test_dirty_page_flushed_on_eviction():
    pool, disk = make_pool(capacity=2)
    page = pool.allocate_page("r")
    page.buffer[0:4] = b"\xaa\xbb\xcc\xdd"
    pool.unpin(page, dirty=True)
    # fetch enough pages to force eviction of page 0
    for i in range(1, 5):
        p = pool.fetch_page("r", i)
        pool.unpin(p)
    assert disk.read_page(0)[0:4] == b"\xaa\xbb\xcc\xdd"


def test_pin_prevents_eviction():
    pool, disk = make_pool(capacity=3)
    pinned = [pool.allocate_page("r") for _ in range(3)]  # pin_count = 1 each
    with pytest.raises(BufferPoolFullError):
        pool.fetch_page("r", 99)
    # releasing a pin makes room
    pool.unpin(pinned[0])
    page = pool.fetch_page("r", 99)
    pool.unpin(page)


def test_lru_eviction_order():
    pool, disk = make_pool(capacity=3, policy="lru")
    for i in range(3):
        p = pool.fetch_page("r", i)
        pool.unpin(p)
    # touch page 0 -> LRU order becomes (1, 2, 0)
    p = pool.fetch_page("r", 0)
    pool.unpin(p)
    p = pool.fetch_page("r", 3)  # must evict page 1 (least recently used)
    pool.unpin(p)
    assert "r", 1 not in pool.pages
    assert ("r", 0) in pool.pages
    assert ("r", 2) in pool.pages


def test_clock_second_chance():
    pool, disk = make_pool(capacity=2, policy="clock")
    # a burst of 3 distinct pages over a 2-frame pool: the first eviction
    # (of page 0) moves the hand and clears every reference bit
    for i in range(3):
        p = pool.fetch_page("r", i)
        pool.unpin(p)
    assert ("r", 0) not in pool.pages
    # re-touch page 0: its ref bit now grants a second chance, so the next
    # eviction takes page 2 (untouched) instead
    p = pool.fetch_page("r", 0)
    pool.unpin(p)
    p = pool.fetch_page("r", 3)
    pool.unpin(p)
    assert ("r", 0) in pool.pages
    assert ("r", 2) not in pool.pages


def test_random_policy_deterministic_with_seed():
    pool_a, _ = make_pool(capacity=2, policy="random", seed=7)
    pool_b, _ = make_pool(capacity=2, policy="random", seed=7)
    for pool in (pool_a, pool_b):
        for i in range(5):
            p = pool.fetch_page("r", i)
            pool.unpin(p)
    assert sorted(k[1] for k in pool_a.pages) == sorted(k[1] for k in pool_b.pages)


def test_hit_ratio_and_reset():
    pool, disk = make_pool(capacity=3)
    for i in range(3):
        p = pool.fetch_page("r", i)
        pool.unpin(p)
    assert pool.stats["hits"] == 0
    assert pool.stats["misses"] == 3
    p = pool.fetch_page("r", 0)  # still resident -> guaranteed hit
    pool.unpin(p)
    assert pool.stats["hits"] == 1
    assert pool.hit_ratio() == pytest.approx(1 / 4)
    pool.reset_stats()
    assert pool.stats == {"hits": 0, "misses": 0, "reads": 0, "writes": 0}


def test_flush_all_writes_all_dirty_pages():
    pool, disk = make_pool(capacity=4)
    for i in range(4):
        p = pool.allocate_page("r")
        p.buffer[8:12] = i.to_bytes(4, "little")
        pool.unpin(p, dirty=True)
    assert pool.flush_all() == 4
    for i in range(4):
        assert disk.read_page(i)[8:12] == i.to_bytes(4, "little")


def test_delete_page_reclaims_id():
    pool, disk = make_pool(capacity=4)
    p = pool.allocate_page("r")
    pid = p.page_id
    pool.unpin(p)
    pool.delete_page("r", pid)
    p2 = pool.allocate_page("r")
    assert p2.page_id == pid
