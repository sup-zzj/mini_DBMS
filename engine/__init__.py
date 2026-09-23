"""mini_DBMS 存储引擎内核。

分层结构（自底向上）::

    page / buffer_pool  物理页 + 磁盘管理 + 缓冲池（LRU/Clock/Random）
    wal                 预写日志（redo + CRC，no-steal/no-force）
    txn                 事务对象 + 事务管理器（BEGIN/COMMIT/ABORT）
    table / catalog     堆表（slotted page）+ 系统目录（JSON）
    btree               B+ 树索引（可插拔 NodeStore）
    learned_index       学习式索引（两级 RMI，供基准实验）
    storage_engine      以上各层的编排：DDL/DML/事务/崩溃恢复
"""

from .buffer_pool import BufferPool, BufferPoolFullError
from .btree import BTree, DuplicateKeyError, MemNodeStore, PoolNodeStore
from .catalog import Catalog, Column, IndexInfo, INT, REAL, TEXT, TableMeta
from .learned_index import LearnedIndex
from .page import DiskManager, PAGE_SIZE, Page
from .storage_engine import StorageEngine
from .table import encode_row, decode_row
from .txn import Transaction, TxnManager
from .wal import LogRecord, RecoveryResult, WAL

__version__ = "0.1.0"

__all__ = [
    "BufferPool",
    "BufferPoolFullError",
    "BTree",
    "DuplicateKeyError",
    "MemNodeStore",
    "PoolNodeStore",
    "Catalog",
    "Column",
    "IndexInfo",
    "INT",
    "REAL",
    "TEXT",
    "TableMeta",
    "LearnedIndex",
    "DiskManager",
    "PAGE_SIZE",
    "Page",
    "StorageEngine",
    "encode_row",
    "decode_row",
    "Transaction",
    "TxnManager",
    "LogRecord",
    "RecoveryResult",
    "WAL",
]
