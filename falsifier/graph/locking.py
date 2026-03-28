"""
Atomic file locking for the knowledge graph to prevent race conditions
when multiple falsifiers run concurrently.

Uses fcntl for exclusive file locking on Unix/Linux/Mac systems.
Implements atomic writes and stale lock detection.
"""

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _is_lock_stale(lock_path: Path, max_age_seconds: float = 300.0) -> bool:
    """
    Check if a lock file is stale (process no longer exists or lock is old).

    Args:
        lock_path: Path to the lock file
        max_age_seconds: Maximum age before considering lock stale (default 5 min)

    Returns:
        True if the lock is considered stale
    """
    if not lock_path.exists():
        return False

    try:
        # Check if lock file is too old
        mtime = lock_path.stat().st_mtime
        if time.time() - mtime > max_age_seconds:
            logger.warning(f"Lock file {lock_path} is older than {max_age_seconds}s, considering stale")
            return True

        # Try to read PID from lock file
        content = lock_path.read_text().strip()
        if content:
            try:
                lock_data = json.loads(content)
                pid = lock_data.get("pid")
                if pid:
                    # Check if process still exists
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        logger.warning(f"Lock holder process {pid} no longer exists, lock is stale")
                        return True
            except (json.JSONDecodeError, ValueError):
                # Old format or corrupted lock file - check if process from filename exists
                pass

    except Exception as e:
        logger.error(f"Error checking lock staleness: {e}")

    return False


def acquire_lock(lock_path: Path, timeout: float = 30.0) -> bool:
    """
    Acquire an exclusive file lock with timeout and stale detection.

    Args:
        lock_path: Path to the lock file
        timeout: Maximum seconds to wait for lock (default 30.0)

    Returns:
        True if lock acquired, False on timeout

    Raises:
        IOError: If unable to create lock file
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Check and remove stale lock
    if lock_path.exists() and _is_lock_stale(lock_path):
        logger.info(f"Removing stale lock file: {lock_path}")
        try:
            lock_path.unlink()
        except OSError as e:
            logger.warning(f"Could not remove stale lock: {e}")

    start_time = time.time()
    pid = os.getpid()
    lock_info = {
        "pid": pid,
        "timestamp": time.time(),
        "uuid": str(uuid4())[:8]
    }

    while time.time() - start_time < timeout:
        try:
            # Open in create+exclusive mode - fails if file exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # Write lock info atomically
                os.write(fd, json.dumps(lock_info).encode())
                os.fsync(fd)
            finally:
                os.close(fd)

            # Now acquire fcntl lock for true inter-process exclusion
            lock_fd = open(str(lock_path), "r+")
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Store file descriptor to prevent GC from closing it
            acquire_lock._lock_fds[lock_path] = lock_fd

            logger.debug(f"Acquired lock: {lock_path} (pid={pid})")
            return True

        except (OSError, IOError):
            # Lock file exists or fcntl failed
            time.sleep(0.1)
            continue
        except Exception as e:
            logger.error(f"Unexpected error acquiring lock: {e}")
            time.sleep(0.1)
            continue

    logger.warning(f"Timeout acquiring lock after {timeout}s: {lock_path}")
    return False


# Storage for lock file descriptors to prevent garbage collection
acquire_lock._lock_fds = {}


def release_lock(lock_path: Path) -> None:
    """
    Release a file lock and clean up lock file.

    Args:
        lock_path: Path to the lock file
    """
    lock_path = Path(lock_path)

    # Release fcntl lock
    if lock_path in acquire_lock._lock_fds:
        try:
            fd = acquire_lock._lock_fds[lock_path]
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
            del acquire_lock._lock_fds[lock_path]
            logger.debug(f"Released fcntl lock: {lock_path}")
        except Exception as e:
            logger.warning(f"Error releasing fcntl lock: {e}")

    # Remove lock file
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.debug(f"Removed lock file: {lock_path}")
    except OSError as e:
        logger.warning(f"Could not remove lock file {lock_path}: {e}")


@contextmanager
def lock_context(lock_path: Path, timeout: float = 30.0):
    """
    Context manager for safe lock acquisition and release.

    Args:
        lock_path: Path to the lock file
        timeout: Maximum seconds to wait for lock

    Yields:
        bool: True if lock was acquired

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_path = Path(lock_path)
    acquired = False

    try:
        acquired = acquire_lock(lock_path, timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire lock: {lock_path}")
        yield acquired
    finally:
        if acquired:
            release_lock(lock_path)


def atomic_write_json(data: dict, target_path: Path, indent: int = 2) -> None:
    """
    Atomically write JSON data to file using write-to-temp-then-rename pattern.

    This ensures that readers never see partially written files.

    Args:
        data: Dictionary to serialize
        target_path: Final destination path
        indent: JSON indentation level

    Raises:
        IOError: If write fails
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (same filesystem for atomic rename)
    temp_path = target_path.with_suffix(f".tmp.{uuid4().hex[:8]}")

    try:
        # Write to temp file
        temp_path.write_text(
            json.dumps(data, indent=indent, default=str),
            encoding="utf-8"
        )

        # Sync to ensure data is on disk before rename
        fd = os.open(str(temp_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        # Atomic rename (POSIX guarantees this is atomic on same filesystem)
        temp_path.rename(target_path)

        # Sync directory to ensure rename is persisted
        dir_fd = os.open(str(target_path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        logger.debug(f"Atomic write completed: {target_path}")

    except Exception as e:
        # Clean up temp file on failure
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise IOError(f"Atomic write failed for {target_path}: {e}")


def atomic_read_json(target_path: Path, default: Any = None) -> Any:
    """
    Read JSON data from file with proper error handling.

    Args:
        target_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    target_path = Path(target_path)

    if not target_path.exists():
        return default

    try:
        return json.loads(target_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading {target_path}: {e}")
        return default


class AtomicGraphUpdate:
    """
    Wrapper for atomic graph operations with file locking.

    Ensures all reads and writes to the graph are properly synchronized
    and atomic, preventing data corruption from concurrent access.
    """

    def __init__(self, graph_path: Path, lock_timeout: float = 30.0):
        """
        Initialize atomic graph updater.

        Args:
            graph_path: Path to the graph JSON file
            lock_timeout: Maximum seconds to wait for lock acquisition
        """
        self.graph_path = Path(graph_path)
        self.lock_path = self.graph_path.with_suffix(".lock")
        self.lock_timeout = lock_timeout

        # Ensure directory exists
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"AtomicGraphUpdate initialized: graph={graph_path}, lock={self.lock_path}")

    def _load_graph(self) -> dict:
        """Load graph data with lock held."""
        data = atomic_read_json(self.graph_path, {"nodes": {}, "edges": [], "metadata": {}})
        # Ensure required structure
        if "nodes" not in data:
            data["nodes"] = {}
        if "edges" not in data:
            data["edges"] = []
        if "metadata" not in data:
            data["metadata"] = {}
        return data

    def _save_graph(self, data: dict) -> None:
        """Save graph data atomically with lock held."""
        atomic_write_json(data, self.graph_path)

    def update_node(self, node_id: str, updates: dict) -> None:
        """
        Update an existing node in the graph atomically.

        Args:
            node_id: Unique identifier for the node
            updates: Dictionary of fields to update

        Raises:
            TimeoutError: If lock cannot be acquired
            KeyError: If node does not exist
        """
        with lock_context(self.lock_path, self.lock_timeout):
            graph = self._load_graph()

            if node_id not in graph["nodes"]:
                raise KeyError(f"Node '{node_id}' does not exist in graph")

            # Apply updates
            graph["nodes"][node_id].update(updates)

            # Update metadata
            graph["metadata"]["last_modified"] = time.time()
            graph["metadata"]["last_modified_by"] = os.getpid()

            self._save_graph(graph)
            logger.info(f"Updated node '{node_id}' with {len(updates)} fields")

    def create_node(self, node_id: str, node_data: dict) -> None:
        """
        Create a new node in the graph atomically.

        Args:
            node_id: Unique identifier for the node
            node_data: Dictionary containing node data

        Raises:
            TimeoutError: If lock cannot be acquired
            ValueError: If node already exists
        """
        with lock_context(self.lock_path, self.lock_timeout):
            graph = self._load_graph()

            if node_id in graph["nodes"]:
                raise ValueError(f"Node '{node_id}' already exists in graph")

            # Create node with metadata
            graph["nodes"][node_id] = {
                **node_data,
                "_created": time.time(),
                "_created_by": os.getpid()
            }

            # Update metadata
            graph["metadata"]["last_modified"] = time.time()
            graph["metadata"]["last_modified_by"] = os.getpid()
            graph["metadata"]["node_count"] = len(graph["nodes"])

            self._save_graph(graph)
            logger.info(f"Created node '{node_id}' with {len(node_data)} fields")

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from the graph atomically.

        Args:
            node_id: Unique identifier for the node

        Returns:
            True if node was deleted, False if it didn't exist

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        with lock_context(self.lock_path, self.lock_timeout):
            graph = self._load_graph()

            if node_id not in graph["nodes"]:
                return False

            del graph["nodes"][node_id]

            # Remove edges connected to this node
            graph["edges"] = [
                e for e in graph["edges"]
                if e.get("source") != node_id and e.get("target") != node_id
            ]

            # Update metadata
            graph["metadata"]["last_modified"] = time.time()
            graph["metadata"]["last_modified_by"] = os.getpid()
            graph["metadata"]["node_count"] = len(graph["nodes"])
            graph["metadata"]["edge_count"] = len(graph["edges"])

            self._save_graph(graph)
            logger.info(f"Deleted node '{node_id}' and connected edges")
            return True

    def add_edge(self, source: str, target: str, edge_data: Optional[dict] = None) -> None:
        """
        Add an edge between two nodes atomically.

        Args:
            source: Source node ID
            target: Target node ID
            edge_data: Optional additional edge attributes

        Raises:
            TimeoutError: If lock cannot be acquired
            KeyError: If source or target node doesn't exist
        """
        edge_data = edge_data or {}

        with lock_context(self.lock_path, self.lock_timeout):
            graph = self._load_graph()

            if source not in graph["nodes"]:
                raise KeyError(f"Source node '{source}' does not exist")
            if target not in graph["nodes"]:
                raise KeyError(f"Target node '{target}' does not exist")

            edge = {
                "source": source,
                "target": target,
                "_created": time.time(),
                **edge_data
            }

            graph["edges"].append(edge)
            graph["metadata"]["edge_count"] = len(graph["edges"])
            graph["metadata"]["last_modified"] = time.time()

            self._save_graph(graph)
            logger.info(f"Added edge {source} -> {target}")

    def read_graph(self) -> dict:
        """
        Read the entire graph atomically.

        Returns:
            Graph data dictionary

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        with lock_context(self.lock_path, self.lock_timeout):
            return self._load_graph()

    def bulk_update(self, update_func) -> Any:
        """
        Perform a custom bulk update with lock held.

        Args:
            update_func: Function that takes graph dict and returns result

        Returns:
            Result from update_func

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        with lock_context(self.lock_path, self.lock_timeout):
            graph = self._load_graph()
            result = update_func(graph)
            self._save_graph(graph)
            return result


# Convenience functions for simple operations

def update_node_atomic(graph_path: Path, node_id: str, updates: dict, timeout: float = 30.0) -> None:
    """Convenience function to update a node atomically."""
    updater = AtomicGraphUpdate(graph_path, timeout)
    updater.update_node(node_id, updates)


def create_node_atomic(graph_path: Path, node_id: str, node_data: dict, timeout: float = 30.0) -> None:
    """Convenience function to create a node atomically."""
    updater = AtomicGraphUpdate(graph_path, timeout)
    updater.create_node(node_id, node_data)
