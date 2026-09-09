"""Resource-aware process budgets and single-writer locks (standard library only)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import math
import os
from pathlib import Path

GIB = 1024 ** 3
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "STAN_NUM_THREADS",
)


def limit_native_threads(threads: int = 1) -> None:
    """Call before importing numerical libraries; inherited by spawned children."""
    if threads < 1:
        raise ValueError("threads must be positive")
    for name in NATIVE_THREAD_VARIABLES:
        os.environ[name] = str(threads)


def cgroup_directories(root: Path = Path("/sys/fs/cgroup")) -> list[Path]:
    try:
        relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                        if line.startswith("0::"))
        current = root / relative.lstrip("/")
        if not current.is_relative_to(root):
            return [root]
        return [current, *[parent for parent in current.parents if parent.is_relative_to(root)]]
    except (OSError, StopIteration):
        return [root]


def available_resources() -> tuple[int, int]:
    """CPU affinity/quota and available memory, including ancestor cgroup limits."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    memory = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                memory = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    for directory in cgroup_directories():
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if quota != "max":
                cpus = min(cpus, max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError, ZeroDivisionError):
            pass
        try:
            maximum = (directory / "memory.max").read_text().strip()
            if maximum != "max":
                remaining = max(0, int(maximum) - int((directory / "memory.current").read_text()))
                memory = min(memory, remaining) if memory is not None else remaining
        except (OSError, ValueError):
            pass
    return cpus, memory if memory is not None else -1


@dataclass(frozen=True)
class ProcessBudget:
    available_cpus: int
    available_memory_bytes: int
    workers: int
    slots_per_worker: int
    memory_gib_per_worker: float
    reserved_cpus: int
    memory_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def process_budget(
    requested: int, *, tasks: int, slots_per_worker: int = 1,
    memory_gib_per_worker: float = 1.0, reserve_cpus: int = 4,
    memory_fraction: float = 0.8, resources: tuple[int, int] | None = None,
) -> ProcessBudget:
    if requested < 0 or tasks < 1 or slots_per_worker < 1 or memory_gib_per_worker <= 0:
        raise ValueError("Invalid worker, task, CPU-slot or memory budget")
    if reserve_cpus < 0 or not 0 < memory_fraction <= 1:
        raise ValueError("Invalid resource reserve")
    cpus, memory = resources if resources is not None else available_resources()
    if cpus < slots_per_worker:
        raise RuntimeError("A worker requests more concurrent CPU slots than the effective CPU quota")
    cpu_limit = max(1, max(1, cpus - reserve_cpus) // slots_per_worker)
    # Unknown memory means conservative single-process execution.
    memory_limit = int(memory * memory_fraction // (memory_gib_per_worker * GIB)) if memory >= 0 else 1
    if memory_limit < 1:
        raise RuntimeError("Insufficient available memory for one worker at the requested estimate")
    workers = min(tasks, cpu_limit, memory_limit, requested or tasks)
    return ProcessBudget(cpus, memory, workers, slots_per_worker, memory_gib_per_worker, reserve_cpus, memory_fraction)


@contextmanager
def exclusive_run_lock(path: Path):
    """Reject duplicate runs; do not unlink a lock another writer could be using."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process already holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
