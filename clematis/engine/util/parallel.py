# clematis/engine/util/parallel.py
from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar, Generic, Any
from clematis.errors import ClematisError

K = TypeVar("K")  # task key (e.g., graph_id, shard_id)
R = TypeVar("R")  # task result
A = TypeVar("A")  # aggregated (merged) result


@dataclass(frozen=True)
class TaskError(Generic[K]):
    key: K
    exc_type: str
    message: str


class ParallelError(ClematisError):
    """Deterministic container for parallel task failures."""

    def __init__(self, errors: Sequence[TaskError[Any]]):
        self.errors: List[TaskError[Any]] = list(errors)
        msg = "; ".join(f"[{e.key}] {e.exc_type}: {e.message}" for e in self.errors)
        super().__init__(f"{len(self.errors)} parallel task(s) failed: {msg}")


def run_parallel(
    tasks: Sequence[Tuple[K, Callable[[], R]]],
    *,
    max_workers: int,
    merge_fn: Callable[[List[Tuple[K, R]]], A],
    order_key: Callable[[K], Any],
) -> A:
    """
    Execute tasks possibly in parallel and merge results deterministically.

    - tasks: sequence of (key, thunk) where thunk() -> R, no args; key used for ordering.
    - max_workers <= 1  => sequential path (identity-unchanged).
    - merge_fn receives a list of (key, result) sorted by order_key(key) and MUST be pure.
    - Exceptions are collected and raised deterministically as ParallelError,
      with errors sorted by order_key(key) then by first-seen index to break ties.
    """
    n = len(tasks)
    if n == 0:
        # Deterministic: caller controls what empty aggregation means.
        return merge_fn([])

    # Normalize workers (defensive; config already normalizes in PR63).
    if max_workers <= 1:
        results: List[Tuple[K, R]] = []
        for k, fn in tasks:
            try:
                r = fn()
                results.append((k, r))
            except Exception as e:  # noqa: BLE001
                raise ParallelError([TaskError(k, type(e).__name__, str(e))]) from e
        results.sort(key=lambda kr: (order_key(kr[0]),))
        return merge_fn(results)

    eff_workers = max(1, min(max_workers, n))
    results_unordered: List[Tuple[int, K, R]] = []
    errors: List[Tuple[Any, int, TaskError[K]]] = []

    # Submit in a stable order; record submit index to break equal-order_key ties.
    with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="clematis-par") as ex:
        futures: List[Tuple[int, K, Future[R]]] = []
        for idx, (k, fn) in enumerate(tasks):
            fut = ex.submit(fn)
            futures.append((idx, k, fut))

        for idx, k, fut in futures:
            try:
                r = fut.result()
                results_unordered.append((idx, k, r))
            except Exception as e:  # noqa: BLE001
                te = TaskError(k, type(e).__name__, str(e))
                errors.append((order_key(k), idx, te))

    if errors:
        # Sort deterministically by order_key then by original index.
        errors.sort(key=lambda t: (t[0], t[1]))
        raise ParallelError([te for _, _, te in errors])

    # Deterministic merge: sort by order_key(key) then original index to break ties.
    results_ordered: List[Tuple[K, R]] = [
        (k, r) for _, k, r in sorted(results_unordered, key=lambda t: (order_key(t[1]), t[0]))
    ]
    return merge_fn(results_ordered)
