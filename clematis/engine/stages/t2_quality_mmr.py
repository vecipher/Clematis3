from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Sequence

# PR38: Deterministic, side‑effect‑free MMR utilities.
#  • No RNG, no wall‑clock.
#  • Ties break by lex(id).
#  • Operates on pre‑tokenized items; tokenization lives in the quality layer.

__all__ = [
    "MMRItem",
    "jaccard_distance",
    "avg_pairwise_distance",
    "mmr_select",
    "mmr_reorder_full",
]


@dataclass(frozen=True)
class MMRItem:
    """Minimal candidate representation for MMR.

    Attributes
    ----------
    id : str
        Stable identifier used for deterministic tie‑breaking (lex order).
    rel : float
        Relevance score (use PR37 fused score).
    toks : FrozenSet[str]
        Normalized token set used for diversity (e.g., BM25 tokenizer output).
    """

    id: str
    rel: float
    toks: FrozenSet[str]


def jaccard_distance(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Return 1 − Jaccard(a, b) on token sets.

    For two empty sets we define distance = 0.0 (identical/undef but harmless).
    """
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 - (inter / union if union else 0.0)


def avg_pairwise_distance(items: Sequence[MMRItem]) -> float:
    """Average pairwise Jaccard distance across items (0.0 for n ≤ 1)."""
    n = len(items)
    if n <= 1:
        return 0.0
    s = 0.0
    c = 0
    for i in range(n):
        ti = items[i].toks
        for j in range(i + 1, n):
            s += jaccard_distance(ti, items[j].toks)
            c += 1
    return 0.0 if c == 0 else (s / c)


def _initial_order(items: Sequence[MMRItem]) -> List[int]:
    """Deterministic baseline ordering: by (-rel, id)."""
    return sorted(range(len(items)), key=lambda i: (-items[i].rel, items[i].id))


def mmr_select(
    items: Sequence[MMRItem],
    k: int | None = None,
    lam: float = 0.5,
    dist_fn: Callable[[FrozenSet[str], FrozenSet[str]], float] = jaccard_distance,
) -> List[int]:
    """Greedy Maximal Marginal Relevance selection (deterministic).

    Parameters
    ----------
    items : sequence of MMRItem
        Candidates with relevance and token sets.
    k : int | None
        Maximum number of items to select (defaults to len(items)).
    lam : float
        λ in [0, 1]; higher emphasizes diversity, 0 is relevance‑only.
    dist_fn : callable
        Distance on token sets; defaults to Jaccard distance.

    Returns
    -------
    List[int]
        Indices of selected items into the `items` sequence (length ≤ k).
    """
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0,1]")
    n = len(items)
    if n == 0:
        return []
    if k is None or k > n:
        k = n

    selected: List[int] = []
    remaining: List[int] = _initial_order(items)

    while remaining and len(selected) < k:
        best_i = remaining[0]
        best_val = None  # type: float | None
        for i in remaining:
            rel = items[i].rel
            if selected:
                # max distance to any already‑selected item
                div = 0.0
                for j in selected:
                    d = dist_fn(items[i].toks, items[j].toks)
                    if d > div:
                        div = d
                score = lam * div + (1.0 - lam) * rel
            else:
                # first pick purely by relevance (stable against id tie‑break below)
                score = rel
            if best_val is None or score > best_val or (score == best_val and items[i].id < items[best_i].id):
                best_val = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def mmr_reorder_full(items: Sequence[MMRItem], k: int | None, lam: float) -> List[int]:
    """Return a full permutation: MMR head then deterministic tail.

    The head is the MMR selection (size k or n). The tail is the remaining
    indices in the baseline deterministic order. No duplicates.
    """
    order0 = _initial_order(items)
    head = mmr_select(items, k=k, lam=lam)
    picked = set(head)
    tail = [i for i in order0 if i not in picked]
    return head + tail
