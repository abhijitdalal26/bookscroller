"""
Runtime taste vector — tracks user preferences in embedding space.

The taste vector is a weighted sum of all books the user has interacted with,
living in the same 1152-dim space as combined_embeddings.npy.
This module is the bridge between user behaviour and pgvector ANN search.

Signal weights (tuned from first principles, adjust after real data):
    save / add to list   +1.00   strongest positive
    tap for details      +0.50   moderate positive
    scroll-back          +0.40   strong positive
    dwell  3–10 s        +0.20   mild positive
    dwell  1–3  s        +0.05   weak positive
    skip  < 1   s        -0.30   negative
"""
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Signal = Literal['save', 'tap', 'scroll_back', 'dwell_long', 'dwell_short', 'skip']

SIGNAL_WEIGHT: dict[Signal, float] = {
    'save':         1.00,
    'tap':          0.50,
    'scroll_back':  0.40,
    'dwell_long':   0.20,   # 3–10 s
    'dwell_short':  0.05,   # 1–3 s
    'skip':        -0.30,
}

DECAY = 0.95   # multiply older interactions by this each new signal (recency bias)


@dataclass
class TasteVector:
    """
    Maintains a running user taste vector.
    Persist the vector between sessions by saving/loading to JSON.
    """
    dim: int = 1152
    _vec: np.ndarray = field(default=None, repr=False)
    _total_weight: float = 0.0

    def __post_init__(self):
        if self._vec is None:
            self._vec = np.zeros(self.dim, dtype=np.float32)

    # ── Core update ──────────────────────────────────────────────
    def update(self, embedding: np.ndarray, signal: Signal):
        """Add one interaction to the taste vector."""
        w = SIGNAL_WEIGHT[signal]
        # Apply decay to existing vector (recency bias)
        self._vec          *= DECAY
        self._total_weight *= DECAY
        # Add weighted embedding
        self._vec          += w * embedding
        self._total_weight += abs(w)

    def update_from_dwell(self, embedding: np.ndarray, dwell_ms: int):
        """Convenience wrapper — converts dwell_ms to the right signal."""
        if dwell_ms < 1000:
            signal: Signal = 'skip'
        elif dwell_ms < 3000:
            signal = 'dwell_short'
        else:
            signal = 'dwell_long'
        self.update(embedding, signal)

    # ── Query vector ─────────────────────────────────────────────
    def query_vector(self) -> np.ndarray | None:
        """
        Return L2-normalized taste vector for pgvector cosine search.
        Returns None if no interactions yet (use popularity ranking instead).
        """
        if self._total_weight < 0.01:
            return None
        norm = np.linalg.norm(self._vec)
        if norm < 1e-8:
            return None
        return (self._vec / norm).astype(np.float32)

    def is_cold(self) -> bool:
        return self._total_weight < 0.01

    # ── Persistence ──────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            'dim':           self.dim,
            'vec':           self._vec.tolist(),
            'total_weight':  self._total_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TasteVector':
        tv = cls(dim=d['dim'])
        tv._vec          = np.array(d['vec'], dtype=np.float32)
        tv._total_weight = d['total_weight']
        return tv

    def save(self, path: str | Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> 'TasteVector':
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── pgvector query helper ─────────────────────────────────────────
def format_pgvector_query(taste: TasteVector, top_k: int = 20) -> dict:
    """
    Build the Supabase RPC payload for ANN search.

    In Supabase, call this SQL function:

        SELECT isbn13, title, author, cover_file,
               (combined_vec <=> $1::halfvec) AS distance,
               source_count
        FROM   book_embeddings
        ORDER  BY combined_vec <=> $1::halfvec
        LIMIT  $2;

    Or via Supabase Python client:
        client.rpc('match_books', payload).execute()
    where the RPC function wraps the query above.
    """
    vec = taste.query_vector()
    if vec is None:
        return {'cold_start': True, 'top_k': top_k}

    return {
        'cold_start':    False,
        'query_vector':  vec.tolist(),
        'top_k':         top_k,
    }
