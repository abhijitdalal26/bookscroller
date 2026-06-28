"""
Text embedding generation.
Model: all-MiniLM-L6-v2 → 384-dim unit vectors.

Text input per book:
    "{title} by {author}. Genres: {genres}. Series: {series}. {description[:500]}"
"""
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import TEXT_MODEL, TEXT_BATCH_SIZE


def _parse_genres(raw) -> str:
    """Normalize genres — stored as JSON arrays or plain strings."""
    if not raw or raw in ('null', 'None', ''):
        return ''
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, list):
            return ', '.join(str(g) for g in parsed[:5])
        return str(parsed)
    except (json.JSONDecodeError, TypeError):
        return str(raw)[:100]


def build_text(row: pd.Series) -> str:
    """Build a single text string from a book's metadata fields."""
    parts = [f"{row['title']} by {row['author']}."]

    genres = _parse_genres(row.get('genres'))
    if genres:
        parts.append(f"Genres: {genres}.")

    if row.get('series_name'):
        parts.append(f"Series: {row['series_name']}.")

    desc = str(row.get('description') or '')
    if len(desc) > 10:
        parts.append(desc[:500])

    return ' '.join(parts)


def generate_text_embeddings(
    df: pd.DataFrame,
    batch_size: int = TEXT_BATCH_SIZE,
) -> np.ndarray:
    """
    Returns float32 array of shape (N, 384), L2-normalized.
    """
    print(f"Loading text model [{TEXT_MODEL}]...")
    model = SentenceTransformer(TEXT_MODEL)

    texts = [build_text(row) for _, row in df.iterrows()]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
