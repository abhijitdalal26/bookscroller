"""
Combine CLIP + text embeddings and save outputs.

combined_vec = concat(clip * 0.45, text * 0.55)  → 1152-dim float16
Both inputs must already be L2-normalized unit vectors.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from config import CLIP_WEIGHT, TEXT_WEIGHT, OUTPUT_DIR


def combine_and_save(
    df: pd.DataFrame,
    clip_emb: np.ndarray,
    text_emb: np.ndarray,
    output_dir: Path | None = None,
) -> np.ndarray:
    """
    Weighted concat → float16.
    Saves:
        clip_embeddings.npy     (N, 768)  float16 — raw, for future retuning
        text_embeddings.npy     (N, 384)  float16 — raw
        combined_embeddings.npy (N, 1152) float16 — used for pgvector ANN search
        catalog_index.json              — id→metadata map for Supabase upload
    Returns combined array.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert clip_emb.shape[0] == text_emb.shape[0] == len(df), \
        "Embedding row count does not match catalog row count"

    combined = np.concatenate([
        clip_emb  * CLIP_WEIGHT,
        text_emb  * TEXT_WEIGHT,
    ], axis=1).astype(np.float16)

    np.save(output_dir / 'clip_embeddings.npy',     clip_emb.astype(np.float16))
    np.save(output_dir / 'text_embeddings.npy',     text_emb.astype(np.float16))
    np.save(output_dir / 'combined_embeddings.npy', combined)

    # Catalog index — minimal metadata for Supabase upload
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append({
            'idx':          i,
            'isbn13':       row['isbn13'],
            'title':        row['title'],
            'author':       row['author'],
            'genres':       row.get('genres') or '',
            'avg_rating':   float(row['avg_rating']) if pd.notna(row.get('avg_rating')) else None,
            'source_count': int(row['source_count']),
            'cover_file':   row['cover_file'],
            'src':          row['src'],
        })

    with open(output_dir / 'catalog_index.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  clip_embeddings.npy     {clip_emb.shape}  "
          f"{clip_emb.astype(np.float16).nbytes / 1e6:.1f} MB")
    print(f"  text_embeddings.npy     {text_emb.shape}  "
          f"{text_emb.astype(np.float16).nbytes / 1e6:.1f} MB")
    print(f"  combined_embeddings.npy {combined.shape}  "
          f"{combined.nbytes / 1e6:.1f} MB")
    print(f"  catalog_index.json      {len(records)} records")

    return combined
