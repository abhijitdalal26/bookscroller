"""
Upload embeddings + metadata to Supabase pgvector.

Run after main.py has generated embeddings (on Kaggle or locally).
Requires SUPABASE_URL and SUPABASE_KEY environment variables.

Setup in Supabase SQL editor before running:
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE book_embeddings (
        id            SERIAL PRIMARY KEY,
        isbn13        TEXT NOT NULL UNIQUE,
        title         TEXT,
        author        TEXT,
        genres        TEXT,
        avg_rating    FLOAT,
        source_count  INTEGER DEFAULT 1,
        cover_file    TEXT,
        src           TEXT,
        combined_vec  halfvec(1152)
    );

    CREATE INDEX ON book_embeddings
        USING hnsw (combined_vec vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);

Usage:
    export SUPABASE_URL=https://xxxx.supabase.co
    export SUPABASE_KEY=your-service-role-key
    python supabase_upload.py --embeddings /path/to/embeddings/dir
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    print("Install supabase: pip install supabase")
    sys.exit(1)


BATCH_SIZE = 500   # rows per upsert batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', required=True,
                   help='Directory containing combined_embeddings.npy + catalog_index.json')
    p.add_argument('--table', default='book_embeddings')
    p.add_argument('--dry-run', action='store_true',
                   help='Validate data without uploading')
    return p.parse_args()


def load_embeddings(emb_dir: Path):
    combined = np.load(emb_dir / 'combined_embeddings.npy').astype(np.float32)
    with open(emb_dir / 'catalog_index.json', encoding='utf-8') as f:
        catalog = json.load(f)
    assert len(combined) == len(catalog), \
        f"Mismatch: {len(combined)} embeddings vs {len(catalog)} catalog rows"
    return combined, catalog


def vec_to_list(v: np.ndarray) -> list[float]:
    return v.tolist()


def upload(client: Client, table: str, combined: np.ndarray, catalog: list[dict]):
    total   = len(catalog)
    success = 0
    errors  = 0

    for start in range(0, total, BATCH_SIZE):
        batch_cat = catalog[start : start + BATCH_SIZE]
        batch_emb = combined[start : start + BATCH_SIZE]

        rows = []
        for meta, vec in zip(batch_cat, batch_emb):
            rows.append({
                'isbn13':       meta['isbn13'],
                'title':        meta.get('title'),
                'author':       meta.get('author'),
                'genres':       meta.get('genres'),
                'avg_rating':   meta.get('avg_rating'),
                'source_count': meta.get('source_count', 1),
                'cover_file':   meta.get('cover_file'),
                'src':          meta.get('src'),
                'combined_vec': vec_to_list(vec),
            })

        try:
            client.table(table).upsert(rows, on_conflict='isbn13').execute()
            success += len(rows)
            pct = 100 * success / total
            print(f"  Uploaded {success}/{total} ({pct:.1f}%)", end='\r')
        except Exception as e:
            errors += len(rows)
            print(f"\n  ERROR at batch {start}: {e}")

    print(f"\nDone: {success} uploaded, {errors} errors")


def main():
    args   = parse_args()
    emb_dir = Path(args.embeddings)

    if not emb_dir.exists():
        print(f"Embeddings dir not found: {emb_dir}")
        sys.exit(1)

    print(f"Loading embeddings from {emb_dir}...")
    combined, catalog = load_embeddings(emb_dir)
    print(f"  {len(catalog)} records, embedding shape {combined.shape}")

    if args.dry_run:
        print("Dry run — first record preview:")
        print(json.dumps({k: v for k, v in catalog[0].items()}, indent=2))
        print("combined_vec[:5]:", combined[0, :5].tolist())
        print("Dry run complete — no data uploaded")
        return

    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)

    print(f"Connecting to Supabase...")
    client: Client = create_client(url, key)

    print(f"Uploading {len(catalog)} rows to '{args.table}' in batches of {BATCH_SIZE}...")
    upload(client, args.table, combined, catalog)


if __name__ == '__main__':
    main()
