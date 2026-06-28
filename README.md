# BookScroller — Embedding Pipeline

TikTok-style book cover discovery app. This repo contains the embedding generation pipeline that powers the recommendation engine.

## How it works

Each book gets a **1152-dim combined embedding**:
- `CLIP ViT-L/14` (768-dim) — visual style and genre from the cover image
- `all-MiniLM-L6-v2` (384-dim) — themes and tone from title + author + genres + description
- Weighted concat: `clip × 0.45 + text × 0.55`

At runtime, user interactions (dwell time, saves, skips) update a taste vector in the same 1152-dim space. Supabase pgvector ANN search returns the next 20 books.

## Files

```
recommandation/
├── main.py          # Entry point — run this
├── config.py        # Paths + model config (auto-detects Kaggle vs local)
├── db_utils.py      # DB loading and isbn13 deduplication
├── embed_clip.py    # CLIP visual embedding generation
├── embed_text.py    # Text embedding generation
├── combine.py       # Weighted concat + save outputs
└── requirements.txt
```

## Run on Kaggle (full pipeline)

**Step 1 — Mount Drive and unzip data**
```python
from google.colab import drive
drive.mount('/kaggle/drive')

import subprocess
subprocess.run(["cp", "/kaggle/drive/MyDrive/BookScroller/bookscroller_data.zip", "/kaggle/working/"])
subprocess.run(["unzip", "-q", "/kaggle/working/bookscroller_data.zip", "-d", "/kaggle/working/"])
subprocess.run(["rm", "/kaggle/working/bookscroller_data.zip"])
```

**Step 2 — Clone repo and install**
```bash
git clone https://github.com/abhijitdalal26/bookscroller.git
cd bookscroller
pip install -r recommandation/requirements.txt
```

**Step 3 — Run**
```bash
python recommandation/main.py
```

Outputs saved to `/kaggle/drive/MyDrive/BookScroller/embeddings/`:
- `clip_embeddings.npy`      — (N, 768) float16
- `text_embeddings.npy`      — (N, 384) float16
- `combined_embeddings.npy`  — (N, 1152) float16  ← upload this to Supabase pgvector
- `catalog_index.json`       — metadata index for Supabase upload

## Run locally (test mode)

```bash
conda create -n bookscroller-env python=3.11 -y
conda activate bookscroller-env
pip install -r recommandation/requirements.txt

# Test with 200 samples
python recommandation/main.py --test

# Text embeddings only (no GPU needed)
python recommandation/main.py --test --text-only
```

## Catalog

106,558 rows across 4 tables → ~86K unique books after deduplication by `isbn13`.
Books appearing in multiple sources (books + nyt + scraped) get a `source_count` popularity boost applied at re-ranking time.
