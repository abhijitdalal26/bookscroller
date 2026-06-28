# BookScroller — How Recommendation Works

## The Big Picture

User opens app → sees book covers full-screen (TikTok-style scroll) → their interactions (dwell, save, skip) update a taste vector → taste vector queries pgvector → next 20 books returned.

No user history needed at start. Gets smarter with every scroll.

---

## Step 1 — Embeddings (done offline, once)

Every book gets a single 1152-dim vector that captures both its visual style and its content.

```
Cover image   → CLIP ViT-L/14      → 768-dim  (what the cover looks like)
Title+Author+Genres+Description → MiniLM-L6-v2 → 384-dim  (what the book is about)

combined_vec = concat(clip × 0.45, text × 0.55)  → 1152-dim float16
```

These live in Supabase pgvector table `book_embeddings`, one row per book.

---

## Step 2 — Cold Start (new user, no history)

User has no taste vector yet. Show books by popularity:

```sql
SELECT * FROM book_embeddings
ORDER BY source_count DESC, avg_rating DESC
LIMIT 20
```

`source_count` = how many sources the book appears in (books + nyt + scraped). Max 3. Books appearing in all 3 sources are high-quality/popular — show these first.

---

## Step 3 — Taste Vector (builds up as user scrolls)

Every interaction updates the taste vector in real time:

```
save / add to list   → weight +1.00   (strongest signal)
tap for details      → weight +0.50
scroll back          → weight +0.40
dwell 3–10 seconds   → weight +0.20
dwell 1–3 seconds    → weight +0.05
skip < 1 second      → weight -0.30   (negative signal)
```

The taste vector is a weighted sum of all interacted book embeddings, with a decay of 0.95 applied each update (recent interactions matter more).

Code: `taste_vector.py` → `TasteVector` class.

---

## Step 4 — ANN Search (every 20 books)

When the user reaches book #15 in the buffer, background-fetch the next 20:

```sql
SELECT isbn13, title, cover_file, source_count,
       (combined_vec <=> '[...taste vector...]'::halfvec) AS distance
FROM book_embeddings
ORDER BY combined_vec <=> '[...taste vector...]'::halfvec
LIMIT 40
```

Fetch 40, re-rank by distance + source_count boost, return top 20:

```python
final_score = cosine_similarity + (0.05 * source_count)
```

---

## Step 5 — Cover Display

Covers are stored in Supabase Storage as `{isbn13}.jpg` (manga: `manga_{anilist_id}.jpg`).
Resized to 600×900 px before upload (`resize_covers.py`).
Android Glide stretches to full screen — no extra work needed.

For books missing a cover (runtime fetch): try Open Library first, then Google Books API as fallback.

---

## Supabase Schema

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table
CREATE TABLE book_embeddings (
    id            SERIAL PRIMARY KEY,
    isbn13        TEXT NOT NULL UNIQUE,
    title         TEXT,
    author        TEXT,
    genres        TEXT,
    avg_rating    FLOAT,
    source_count  INTEGER DEFAULT 1,   -- 1/2/3, used for cold-start + re-ranking
    cover_file    TEXT,                -- filename in Supabase Storage
    src           TEXT,                -- books / nyt / scraped / manga
    combined_vec  halfvec(1152)        -- queried for ANN search
);

-- HNSW index for fast ANN search
CREATE INDEX ON book_embeddings
    USING hnsw (combined_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

---

## Files in This Folder

| File | When it runs | What it does |
|---|---|---|
| `main.py` | Kaggle (once) | Runs full embedding pipeline |
| `config.py` | Always | Sets paths for Colab / Kaggle / local |
| `db_utils.py` | Kaggle (once) | Loads + deduplicates catalog DB |
| `embed_clip.py` | Kaggle (once) | CLIP visual embeddings |
| `embed_text.py` | Kaggle (once) | Text embeddings |
| `combine.py` | Kaggle (once) | Merges embeddings, saves .npy files |
| `supabase_upload.py` | Once after Kaggle | Uploads embeddings to Supabase |
| `resize_covers.py` | Once after Kaggle | Resizes covers to 600×900 for upload |
| `taste_vector.py` | Android app (runtime) | Updates user taste vector per interaction |
| `kaggle_setup.py` | Kaggle (once) | One-shot setup script for Kaggle |

---

## What's Next (for the next agent)

1. **Kaggle run** — run `main.py` on Colab/Kaggle GPU → embeddings saved to Drive
2. **Supabase setup** — create table + index using SQL above, then run `supabase_upload.py`
3. **Cover upload** — run `resize_covers.py` then upload `covers_resized/` to Supabase Storage
4. **Android app** — TikTok-style scroll UI, call Supabase ANN endpoint, update taste vector per interaction using `taste_vector.py` logic
