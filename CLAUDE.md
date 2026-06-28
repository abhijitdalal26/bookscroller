# BookScroll — Project Overview

Book cover discovery app ("Tinder for Books"). Users swipe through covers, get recommendations.

## Project Structure

```
TinderforBooks/
├── recommandation/
│   └── data/
│       ├── catalog.db          # SQLite — all metadata (books, manga, nyt, scraped)
│       ├── covers/             # .jpg cover images (book: {isbn13}.jpg, manga: manga_{id}.jpg)
│       └── pipeline/           # Data pipeline scripts (run in order)
│           ├── 01_parse_goodreads_dump.py
│           ├── 01b_add_series_info.py
│           ├── 02_nyt_bestsellers.py
│           ├── 03_anilist_manga.py
│           ├── 03b_anilist_adult_manga.py
│           ├── 04_merge_catalog.py
│           ├── 05_download_covers.py      # Pass 1 GR CDN + Pass 2 OL (VPN needed)
│           ├── 05b_nyt_covers.py
│           └── 05d_google_books_covers.py
│       ├── Goodreads_Choice_Awards_(2011–2025)/
│       │   └── goodreads_choice_awards_scraper.py
│       └── Goodreads_Best_Books/
│           └── goodreads_best_books_scraper.py
└── research/                   # App research notes
```

## Catalog Database (catalog.db)

### Tables

| Table | Rows | Description |
|---|---|---|
| `books` | 148,183 | Goodreads dump (ratings ≥ 500, English) |
| `nyt_books` | 21,493 | NYT Bestseller lists 2008–2026 |
| `scraped_books` | 21,178 | GR Choice Awards 2011–2025 + GR Best Books top-5000 |
| `manga` | 6,000 | 5,000 non-adult + 1,000 adult (is_adult=1) from AniList |

### books table columns
`book_id, title, author, isbn13, isbn10, avg_rating, ratings_count, description, genres, publisher, format, pages, pub_year, language_code, image_url, goodreads_url, source, series_name, series_position`

### manga table columns
`anilist_id, mal_id, title_english, title_romaji, title_native, author, artist, description, genres, avg_score, popularity, favourites, volumes, chapters, status, start_year, country, cover_url, source, is_adult`

## MVP vs Local

- **MVP set (for deployment):** books with `ratings_count >= 1000` (86,491) + all NYT + all scraped + non-adult manga (5,000) ≈ **~91,500 unique items**
- **Local only (not in MVP):** books with ratings 500–999 (61,692 rows in DB) + adult manga (1,000 rows, `is_adult=1`)
- Books with 500–999 ratings: full metadata in DB, covers fetched at runtime in app (not pre-downloaded)

## Cover Images

### Naming convention
- Books: `{isbn13}.jpg` (e.g. `9780439023480.jpg`)
- Books without ISBN: `goodreads_{book_id}.jpg`
- Manga: `manga_{anilist_id}.jpg`
- All in single `covers/` folder — `manga_` prefix distinguishes them

### Status (as of June 2026)
- **On disk:** ~110,000+ files, ~5.9 GB
- **Missing (need OL via VPN):** ~25,429 book covers (ratings ≥ 1,000 only)
- **Runtime fetch:** 61,692 books (500–999 ratings) — app fetches cover on demand

### Download sources used
1. **Goodreads CDN** (`images.gr-assets.com`) — old format only, parallel 16 workers, ~89K covers. New CDN (`i.gr-assets.com/compressed.photo`) is hotlink-blocked (HTTP 403).
2. **NYT CDN** (`static01.nyt.com/bestsellers/images/{isbn13}.jpg`) — ~17K covers
3. **Open Library** (`covers.openlibrary.org/b/isbn/{isbn13}-L.jpg`) — blocked on home network, needs VPN. Run `05_download_covers.py` with VPN on. ~22 hours for remaining 25K.
4. **Google Books API** — used as fallback (1,000/day limit). Covers deleted; OL will replace with better quality.

## Data Sources

| Source | Filter | Count |
|---|---|---|
| UCSD Goodreads Book Graph (2017 snapshot) | ratings ≥ 500, English | 148,183 |
| NYT Bestsellers API | All lists 2008–2026 (monthly mode) | 21,493 |
| GR Choice Awards scraper | 2011–2025, all categories | 4,145 |
| GR Best Books scraper | Top 5,000 most-voted | 4,967 |
| AniList GraphQL API | Top 5,000 by popularity, non-adult | 5,000 |
| AniList GraphQL API | Top 1,000 adult manga (`is_adult=1`) | 1,000 |

## App UX — TikTok-style vertical scroll (NOT Tinder swipe)

- Full-screen cover, scroll UP for next book (like TikTok For You Page)
- **Signals collected:** dwell time per cover (ms before scroll), scroll-back (strong positive), tap for details (moderate positive), save/add to list (strongest positive), scroll past in <1s (negative)
- **Retrieval:** buffer 20 books from pgvector, background-query next 20 at book #15, taste vector updates continuously
- Cover displayed full-screen at phone resolution (~1080×1920px) — upscaling matters

## Cover Upscaling (TODO — before uploading to Supabase)

Current covers are 300–500px wide. TikTok-style full-screen will upscale 2–3×. Tiny covers look bad at full screen.

**Recommended approach (do before Supabase upload):**

1. **Pillow Lanczos** — fast, free, decent. Upscale all to 600×900 in one batch (~1 min for 102K):
   ```python
   img = Image.open("cover.jpg").resize((600, 900), Image.LANCZOS)
   ```
2. **Real-ESRGAN** — AI super-resolution, genuine detail reconstruction, much better for small covers (200–300px range). ~15 min on GPU, ~3–4 hours on CPU. Run on covers under 400px before Lanczos pass.
3. **Android Glide** handles final stretch to screen size (no extra work needed in app).
4. **Supabase image transform API** as alternative: store originals, serve resized via CDN URL param `?width=1080&quality=85` — zero preprocessing, on-the-fly resize.

**Verdict:** Run Lanczos upscale to 600×900 on all covers before upload. Optionally run Real-ESRGAN first on covers < 400px wide if GPU is available.

## Training Data (Final State — as of June 2026)

All training data lives in `training_final/`:
- `covers/` — **85,957 cover images** (≥200px quality)
- `catalog.db` — **106,558 rows** across 4 tables, every row has `cover_file` column linking to its jpg
  - `books`: 63,619 | `nyt_books`: 21,019 | `scraped_books`: 17,286 | `manga`: 4,634

Items with metadata but no cover: `metadata_no_cover/catalog_no_cover.db` (~24K rows, runtime fetch needed)

## Next Pipeline Steps

1. ✅ Training data complete — covers + metadata synced in `training_final/`
2. **Generate CLIP embeddings** (512-dim visual) on Kaggle GPU — use covers as-is, NO resize needed for CLIP (it internally resizes to 224×224)
3. **Generate text embeddings** — sentence-transformers all-MiniLM-L6-v2 (384-dim) for descriptions
4. **Upload to Supabase pgvector** — halfvec (float16); covers to Supabase Storage
5. **Resize covers BEFORE mobile display** — Lanczos to 600×900 before Supabase upload (current covers 300–500px look blurry at full-screen phone resolution). Optionally run Real-ESRGAN first on covers <400px wide if Kaggle GPU available.
6. **Android app** — TikTok-style vertical scroll UI, dwell time tracking, runtime OL/Google Books fallback for uncached covers

## API Keys (reset after use)
- NYT Books API key used in `02_nyt_bestsellers.py` — user will reset
- Google Books API key in `05d_google_books_covers.py` — 1,000/day limit, extreme fallback only
