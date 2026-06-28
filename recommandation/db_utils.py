"""
Load and deduplicate the catalog DB.

All four tables share a cover_file column that links each row to its jpg.
Books appear in up to 3 tables (books + nyt_books + scraped_books) — we
deduplicate by isbn13 and record source_count as a popularity signal.
Manga has its own ID space and is never deduplicated against books.
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path


def _load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read one table into a normalized DataFrame."""
    if table == 'books':
        return pd.read_sql_query("""
            SELECT
                isbn13,
                title,
                author,
                description,
                genres,
                series_name,
                CAST(avg_rating    AS REAL)    AS avg_rating,
                CAST(ratings_count AS INTEGER) AS ratings_count,
                cover_file,
                'books' AS src
            FROM books
            WHERE isbn13 IS NOT NULL AND cover_file IS NOT NULL
        """, conn)

    if table == 'nyt_books':
        return pd.read_sql_query("""
            SELECT
                isbn13,
                title,
                author,
                description,
                NULL AS genres,
                NULL AS series_name,
                NULL AS avg_rating,
                NULL AS ratings_count,
                cover_file,
                'nyt' AS src
            FROM nyt_books
            WHERE isbn13 IS NOT NULL AND cover_file IS NOT NULL
        """, conn)

    if table == 'scraped_books':
        return pd.read_sql_query("""
            SELECT
                isbn13,
                title,
                author,
                description,
                genres,
                NULL AS series_name,
                CAST(avg_rating AS REAL) AS avg_rating,
                NULL AS ratings_count,
                cover_file,
                'scraped' AS src
            FROM scraped_books
            WHERE isbn13 IS NOT NULL AND cover_file IS NOT NULL
        """, conn)

    raise ValueError(f"Unknown table: {table}")


def _load_manga(conn: sqlite3.Connection) -> pd.DataFrame:
    """Manga gets its own row; no isbn13 — use anilist_id as key."""
    df = pd.read_sql_query("""
        SELECT
            CAST(anilist_id AS TEXT)               AS isbn13,
            COALESCE(title_english, title_romaji)  AS title,
            author,
            description,
            genres,
            NULL AS series_name,
            NULL AS avg_rating,
            NULL AS ratings_count,
            cover_file,
            'manga' AS src
        FROM manga
        WHERE is_adult = 0 AND cover_file IS NOT NULL
    """, conn)
    df['source_count'] = 1
    return df


def load_catalog(
    db_path: str | Path,
    covers_dir: str | Path,
    test_n: int | None = None,
) -> pd.DataFrame:
    """
    Return a deduplicated DataFrame ready for embedding.

    Columns:
        isbn13, title, author, description, genres, series_name,
        avg_rating, ratings_count, cover_file, cover_path, source_count, src
    """
    covers_dir = Path(covers_dir)
    conn = sqlite3.connect(db_path)

    # Load all book tables
    books   = _load_table(conn, 'books')
    nyt     = _load_table(conn, 'nyt_books')
    scraped = _load_table(conn, 'scraped_books')
    manga   = _load_manga(conn)
    conn.close()

    # Merge book tables
    all_books = pd.concat([books, nyt, scraped], ignore_index=True)

    # Count sources per isbn13 (books + nyt + scraped = max 3)
    source_count = (
        all_books.groupby('isbn13')['src']
        .nunique()
        .rename('source_count')
        .reset_index()
    )

    # Deduplicate: books > scraped > nyt (books has richest metadata)
    _priority = {'books': 0, 'scraped': 1, 'nyt': 2}
    all_books['_p'] = all_books['src'].map(_priority)
    deduped = (
        all_books
        .sort_values('_p')
        .drop_duplicates(subset='isbn13', keep='first')
        .drop(columns='_p')
        .merge(source_count, on='isbn13')
        .reset_index(drop=True)
    )

    # Combine books + manga
    final = pd.concat([deduped, manga], ignore_index=True)

    # Verify cover file actually exists on disk (guards against DB/disk mismatch)
    final['cover_path'] = final['cover_file'].apply(
        lambda f: str(covers_dir / f) if f and (covers_dir / f).exists() else None
    )
    final = final[final['cover_path'].notna()].reset_index(drop=True)

    if test_n:
        final = final.sample(
            min(test_n, len(final)), random_state=42
        ).reset_index(drop=True)

    print(f"Catalog loaded: {len(final)} unique items "
          f"({final['source_count'].gt(1).sum()} multi-source, "
          f"{(final['src']=='manga').sum()} manga)")
    return final
