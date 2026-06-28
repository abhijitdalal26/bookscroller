#!/usr/bin/env python3
"""
BookScroller — Embedding Generation Pipeline
============================================
Generates CLIP (visual) + text embeddings for all book covers,
combines them into a single taste-queryable vector, and saves outputs
ready for Supabase pgvector upload.

Usage
-----
  Full run (Kaggle GPU):       python main.py
  Local test (200 samples):    python main.py --test
  CLIP only:                   python main.py --clip-only
  Text only:                   python main.py --text-only
  Custom output dir:           python main.py --output /path/to/dir

On Kaggle, mount Google Drive first:
    from google.colab import drive
    drive.mount('/kaggle/drive')
    !cp /kaggle/drive/MyDrive/BookScroller/bookscroller_data.zip /kaggle/working/
    !unzip -q /kaggle/working/bookscroller_data.zip -d /kaggle/working/
    !rm /kaggle/working/bookscroller_data.zip
"""
import argparse
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='BookScroller embedding pipeline')
    p.add_argument('--test',      action='store_true',
                   help='Run on 200 samples for local testing')
    p.add_argument('--clip-only', action='store_true',
                   help='Generate CLIP embeddings only')
    p.add_argument('--text-only', action='store_true',
                   help='Generate text embeddings only')
    p.add_argument('--output',    type=str, default=None,
                   help='Override output directory')
    p.add_argument('--batch-clip', type=int, default=None,
                   help='Override CLIP batch size')
    p.add_argument('--batch-text', type=int, default=None,
                   help='Override text batch size')
    return p.parse_args()


def check_paths(config) -> bool:
    ok = True
    if not config.DB_PATH.exists():
        print(f"ERROR: DB not found at {config.DB_PATH}")
        ok = False
    if not config.COVERS_DIR.exists():
        print(f"ERROR: Covers dir not found at {config.COVERS_DIR}")
        ok = False
    return ok


def main():
    args = parse_args()

    import config
    from db_utils    import load_catalog
    from embed_clip  import generate_clip_embeddings
    from embed_text  import generate_text_embeddings
    from combine     import combine_and_save

    output_dir = Path(args.output) if args.output else config.OUTPUT_DIR
    test_n     = config.TEST_SAMPLE if args.test else None

    if args.batch_clip: config.CLIP_BATCH_SIZE = args.batch_clip
    if args.batch_text: config.TEXT_BATCH_SIZE = args.batch_text

    print("=" * 60)
    print("  BookScroller — Embedding Pipeline")
    print("=" * 60)
    print(f"  DB:       {config.DB_PATH}")
    print(f"  Covers:   {config.COVERS_DIR}")
    print(f"  Output:   {output_dir}")
    print(f"  CLIP:     {config.CLIP_MODEL}  ({config.CLIP_DIM}-dim)")
    print(f"  Text:     {config.TEXT_MODEL}  ({config.TEXT_DIM}-dim)")
    print(f"  Combined: {config.COMBINED_DIM}-dim  "
          f"(clip×{config.CLIP_WEIGHT} + text×{config.TEXT_WEIGHT})")
    print(f"  Mode:     {'TEST (' + str(test_n) + ' samples)' if test_n else 'FULL'}")
    print("=" * 60)

    if not check_paths(config):
        sys.exit(1)

    t_start = time.time()

    # ── 1. Load catalog ──────────────────────────────────────────────
    print("\n[1/4] Loading catalog...")
    df = load_catalog(config.DB_PATH, config.COVERS_DIR, test_n=test_n)

    # ── 2. CLIP embeddings ───────────────────────────────────────────
    clip_emb = None
    if not args.text_only:
        print(f"\n[2/4] CLIP embeddings  ({len(df)} covers)...")
        t = time.time()
        clip_emb = generate_clip_embeddings(
            df['cover_path'].tolist(),
            batch_size=config.CLIP_BATCH_SIZE,
            output_dir=output_dir,
        )
        print(f"  Done in {(time.time()-t)/60:.1f} min  shape={clip_emb.shape}")

    # ── 3. Text embeddings ───────────────────────────────────────────
    text_emb = None
    if not args.clip_only:
        print(f"\n[3/4] Text embeddings  ({len(df)} books)...")
        t = time.time()
        text_emb = generate_text_embeddings(
            df,
            batch_size=config.TEXT_BATCH_SIZE,
        )
        print(f"  Done in {(time.time()-t)/60:.1f} min  shape={text_emb.shape}")

    # ── 4. Combine & save ────────────────────────────────────────────
    if clip_emb is not None and text_emb is not None:
        print("\n[4/4] Combining and saving...")
        combine_and_save(df, clip_emb, text_emb, output_dir)

    elif clip_emb is not None:
        import numpy as np
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'clip_embeddings.npy', clip_emb.astype('float16'))
        print(f"[4/4] Saved clip_embeddings.npy → {output_dir}")

    elif text_emb is not None:
        import numpy as np
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / 'text_embeddings.npy', text_emb.astype('float16'))
        print(f"[4/4] Saved text_embeddings.npy → {output_dir}")

    else:
        print("Nothing to save (both --clip-only and --text-only? Pick one.)")

    # ── 5. Copy to Drive (Colab only) ───────────────────────────────
    if config.IS_COLAB and config.DRIVE_OUTPUT:
        import shutil
        drive_out = config.DRIVE_OUTPUT
        drive_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[5/5] Copying outputs to Drive ({drive_out})...")
        for f in output_dir.glob('*'):
            shutil.copy2(f, drive_out / f.name)
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
        print("Drive copy done.")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("Done.")


if __name__ == '__main__':
    main()
