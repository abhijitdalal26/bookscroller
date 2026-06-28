"""
Resize all covers to 600×900 px (Lanczos) before Supabase Storage upload.

Current covers are 300–500px wide. TikTok full-screen will upscale 2–3×,
which looks blurry. Lanczos upscale to 600×900 gives Android Glide a
reasonable source to stretch to ~1080×1620 without heavy artefacts.

Run on Kaggle GPU (after embedding generation):
    python resize_covers.py --input /kaggle/working/covers --output /kaggle/working/covers_resized

Run locally (test):
    python resize_covers.py --test
"""
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

TARGET_W = 600
TARGET_H = 900
QUALITY  = 90      # JPEG quality for resized output
WORKERS  = 8       # parallel threads


def resize_one(src: Path, dst: Path) -> tuple[str, bool]:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert('RGB')
        # Lanczos upscale (or downscale if somehow larger)
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        img.save(dst, 'JPEG', quality=QUALITY, optimize=True)
        return str(src.name), True
    except Exception as e:
        return str(src.name), False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  default=None, help='Source covers directory')
    p.add_argument('--output', default=None, help='Output directory for resized covers')
    p.add_argument('--test',   action='store_true', help='Resize 50 covers locally')
    p.add_argument('--workers', type=int, default=WORKERS)
    return p.parse_args()


def main():
    args = parse_args()

    if args.test:
        import config
        src_dir = config.COVERS_DIR
        dst_dir = Path('output_test/covers_resized')
        max_files = 50
    else:
        if not args.input or not args.output:
            print("Provide --input and --output, or use --test")
            return
        src_dir   = Path(args.input)
        dst_dir   = Path(args.output)
        max_files = None

    covers = list(src_dir.glob('*.jpg'))
    if max_files:
        covers = covers[:max_files]

    print(f"Resizing {len(covers)} covers -> {TARGET_W}x{TARGET_H} px")
    print(f"  Source:  {src_dir}")
    print(f"  Output:  {dst_dir}")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(resize_one, f, dst_dir / f.name): f
            for f in covers
        }
        for fut in tqdm(as_completed(futures), total=len(covers), unit='img'):
            name, ok = fut.result()
            if not ok:
                failed.append(name)

    success = len(covers) - len(failed)
    size_mb = sum(f.stat().st_size for f in dst_dir.glob('*.jpg')) / 1e6
    print(f"\nDone: {success}/{len(covers)} resized  ({size_mb:.0f} MB in {dst_dir})")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]}")


if __name__ == '__main__':
    main()
