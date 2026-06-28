"""
Kaggle one-shot setup + run script.

Paste this into a Kaggle notebook code cell and run it.
It handles Drive mount, data extraction, repo clone, install, and embedding run.

Usage in Kaggle:
    exec(open('kaggle_setup.py').read())
OR just paste the whole file into a notebook cell.
"""
import os
import subprocess
import sys
from pathlib import Path

DRIVE_MOUNT = '/content/drive'
DRIVE_ZIP   = '/content/drive/MyDrive/BookScroller/bookscroller_data.zip'
WORKING_DIR = Path('/kaggle/working')
REPO_URL    = 'https://github.com/abhijitdalal26/bookscroller.git'
REPO_DIR    = WORKING_DIR / 'bookscroller'


def run(cmd: str, **kwargs):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")


def step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ── Step 1: Mount Google Drive ───────────────────────────────────
step("1 / 5  Mounting Google Drive")
try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT)
    print(f"Drive mounted at {DRIVE_MOUNT}")
except Exception as e:
    print(f"Drive mount skipped ({e}) — assuming already mounted or data copied manually")


# ── Step 2: Extract data ──────────────────────────────────────────
step("2 / 5  Extracting bookscroller_data.zip")
covers_dir = WORKING_DIR / 'covers'
db_path    = WORKING_DIR / 'catalog.db'

if not db_path.exists():
    if not Path(DRIVE_ZIP).exists():
        raise FileNotFoundError(
            f"Zip not found at {DRIVE_ZIP}\n"
            "Make sure bookscroller_data.zip is in your Drive at:\n"
            "  MyDrive/BookScroller/bookscroller_data.zip"
        )
    run(f"cp '{DRIVE_ZIP}' '{WORKING_DIR}/bookscroller_data.zip'")
    run(f"unzip -q '{WORKING_DIR}/bookscroller_data.zip' -d '{WORKING_DIR}'")
    run(f"rm '{WORKING_DIR}/bookscroller_data.zip'")
    print(f"Covers: {len(list(covers_dir.glob('*.jpg')))} files")
    print(f"DB:     {db_path.stat().st_size / 1e6:.1f} MB")
else:
    print("Data already extracted — skipping unzip")


# ── Step 3: Clone / update repo ──────────────────────────────────
step("3 / 5  Cloning bookscroller repo")
if not REPO_DIR.exists():
    run(f"git clone {REPO_URL} '{REPO_DIR}'")
else:
    run(f"git -C '{REPO_DIR}' pull")


# ── Step 4: Install dependencies ─────────────────────────────────
step("4 / 5  Installing dependencies")
run(f"pip install -q -r '{REPO_DIR}/recommandation/requirements.txt'")


# ── Step 5: Run embedding pipeline ───────────────────────────────
step("5 / 5  Running embedding pipeline")

output_dir = '/content/drive/MyDrive/BookScroller/embeddings'
script     = str(REPO_DIR / 'recommandation' / 'main.py')

# Change to repo dir so relative imports work
os.chdir(str(REPO_DIR / 'recommandation'))
sys.path.insert(0, str(REPO_DIR / 'recommandation'))

run(f"python '{script}' --output '{output_dir}'")

print("\nAll done! Embeddings saved to Google Drive.")
print(f"  {output_dir}/clip_embeddings.npy")
print(f"  {output_dir}/text_embeddings.npy")
print(f"  {output_dir}/combined_embeddings.npy")
print(f"  {output_dir}/catalog_index.json")
