"""
Environment-aware config.
Auto-detects Colab, Kaggle, or local and sets paths accordingly.
"""
import os
from pathlib import Path

IS_COLAB  = os.path.exists('/content') and os.path.exists('/usr/local/lib/python3.12/dist-packages/google/colab')
IS_KAGGLE = os.path.exists('/kaggle/working') and not IS_COLAB

if IS_COLAB:
    COVERS_DIR   = Path('/content/covers')
    DB_PATH      = Path('/content/catalog.db')
    OUTPUT_DIR   = Path('/content/embeddings')           # local — fast writes, no network lag
    DRIVE_OUTPUT = Path('/content/drive/MyDrive/BookScroller/embeddings')  # copied here at end
elif IS_KAGGLE:
    COVERS_DIR   = Path('/kaggle/working/covers')
    DB_PATH      = Path('/kaggle/working/catalog.db')
    OUTPUT_DIR   = Path('/kaggle/working/embeddings')
    DRIVE_OUTPUT = None
else:
    _BASE        = Path(__file__).parent
    COVERS_DIR   = _BASE / 'data/training_final/covers'
    DB_PATH      = _BASE / 'data/training_final/catalog.db'
    OUTPUT_DIR   = _BASE / 'output'
    DRIVE_OUTPUT = None

# Models
CLIP_MODEL = 'openai/clip-vit-large-patch14'  # ViT-L/14 → 768-dim
TEXT_MODEL = 'all-MiniLM-L6-v2'               # → 384-dim

# Dimensions
CLIP_DIM     = 768
TEXT_DIM     = 384
COMBINED_DIM = CLIP_DIM + TEXT_DIM  # 1152

# Embedding weights (tuned after user data; start 45/55 favouring text)
CLIP_WEIGHT = 0.45
TEXT_WEIGHT = 0.55

# Batch sizes — GPU-safe defaults, auto-halved if OOM
CLIP_BATCH_SIZE = 256
TEXT_BATCH_SIZE = 512

# Local test sample size
TEST_SAMPLE = 200
