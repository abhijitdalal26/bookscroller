"""
CLIP visual embedding generation.
Model: openai/clip-vit-large-patch14 (ViT-L/14) → 768-dim unit vectors.
Covers are loaded as RGB, CLIP internally resizes to 224×224.

Checkpoint: saves every CHECKPOINT_EVERY batches so Kaggle session
timeouts don't lose progress. Resume is automatic on next run.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from config import CLIP_MODEL, CLIP_BATCH_SIZE

CHECKPOINT_EVERY = 50   # save partial results every N batches (~12K images at batch=256)
CHECKPOINT_FILE  = 'clip_checkpoint.npy'
CHECKPOINT_IDX   = 'clip_checkpoint_idx.txt'


def _load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224), (128, 128, 128))


def _load_checkpoint(output_dir: Path) -> tuple[np.ndarray | None, int]:
    """Return (partial_embeddings, start_index) or (None, 0) if no checkpoint."""
    ckpt_emb = output_dir / CHECKPOINT_FILE
    ckpt_idx = output_dir / CHECKPOINT_IDX
    if ckpt_emb.exists() and ckpt_idx.exists():
        start = int(ckpt_idx.read_text().strip())
        emb = np.load(str(ckpt_emb))
        print(f"Resuming CLIP from checkpoint: {start} images already done")
        return emb, start
    return None, 0


def _save_checkpoint(output_dir: Path, embeddings: list[np.ndarray], count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = np.vstack(embeddings)
    np.save(str(output_dir / CHECKPOINT_FILE), arr)
    (output_dir / CHECKPOINT_IDX).write_text(str(count))


def generate_clip_embeddings(
    cover_paths: list[str],
    batch_size: int = CLIP_BATCH_SIZE,
    device: str | None = None,
    output_dir: Path | None = None,
) -> np.ndarray:
    """
    Returns float32 array of shape (N, 768), L2-normalized.
    - Auto-detects GPU
    - Halves batch_size on CUDA OOM and retries
    - Saves checkpoint every CHECKPOINT_EVERY batches for resume on timeout
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if output_dir is None:
        output_dir = Path('.')

    print(f"Loading CLIP [{CLIP_MODEL}] on {device}...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    # Resume from checkpoint if available
    prior_emb, start_i = _load_checkpoint(output_dir)
    all_embeddings = [prior_emb] if prior_emb is not None else []
    i = start_i
    batches_since_ckpt = 0

    with torch.no_grad():
        pbar = tqdm(total=len(cover_paths), initial=start_i,
                    desc='CLIP embeddings', unit='img')
        while i < len(cover_paths):
            batch_paths = cover_paths[i : i + batch_size]
            images = [_load_image(p) for p in batch_paths]

            try:
                pixel_values = processor(
                    images=images, return_tensors='pt'
                )['pixel_values'].to(device)
                # vision_model + projection works across all transformers versions
                vision_out = model.vision_model(pixel_values=pixel_values)
                feats = model.visual_projection(vision_out.pooler_output)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
                all_embeddings.append(feats.cpu().float().numpy())
                pbar.update(len(batch_paths))
                i += batch_size
                batches_since_ckpt += 1

                if batches_since_ckpt >= CHECKPOINT_EVERY:
                    _save_checkpoint(output_dir, all_embeddings, i)
                    batches_since_ckpt = 0

            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 8:
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    print(f"\nOOM — reducing batch size to {batch_size}")
                else:
                    all_embeddings.append(
                        np.zeros((len(batch_paths), 768), dtype=np.float32)
                    )
                    pbar.update(len(batch_paths))
                    i += batch_size

        pbar.close()

    result = np.vstack(all_embeddings)

    # Clean up checkpoint files on success
    for f in [output_dir / CHECKPOINT_FILE, output_dir / CHECKPOINT_IDX]:
        if f.exists():
            f.unlink()

    return result
