"""
CLIP visual embedding generation.
Model: openai/clip-vit-large-patch14 (ViT-L/14) → 768-dim unit vectors.

Uses PyTorch DataLoader with multiple worker processes so CLIPProcessor
preprocessing (resize + normalize) runs in parallel with GPU inference.
GPU stays busy continuously instead of waiting for CPU each batch.

Checkpoint: saves every CHECKPOINT_EVERY batches for resume on timeout.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import CLIP_MODEL, CLIP_BATCH_SIZE

CHECKPOINT_EVERY = 50
CHECKPOINT_FILE  = 'clip_checkpoint.npy'
CHECKPOINT_IDX   = 'clip_checkpoint_idx.txt'
NUM_WORKERS      = 4   # parallel CLIPProcessor workers


class CoverDataset(Dataset):
    def __init__(self, paths: list[str], processor: CLIPProcessor):
        self.paths     = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        # CLIPProcessor runs here — inside worker process, fully parallel
        return self.processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)


def _load_checkpoint(output_dir: Path) -> tuple[np.ndarray | None, int]:
    ckpt_emb = output_dir / CHECKPOINT_FILE
    ckpt_idx = output_dir / CHECKPOINT_IDX
    if ckpt_emb.exists() and ckpt_idx.exists():
        start = int(ckpt_idx.read_text().strip())
        emb   = np.load(str(ckpt_emb))
        print(f"Resuming CLIP from checkpoint: {start} images done")
        return emb, start
    return None, 0


def _save_checkpoint(output_dir: Path, embeddings: list[np.ndarray], count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / CHECKPOINT_FILE), np.vstack(embeddings))
    (output_dir / CHECKPOINT_IDX).write_text(str(count))


def generate_clip_embeddings(
    cover_paths: list[str],
    batch_size:  int = CLIP_BATCH_SIZE,
    device:      str | None = None,
    output_dir:  Path | None = None,
) -> np.ndarray:
    """
    Returns float32 array of shape (N, 768), L2-normalized.
    - DataLoader with NUM_WORKERS processes CLIPProcessor in parallel
    - Checkpoint/resume every CHECKPOINT_EVERY batches
    - OOM-safe: halves batch_size on CUDA OOM
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if output_dir is None:
        output_dir = Path('.')

    print(f"Loading CLIP [{CLIP_MODEL}] on {device}...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    prior_emb, start_i = _load_checkpoint(output_dir)
    all_embeddings      = [prior_emb] if prior_emb is not None else []
    batches_since_ckpt  = 0
    processed           = start_i

    remaining_paths = cover_paths[start_i:]

    dataset = CoverDataset(remaining_paths, processor)
    # pin_memory=True speeds up CPU→GPU transfer
    # persistent_workers=True avoids worker restart overhead between batches
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=(device == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2,
    )

    pbar = tqdm(total=len(cover_paths), initial=start_i,
                desc='CLIP embeddings', unit='img')

    with torch.no_grad():
        for pixel_values in loader:
            try:
                pixel_values = pixel_values.to(device, non_blocking=True)
                vision_out   = model.vision_model(pixel_values=pixel_values)
                feats        = model.visual_projection(vision_out.pooler_output)
                feats        = feats / feats.norm(dim=-1, keepdim=True)
                all_embeddings.append(feats.cpu().float().numpy())

                n = pixel_values.shape[0]
                processed          += n
                batches_since_ckpt += 1
                pbar.update(n)

                if batches_since_ckpt >= CHECKPOINT_EVERY:
                    _save_checkpoint(output_dir, all_embeddings, processed)
                    batches_since_ckpt = 0

            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 8:
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    print(f"\nOOM — reducing batch size to {batch_size}")
                    # Rebuild loader with smaller batch
                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=NUM_WORKERS,
                        pin_memory=(device == 'cuda'),
                        persistent_workers=(NUM_WORKERS > 0),
                        prefetch_factor=2,
                    )
                else:
                    pbar.update(pixel_values.shape[0])

    pbar.close()
    result = np.vstack(all_embeddings)

    for f in [output_dir / CHECKPOINT_FILE, output_dir / CHECKPOINT_IDX]:
        if f.exists():
            f.unlink()

    return result
