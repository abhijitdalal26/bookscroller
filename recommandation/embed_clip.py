"""
CLIP visual embedding generation.
Model: openai/clip-vit-large-patch14 (ViT-L/14) → 768-dim unit vectors.
Covers are loaded as RGB, CLIP internally resizes to 224×224.
"""
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from config import CLIP_MODEL, CLIP_BATCH_SIZE


def _load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        # Grey placeholder keeps batch size consistent; embedding will be near-zero
        return Image.new('RGB', (224, 224), (128, 128, 128))


def generate_clip_embeddings(
    cover_paths: list[str],
    batch_size: int = CLIP_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    """
    Returns float32 array of shape (N, 768), L2-normalized.
    Automatically halves batch_size on CUDA OOM and retries.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading CLIP [{CLIP_MODEL}] on {device}...")
    model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()

    all_embeddings = []
    failed_paths   = []
    i = 0

    with torch.no_grad():
        pbar = tqdm(total=len(cover_paths), desc='CLIP embeddings', unit='img')
        while i < len(cover_paths):
            batch_paths = cover_paths[i : i + batch_size]
            images = [_load_image(p) for p in batch_paths]

            try:
                inputs = processor(
                    images=images, return_tensors='pt', padding=True
                ).to(device)
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
                all_embeddings.append(feats.cpu().float().numpy())
                pbar.update(len(batch_paths))
                i += batch_size

            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and batch_size > 8:
                    batch_size //= 2
                    torch.cuda.empty_cache()
                    print(f"\nOOM — reducing batch size to {batch_size}")
                else:
                    # Log and skip the batch
                    failed_paths.extend(batch_paths)
                    all_embeddings.append(
                        np.zeros((len(batch_paths), 768), dtype=np.float32)
                    )
                    pbar.update(len(batch_paths))
                    i += batch_size

        pbar.close()

    if failed_paths:
        print(f"Warning: {len(failed_paths)} images failed — zero vectors inserted")

    return np.vstack(all_embeddings)
