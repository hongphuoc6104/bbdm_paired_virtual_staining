"""
Patch Extraction Pipeline for Virtual Staining.

Cuts large histopathology images into smaller patches with overlap.
NO filtering — keeps all patches including background, noise, and tissue
so the model learns to handle all types of content during inference.

Usage:
    python patch_extraction.py
    python patch_extraction.py --patch_size 512 --stride 256
    python patch_extraction.py --preview
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def extract_patches_from_image(img: Image.Image, patch_size: int,
                                stride: int) -> list:
    """Extract all patches from a single image. No filtering."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    patches = []

    if h < patch_size or w < patch_size:
        print(f"    [SKIP] Image {w}x{h} < patch_size {patch_size}")
        return patches

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_arr = arr[y:y + patch_size, x:x + patch_size]
            patches.append(Image.fromarray(patch_arr))

    return patches


def process_domain(image_dir: str, domain_name: str, patch_size: int,
                   stride: int) -> dict:
    """Process all images in a domain directory."""
    image_dir = Path(image_dir)
    valid_ext = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    all_patches = {}
    total = 0

    files = sorted([f for f in image_dir.iterdir()
                    if f.suffix.lower() in valid_ext])

    print(f"\n{'='*60}")
    print(f"Domain: {domain_name} ({len(files)} images)")
    print(f"{'='*60}")

    for img_path in files:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        patches = extract_patches_from_image(img, patch_size, stride)
        total += len(patches)
        all_patches[img_path.name] = patches
        print(f"  {img_path.name:25s}  {w}x{h}  → {len(patches):5d} patches")

    print(f"\n  Total: {total} patches")
    return all_patches


def split_train_test(patches_by_source: dict, test_ratio: float = 0.2) -> tuple:
    """Split by source image to prevent data leakage."""
    sources = list(patches_by_source.keys())
    random.shuffle(sources)

    n_test = max(1, int(len(sources) * test_ratio))
    test_src = set(sources[:n_test])
    train_src = set(sources[n_test:])

    train_patches = [p for s in train_src for p in patches_by_source[s]]
    test_patches = [p for s in test_src for p in patches_by_source[s]]

    print(f"  Split: {len(train_src)} train ({len(train_patches)} patches) "
          f"/ {len(test_src)} test ({len(test_patches)} patches)")
    print(f"  Test: {sorted(test_src)}")

    return train_patches, test_patches


def save_patches(patches: list, output_dir: Path, prefix: str = ""):
    """Save patches as PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, patch in enumerate(patches):
        patch.save(output_dir / f"{prefix}{i:05d}.png")


def generate_preview(trainA, trainB, output_dir, n=10):
    """Preview grid: row 1 = unstained, row 2 = stained."""
    if not trainA or not trainB:
        return
    ps = trainA[0].size[0]
    canvas = Image.new('RGB', (n * ps, 2 * ps), (255, 255, 255))

    for i, p in enumerate(random.sample(trainA, min(n, len(trainA)))):
        canvas.paste(p, (i * ps, 0))
    for i, p in enumerate(random.sample(trainB, min(n, len(trainB)))):
        canvas.paste(p, (i * ps, ps))

    path = output_dir / "preview_samples.png"
    canvas.save(path)
    print(f"\n  Preview: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract patches for virtual staining."
    )
    parser.add_argument("--unstained_dir", type=str,
                        default="NHUOM AO/CHƯA NHUỘM")
    parser.add_argument("--stained_dir", type=str,
                        default="NHUOM AO/ĐÃ NHUỘM")
    parser.add_argument("--output_dir", type=str, default="patches")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride (default: 128 = 50%% overlap)")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("PATCH EXTRACTION — NO FILTERING")
    print("=" * 60)
    print(f"  Patch:   {args.patch_size}x{args.patch_size}")
    print(f"  Stride:  {args.stride} (overlap {(1 - args.stride/args.patch_size)*100:.0f}%)")
    print(f"  Filter:  NONE (keep all patches)")
    print(f"  Output:  {output_dir}")

    # Extract
    unstained = process_domain(args.unstained_dir, "CHƯA NHUỘM",
                               args.patch_size, args.stride)
    stained = process_domain(args.stained_dir, "ĐÃ NHUỘM",
                             args.patch_size, args.stride)

    # Split
    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    print("\nUnstained:")
    trainA, testA = split_train_test(unstained, args.test_ratio)
    print("\nStained:")
    trainB, testB = split_train_test(stained, args.test_ratio)

    random.shuffle(trainA)
    random.shuffle(testA)
    random.shuffle(trainB)
    random.shuffle(testB)

    # Save
    print(f"\n{'='*60}")
    print("SAVING")
    print("=" * 60)

    save_patches(trainA, output_dir / "trainA", "unstained_")
    print(f"  trainA: {len(trainA)} patches")
    save_patches(trainB, output_dir / "trainB", "stained_")
    print(f"  trainB: {len(trainB)} patches")
    save_patches(testA, output_dir / "testA", "unstained_")
    print(f"  testA:  {len(testA)} patches")
    save_patches(testB, output_dir / "testB", "stained_")
    print(f"  testB:  {len(testB)} patches")

    if args.preview:
        generate_preview(trainA, trainB, output_dir)

    total = len(trainA) + len(testA) + len(trainB) + len(testB)
    print(f"\n{'='*60}")
    print(f"DONE — {total} patches total")
    print(f"  Unstained: {len(trainA)} train + {len(testA)} test")
    print(f"  Stained:   {len(trainB)} train + {len(testB)} test")
    print(f"  Output:    {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
