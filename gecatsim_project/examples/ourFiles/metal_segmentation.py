import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import gecatsim as xc
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_combined(filename, size=512):
    """Load combined image saved by combined_visualization.py"""
    if not os.path.exists(filename):
        print(f"  ERROR: {filename} not found.")
        print("  Run combined_visualization.py first!")
        return None
    img = np.fromfile(filename, dtype=np.float32).reshape((size, size))
    return img

def segment_metal(img, hu_threshold=2500, dilation_radius=3):
    """
    NMAR Step 1 — Metal Segmentation (with dilation only).

    Step A: Threshold (same as before)
    Step B: Dilation — expand mask to include full metal edges
    """

    # Step A: threshold
    mask = (img > hu_threshold)

    # Step B: dilation 
    struct = ndimage.generate_binary_structure(2, 2)
    mask = ndimage.binary_dilation(
        mask,
        structure=struct,
        iterations=dilation_radius
    )

    return mask.astype(np.float32)

def show_segmentation(img, metal_mask, title=""):
    """
    3 panels:
      1. Combined CT image (all artifacts averaged)
      2. Binary metal mask
      3. Overlay — metal regions in red
    """
    img_norm = np.clip((img + 1000) / 1500, 0, 1)
    overlay  = np.stack([img_norm, img_norm, img_norm], axis=-1)
    overlay[metal_mask == 1] = [1, 0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"NMAR Step 1 — Metal Segmentation  |  {title}",
        fontsize=13, fontweight='bold'
    )

    im0 = axes[0].imshow(img, cmap='gray', vmin=-1000, vmax=500)
    axes[0].set_title(
        "Combined CT Image\n(average of all 5 artifact types)",
        fontsize=11
    )
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='HU')

    axes[1].imshow(metal_mask, cmap='hot')
    axes[1].set_title(
        "Metal Mask\n(white = metal  |  threshold > 2500 HU)",
        fontsize=11
    )
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(
        "Overlay\n(metal regions = red)",
        fontsize=11
    )
    axes[2].axis('off')

    plt.tight_layout()
    safe  = title.replace(" ","_").replace("/","_").replace("(","") \
                 .replace(")","").replace("—","").strip("_")
    fname = f"metal_segmentation_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")

def run_segmentation_both_phantoms(size=512):
    """
    Run metal segmentation on the combined images produced by
    combined_visualization.py — one result per phantom.
    """
    phantoms = {
    "Phantom 1 — Water Bowl + Iron Rod":
        "artifact_outputs/combined_p1_512x512x1.raw",
      "Phantom 2 — Plexiglas + Metal Cylinders":
     "artifact_outputs/combined_p2_512x512x1.raw",
    }

    for label, fname in phantoms.items():
        print(f"\n{'='*55}")
        print(f"Processing: {label}")

        img = load_combined(fname, size)
        if img is None:
            continue

        mask = segment_metal(img, hu_threshold=2500, dilation_radius=3)
        n    = int(mask.sum())

        print(f"  Metal pixels detected : {n}")
        print(f"  Min HU : {img.min():.0f}  |  Max HU : {img.max():.0f}")

        show_segmentation(img, mask, title=label)

if __name__ == "__main__":
    run_segmentation_both_phantoms()