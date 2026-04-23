import numpy as np
import matplotlib.pyplot as plt
import gecatsim as xc
import os

def find_raw(filename, size=512, search_dirs=None):
    if search_dirs is None:
        search_dirs = [".", "..", "../.."]
    for d in search_dirs:
        full_path = os.path.join(d, filename)
        if os.path.exists(full_path):
            img = xc.rawread(full_path, [1, size, size], 'float')
            img = img[0, :, :]
            # Force resize to exactly (size x size) just in case
            if img.shape != (size, size):
                from PIL import Image as PILImage
                img = np.array(
                    PILImage.fromarray(img).resize((size, size))
                )
            return img
    print(f"  WARNING: {filename} not found — using blank.")
    return np.zeros((size, size))

def verify_image(name, img):
    print(f"  [{name}]  Min: {img.min():.0f} HU  |  "
          f"Max: {img.max():.0f} HU  |  "
          f"Metal pixels (>500 HU): {int((img > 500).sum())}")

def show_phantom_figure(phantom_label, file_map, size=512):
    """
    Creates ONE figure with 6 subplots.
    Forces all images to exactly size x size.
    """
    print(f"\n{'='*55}")
    print(f"Processing: {phantom_label}")

    images = {}
    from scipy.ndimage import zoom
    
    for name, fname in file_map.items():
        # Try to find and read the file
        img = None
        for d in [".", "..", "../.."]:
            full_path = os.path.join(d, fname)
            if os.path.exists(full_path):
                try:
                    # Read with any size by checking file size first
                    file_size = os.path.getsize(full_path)
                    bytes_per_pixel = 4
                    possible_size = int(np.sqrt(file_size / bytes_per_pixel))
                    
                    if possible_size > 0:
                        img_data = xc.rawread(full_path, [1, possible_size, possible_size], 'float')
                        img = img_data[0, :, :]
                        
                        # Force to correct size
                        if img.shape != (size, size):
                            zoom_factors = (size / img.shape[0], size / img.shape[1])
                            img = zoom(img, zoom_factors, order=1)
                        break
                except:
                    continue
        
        if img is None:
            print(f"  WARNING: Could not read {fname}, using zeros")
            img = np.zeros((size, size))
        
        verify_image(name, img)
        images[name] = img

    # Rest of your function remains the same...
    titles = list(images.keys())
    imgs   = list(images.values())

    # True combined = average of all 5
    combined = np.mean(np.stack(imgs, axis=0), axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f"Metal Streak Artifacts — {phantom_label}\n"
        "5 Artifact Types + Combined Average",
        fontsize=15, fontweight='bold', y=0.98
    )

    vmin, vmax = -1000, 500

    for col in range(3):
        ax = axes[0, col]
        im = ax.imshow(imgs[col], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(titles[col], fontsize=12, fontweight='bold', pad=8)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='HU')

    ax = axes[1, 0]
    im = ax.imshow(imgs[3], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(titles[3], fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='HU')

    ax = axes[1, 1]
    im = ax.imshow(imgs[4], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(titles[4], fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='HU')

    ax = axes[1, 2]
    im = ax.imshow(combined, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title("Combined\n(Average of All 5)",
                 fontsize=12, fontweight='bold', pad=8, color='darkblue')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label='HU')

    plt.subplots_adjust(
        top=0.91, bottom=0.04,
        left=0.04, right=0.96,
        hspace=0.22, wspace=0.18
    )

    safe = phantom_label.replace(" ","_").replace("—","").replace("+","").replace("/","").strip("_")
    fname = f"combined_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")

    return combined

def run_both_phantoms(size=512):

    p1_files = {
        "Beam Hardening":  "bh_poly_p1_512x512x1.raw",
        "Scatter":         "scatter_artifact_p1_512x512x1.raw",
        "Noise (Poisson)": "p1_noisy_512x512x1.raw",
        "Motion":          "motion_artifact_p1_512x512x1.raw",
        "Aliasing":        "aliasing_detector_p1_512x512x1.raw",
    }

    p2_files = {
        "Beam Hardening":  "bh_poly_p2_512x512x1.raw",
        "Scatter":         "scatter_artifact_p2_512x512x1.raw",
        "Noise (Poisson)": "p2_noisy_512x512x1.raw",
        "Motion":          "motion_artifact_p2_512x512x1.raw",
        "Aliasing":        "aliasing_detector_p2_512x512x1.raw",
    }

    combined_p1 = show_phantom_figure(
        "Phantom 1 — Water Bowl + Iron Rod",
        p1_files, size
    )
    combined_p2 = show_phantom_figure(
        "Phantom 2 — Plexiglas + Metal Cylinders",
        p2_files, size
    )

    # Save combined images as .raw for segmentation
    combined_p1.astype(np.float32).tofile("combined_p1_512x512x1.raw")
    combined_p2.astype(np.float32).tofile("combined_p2_512x512x1.raw")
    print("\nSaved combined images for segmentation:")
    print("  combined_p1_512x512x1.raw")
    print("  combined_p2_512x512x1.raw")

    return combined_p1, combined_p2

if __name__ == "__main__":
    run_both_phantoms()