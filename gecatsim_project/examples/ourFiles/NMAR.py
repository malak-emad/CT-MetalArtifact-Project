import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os, sys

# ── import segmentation helper ──────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metal_segmentation import segment_metal


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD COMBINED PHANTOM FILES
# ═══════════════════════════════════════════════════════════════════════════

def load_combined_phantom(filepath, size=512):
 
    if not os.path.exists(filepath):
        print(f"  ERROR: {filepath} not found!")
        print(f"  Please run run_both_phantoms() first to generate combined files.")
        return None
    
    # Read the raw file
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    
    # Reshape to [1, size, size] then take the first slice
    expected_size = size * size
    if len(data) != expected_size:
        print(f"  WARNING: Expected {expected_size} pixels, got {len(data)}")
        # Try to reshape intelligently
        possible_size = int(np.sqrt(len(data)))
        if possible_size * possible_size == len(data):
            img = data.reshape(possible_size, possible_size)
            if possible_size != size:
                from scipy.ndimage import zoom
                zoom_factor = size / possible_size
                img = zoom(img, zoom_factor, order=1)
        else:
            return None
    else:
        img = data.reshape(size, size)
    
    return img

MU_WATER = 0.02  # mm⁻¹ approximate at 120kV

def hu_to_mu(img_hu):
    """HU to linear attenuation — always positive for real tissue."""
    return (img_hu / 1000.0 + 1.0) * MU_WATER


def forward_project(image, n_angles=360, n_detectors=None):
    """
    2-D parallel-beam forward projection (Radon transform).
    """
    N = image.shape[0]
    if n_detectors is None:
        n_detectors = int(np.ceil(N * np.sqrt(2))) | 1

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = np.zeros((n_angles, n_detectors), dtype=np.float32)

    half_d = (n_detectors - 1) / 2.0
    t = np.linspace(-half_d, half_d, n_detectors)
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0

    for i, theta_deg in enumerate(angles):
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        n_steps = int(np.ceil(N * np.sqrt(2))) + 1
        s = np.linspace(-half_d, half_d, n_steps)

        t_col = t[:, np.newaxis]
        s_row = s[np.newaxis, :]

        xs = t_col * cos_t + s_row * sin_t + cx
        ys = t_col * sin_t - s_row * cos_t + cy

        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        x1, y1 = x0 + 1, y0 + 1

        fx = xs - x0
        fy = ys - y0

        def clamp(arr):
            return np.clip(arr, 0, N - 1)

        I00 = image[clamp(y0), clamp(x0)]
        I10 = image[clamp(y0), clamp(x1)]
        I01 = image[clamp(y1), clamp(x0)]
        I11 = image[clamp(y1), clamp(x1)]

        valid = (x0 >= 0) & (x1 < N) & (y0 >= 0) & (y1 < N)

        val = (I00 * (1 - fx) * (1 - fy)
               + I10 * fx * (1 - fy)
               + I01 * (1 - fx) * fy
               + I11 * fx * fy) * valid

        step_size = (2 * half_d) / (n_steps - 1)
        sino[i] = val.sum(axis=1) * step_size

    return sino, angles


def get_metal_trace(metal_sino, threshold=1e-3):
    """
    Binary mask of the metal trace in sinogram space.
    """
    return metal_sino > threshold

def build_prior_image(img, metal_mask, smooth_sigma=1.5,
                      air_threshold=-700,
                      bone_threshold=200):
    """
    Construct the NMAR prior image by multi-threshold segmentation.
    """
    img_smooth = gaussian_filter(img.astype(np.float64), sigma=smooth_sigma)
    prior = np.zeros_like(img, dtype=np.float32)
    
    air_mask = img_smooth < air_threshold
    prior[air_mask] = -1000.0
    
    bone_mask = (img_smooth >= bone_threshold) & (metal_mask == 0)
    prior[bone_mask] = img[bone_mask]
    
    prior[metal_mask == 1] = 0.0
    
    return prior


def nmar_normalize_only(sino_orig, sino_prior, metal_trace, t_eps=1e-2):
    """
    NMAR normalization - only normalize where the object actually is.
    Outside the object both sinograms should be ~0, ratio is meaningless.
    """
    # Object support: where prior sinogram has meaningful signal
    object_support = sino_prior > t_eps
    
    sino_norm = np.ones_like(sino_orig)  # default = 1 outside object
    
    # Inside object: normalize normally
    sino_norm[object_support] = (sino_orig[object_support] / 
                                  sino_prior[object_support])
    
    # Outside object: both are ~0, so normalized should be ~0 too
    sino_norm[~object_support] = 0.0
    
    return sino_norm.astype(np.float32)

def run_nmar(img, metal_mask,
             n_angles=360,
             smooth_sigma=1.5,
             air_threshold=-700,
             bone_threshold=200,
             metal_sino_threshold=1e-3,
             t_eps=1e-2,
             verbose=True):

    if verbose:
        print("  [NMAR] Step 2a — Forward project metal mask …")
    sino_metal, angles = forward_project(metal_mask, n_angles=n_angles)
    metal_trace = get_metal_trace(sino_metal, threshold=metal_sino_threshold)

    if verbose:
        metal_px = metal_trace.sum()
        print(f"          Metal trace pixels in sinogram: {metal_px}")

    if verbose:
        print("  [NMAR] Step 3  — Build prior image …")
    prior = build_prior_image(img, metal_mask,
                               smooth_sigma=smooth_sigma,
                               air_threshold=air_threshold,
                               bone_threshold=bone_threshold)

    if verbose:
        print("  [NMAR] Step 2b — Forward project prior image …")


    sino_orig, _  = forward_project(hu_to_mu(img),   n_angles=n_angles)
    sino_prior, _ = forward_project(hu_to_mu(prior),  n_angles=n_angles)

    if verbose:
        print("  [NMAR] Step 4  — Normalize ONLY …")
    sino_norm = nmar_normalize_only(sino_orig, sino_prior, metal_trace, t_eps=t_eps)

    if verbose:
        print("  [NMAR] Done.")

    return {
        'prior': prior,
        'sino_orig': sino_orig,
        'sino_metal': sino_metal,
        'metal_trace': metal_trace,
        'sino_prior': sino_prior,
        'sino_norm': sino_norm,
        'angles': angles,
    }


def show_prior_image(img, metal_mask, prior, title=""):
    """
    Show prior image construction: original CT | metal mask | prior image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"NMAR Step 3 — Prior Image Construction  |  {title}",
                 fontsize=12, fontweight='bold')

    im0 = axes[0].imshow(img, cmap='gray', vmin=-1000, vmax=500)
    axes[0].set_title("Original CT Image (HU)\n(Combined Artifacts)", fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='HU')

    axes[1].imshow(metal_mask, cmap='hot')
    axes[1].set_title("Metal Mask\n(threshold > 2500 HU)", fontsize=11)
    axes[1].axis('off')

    im2 = axes[2].imshow(prior, cmap='gray', vmin=-1000, vmax=500)
    axes[2].set_title("Prior Image\n(air=−1000 | soft=0 | bone=orig | metal=0)",
                      fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='HU')

    plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_").replace("(", "") \
                .replace(")", "").replace("—", "").strip("_")
    fname = f"nmar_prior_image_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")


def show_sinograms(results, title=""):
    sino_orig  = results['sino_orig']
    sino_prior = results['sino_prior']
    sino_norm  = results['sino_norm']
    metal_trace = results['metal_trace']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"NMAR Sinograms  |  {title}", fontsize=14, fontweight='bold')

    def _panel(ax, data, label, vmin=None, vmax=None):
        im = ax.imshow(data, cmap='gray', aspect='auto',
                       extent=[0, data.shape[1], 180, 0],
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\nRange: [{data.min():.3f}, {data.max():.3f}]",
                     fontsize=10)
        ax.set_xlabel("Detector channel")
        ax.set_ylabel("Angle (°)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _panel(axes[0], sino_orig,  "Original  p(θ,t)")
    _panel(axes[1], sino_prior, "Prior  p_prior(θ,t)")
    
    # CLIP display to [0, 3] to see the actual normalization quality
    # The metal trace spike (>>1) will saturate white — that's expected
    _panel(axes[2], sino_norm,  "Normalized  p/p_prior\n(display clipped to [0,3])",
           vmin=0, vmax=3)

    # Also print non-metal stats
    obj  = sino_prior > 1e-2
    nm   = obj & ~metal_trace
    vals = sino_norm[nm]
    print(f"\n  Non-metal normalized sinogram stats:")
    print(f"    mean  = {vals.mean():.4f}  (ideal: ~1.0)")
    print(f"    std   = {vals.std():.4f}   (ideal: <0.3)")
    print(f"    range = [{vals.min():.4f}, {vals.max():.4f}]")

    plt.tight_layout()
    fname = f"nmar_sinos_{title.replace(' ','_')[:40]}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — Run NMAR on combined phantom files
# ═══════════════════════════════════════════════════════════════════════════

def run_nmar_on_combined_phantoms(size=512, n_angles=360):
    """
    Load the combined phantom files (average of all artifacts) and run NMAR.
    """
    # Paths to combined phantom files (saved by run_both_phantoms)
    combined_files = {
        "Phantom 1 — Water Bowl + Iron Rod": 
            "artifact_outputs/combined_p1_512x512x1.raw",
        "Phantom 2 — Plexiglas + Metal Cylinders": 
            "artifact_outputs/combined_p2_512x512x1.raw",
    }
    
    for label, filepath in combined_files.items():
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"  Loading: {filepath}")
        
        # Load the combined artifact image
        img = load_combined_phantom(filepath, size=size)
        if img is None:
            print(f"  SKIPPING: Could not load {filepath}")
            continue
        
        print(f"  Image shape: {img.shape}")
        print(f"  HU range: [{img.min():.1f}, {img.max():.1f}]")
        
        # Step 1: Segment metal (pixels > 2500 HU are metal)
        metal_mask = segment_metal(img, hu_threshold=2500)
        metal_pixels = int(metal_mask.sum())
        print(f"  Metal pixels: {metal_pixels} ({100*metal_pixels/img.size:.2f}%)")
        
        if metal_pixels == 0:
            print(f"  WARNING: No metal detected! Adjust threshold.")
            # Try lower threshold for Phantom 2 (amalgam might be lower HU)
            if "Plexiglas" in label:
                metal_mask = segment_metal(img, hu_threshold=500)
                metal_pixels = int(metal_mask.sum())
                print(f"  Retry with threshold=500: {metal_pixels} metal pixels")
        
        results = run_nmar(img, metal_mask, n_angles=n_angles, t_eps=1e-2, verbose=True)
        # Visualize
        print("  Generating figures …")
        show_prior_image(img, metal_mask, results['prior'], title=label)
        show_sinograms(results, title=label)
        
        # Summary
        print(f"\n  Summary for {label}")
        print(f"    Metal trace coverage: {results['metal_trace'].mean()*100:.2f}% of sinogram")
        print(f"    Original sinogram range: [{results['sino_orig'].min():.1f}, {results['sino_orig'].max():.1f}]")
        print(f"    Prior sinogram range: [{results['sino_prior'].min():.1f}, {results['sino_prior'].max():.1f}]")
        print(f"    Normalized sinogram range: [{results['sino_norm'].min():.2f}, {results['sino_norm'].max():.2f}]")


if __name__ == "__main__":
    print("="*60)
    print("NMAR Normalization on Combined Artifact Files")
    print("="*60)
    print("\nMake sure you have run run_both_phantoms() first to generate")
    print("the combined artifact files in artifact_outputs/")
    print("\nRunning NMAR...")
    
    run_nmar_on_combined_phantoms(size=512, n_angles=360)