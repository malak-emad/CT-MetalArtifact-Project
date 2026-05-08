"""
NMAR_complete.py
================
Full NMAR pipeline following Meyer et al. (2010), Medical Physics 37(10).

PIPELINE (paper Fig. 1):
  Input CT (HU)
    ├─ [1] Segment metal  →  metal_mask
    ├─ [2] Forward-project metal_mask  →  sino_metal  →  metal_trace (binary)
    ├─ [3] Build prior image  →  prior
    ├─ [4] Forward-project original (μ)  →  sino_orig
    ├─ [5] Forward-project prior (μ)  →  sino_prior
    ├─ [6] Normalize: sino_norm = sino_orig / sino_prior  (outside metal trace)
    ├─ [7] Interpolate metal trace in sino_norm  →  sino_interp   
    ├─ [8] Denormalize: sino_corr = sino_prior * sino_interp       
    ├─ [9] FBP reconstruct sino_corr  →  corrected image           
    └─ [10] Reinsert metal pixels                                   

Visualization functions are all kept SEPARATE (show_* / plot_*).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless – saves PNGs instead of showing GUI
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import os, sys

# ── gecatsim imports (same pipeline as beam_hardening.py) ────────────────
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path

MU_WATER = 0.02   # mm⁻¹ approximate at 120 kV

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY: unit conversions
# ═══════════════════════════════════════════════════════════════════════════

def hu_to_mu(img_hu):
    """HU → linear attenuation coefficient (mm⁻¹). Always ≥ 0 for real tissue."""
    return np.maximum((img_hu / 1000.0 + 1.0) * MU_WATER, 0.0)


def mu_to_hu(img_mu):
    """Linear attenuation coefficient (mm⁻¹) → HU."""
    return (img_mu / MU_WATER - 1.0) * 1000.0


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 0 — LOAD PHANTOM
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


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — SEGMENT METAL
# ═══════════════════════════════════════════════════════════════════════════

def segment_metal(img, hu_threshold=2500):
    """
    Simple HU threshold to find metal pixels.
    Returns a boolean mask (True = metal).
    """
    print(f"  [segment_metal] Threshold = {hu_threshold} HU")
    mask = (img >= hu_threshold).astype(np.uint8)
    n = int(mask.sum())
    pct = 100.0 * n / img.size
    print(f"  [segment_metal] Metal pixels: {n}  ({pct:.3f}% of image)")
    return mask


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — FORWARD PROJECTION  (parallel-beam Radon transform)
# ═══════════════════════════════════════════════════════════════════════════

def forward_project(image, n_angles=180, n_detectors=None):
    """
    2-D parallel-beam forward projection (Radon transform).

    Parameters
    ----------
    image       : 2-D array (already in the desired units, e.g. μ or binary)
    n_angles    : number of projection angles  [0, 180)
    n_detectors : detector width; defaults to ceil(N·√2)|1

    Returns
    -------
    sino   : (n_angles × n_detectors) sinogram
    angles : 1-D array of angles in degrees
    """
    N = image.shape[0]
    if n_detectors is None:
        n_detectors = int(np.ceil(N * np.sqrt(2))) | 1   # always odd

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino   = np.zeros((n_angles, n_detectors), dtype=np.float32)

    half_d = (n_detectors - 1) / 2.0
    t      = np.linspace(-half_d, half_d, n_detectors)
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0

    n_steps  = int(np.ceil(N * np.sqrt(2))) + 1
    s        = np.linspace(-half_d, half_d, n_steps)
    step_size = (2 * half_d) / (n_steps - 1)

    for i, theta_deg in enumerate(angles):
        theta     = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        t_col = t[:, np.newaxis]    # (n_detectors, 1)
        s_row = s[np.newaxis, :]    # (1, n_steps)

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
               + I10 * fx       * (1 - fy)
               + I01 * (1 - fx) * fy
               + I11 * fx       * fy) * valid

        sino[i] = val.sum(axis=1) * step_size

    return sino, angles


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2b — METAL TRACE MASK
# ═══════════════════════════════════════════════════════════════════════════

def get_metal_trace(metal_sino, threshold=1e-3):
    """
    Binary mask of the metal shadow in sinogram space.
    Returns bool array same shape as metal_sino.
    """
    trace = metal_sino > threshold
    pct   = 100.0 * trace.sum() / trace.size
    print(f"  [get_metal_trace] Metal trace: {trace.sum()} pixels  ({pct:.2f}% of sinogram)")
    return trace


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — BUILD PRIOR IMAGE  (paper §II.C)
# ═══════════════════════════════════════════════════════════════════════════

def build_prior_image(img, metal_mask,
                      smooth_sigma=1.5,
                      air_threshold=-700,
                      bone_threshold=200):
    """
    Multi-threshold segmentation prior (Meyer et al. §II.C):
      • air   (HU < air_threshold)        → −1000 HU
      • soft tissue  (air_threshold ≤ HU < bone_threshold) → 0 HU
      • bone  (HU ≥ bone_threshold, non-metal)             → keep original HU
      • metal                                               → 0  (arbitrary; reinserted later)

    Gaussian smoothing is applied before thresholding to reduce streak
    artifacts biasing the segmentation.
    """
    print(f"  [build_prior] Smooth σ={smooth_sigma}  |  air<{air_threshold} HU  |  bone≥{bone_threshold} HU")

    img_s = gaussian_filter(img.astype(np.float64), sigma=smooth_sigma)
    prior = np.zeros_like(img, dtype=np.float32)   # soft tissue → 0 HU by default

    # Air
    air_mask  = img_s < air_threshold
    prior[air_mask] = -1000.0
    print(f"  [build_prior] Air pixels: {air_mask.sum()}")

    # Bone (non-metal)
    bone_mask = (img_s >= bone_threshold) & (metal_mask == 0)
    prior[bone_mask] = img[bone_mask].astype(np.float32)
    print(f"  [build_prior] Bone pixels: {bone_mask.sum()}")

    # Metal set to 0 (does not affect normalization — §II.C)
    prior[metal_mask == 1] = 0.0
    print(f"  [build_prior] Metal pixels zeroed: {(metal_mask==1).sum()}")

    hu_range = (prior[metal_mask==0].min(), prior[metal_mask==0].max())
    print(f"  [build_prior] Prior HU range (non-metal): [{hu_range[0]:.1f}, {hu_range[1]:.1f}]")
    return prior


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6 — NORMALIZE  (paper eq. p_norm = p / p_prior)
# ═══════════════════════════════════════════════════════════════════════════

def nmar_normalize(sino_orig, sino_prior, t_eps=1e-2):
    """
    Normalize the original sinogram by the prior sinogram.

    sino_norm[i,j] = sino_orig[i,j] / sino_prior[i,j]   where sino_prior > t_eps
    sino_norm[i,j] = 0                                    where sino_prior ≤ t_eps

    The normalized sinogram should be ≈ 1 everywhere inside the object
    (and ≈ 0 outside), making it nearly flat → better interpolation.
    """
    print(f"  [normalize] t_eps={t_eps}")
    obj_mask  = sino_prior > t_eps
    sino_norm = np.zeros_like(sino_orig)
    sino_norm[obj_mask] = sino_orig[obj_mask] / sino_prior[obj_mask]

    inside_vals = sino_norm[obj_mask]
    print(f"  [normalize] Inside-object normalized sino: "
          f"mean={inside_vals.mean():.4f}  std={inside_vals.std():.4f}  "
          f"range=[{inside_vals.min():.4f}, {inside_vals.max():.4f}]")
    return sino_norm.astype(np.float32), obj_mask


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7 — INTERPOLATE METAL TRACE  (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_metal_trace(sino_norm, metal_trace):
    """
    Linear interpolation of the metal trace in the normalized sinogram.
    For each projection angle (row) we interpolate across the detector
    channels that are inside the metal trace.

    This is MAR1-style linear interpolation applied to the NORMALIZED
    sinogram (§II.B of Meyer et al.).  Because sino_norm is nearly flat
    (≈ 1 everywhere in the object), the interpolation is very smooth and
    introduces minimal new artifacts.

    Returns
    -------
    sino_interp : same shape as sino_norm, with metal trace filled in
    """
    print(f"  [interpolate] Interpolating {metal_trace.sum()} metal-trace pixels …")
    sino_interp = sino_norm.copy()
    n_angles, n_det = sino_norm.shape
    fixed_rows = 0

    for i in range(n_angles):
        row   = sino_norm[i]
        mtrace_row = metal_trace[i]

        if not mtrace_row.any():
            continue   # no metal in this projection

        good_idx = np.where(~mtrace_row)[0]
        bad_idx  = np.where( mtrace_row)[0]

        if len(good_idx) < 2:
            # can't interpolate — fill with local mean of good pixels
            sino_interp[i, bad_idx] = row[good_idx].mean() if len(good_idx) > 0 else 0.0
            continue

        # linear interpolation across detector channels
        f = interp1d(good_idx, row[good_idx],
                     kind='linear',
                     bounds_error=False,
                     fill_value=(row[good_idx[0]], row[good_idx[-1]]))
        sino_interp[i, bad_idx] = f(bad_idx)
        fixed_rows += 1

    print(f"  [interpolate] Rows with metal interpolated: {fixed_rows} / {n_angles}")
    diff = np.abs(sino_interp - sino_norm)[metal_trace]
    print(f"  [interpolate] Mean absolute change in metal trace: {diff.mean():.5f}")
    return sino_interp.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8 — DENORMALIZE  (NEW)
#  p_corr = p_prior * M(p_norm)   — paper eq. (1)
# ═══════════════════════════════════════════════════════════════════════════

def nmar_denormalize(sino_interp, sino_prior):
    """
    Denormalize the interpolated, normalized sinogram.

    p_corr = p_prior  ×  sino_interp

    This step:
    (a) restores the correct scale and bone/air trace from the prior
    (b) ensures a seamless transition between original and interpolated data
        (the offset cancels because we divided then multiplied by sino_prior)
    """
    print(f"  [denormalize] Computing p_corr = p_prior × sino_interp …")
    sino_corr = sino_prior * sino_interp
    print(f"  [denormalize] p_corr range: [{sino_corr.min():.4f}, {sino_corr.max():.4f}]")
    return sino_corr.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 9 — FILTERED BACK-PROJECTION  (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def ramp_filter(sino):
    """
    Apply Ram-Lak (ramp) filter to each row of the sinogram in the
    Fourier domain.  This is the filter used in standard FBP.

    Returns filtered sinogram (same shape, float32).
    """
    n_angles, n_det = sino.shape
    # Pad to next power-of-2 for FFT efficiency
    pad = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))

    freqs = np.fft.rfftfreq(pad)
    ramp  = 2.0 * np.abs(freqs)           # |ω| in [0, 0.5]

    sino_f = np.zeros_like(sino)
    for i in range(n_angles):
        F            = np.fft.rfft(sino[i], n=pad)
        sino_f[i]   = np.fft.irfft(F * ramp, n=pad)[:n_det]

    return sino_f.astype(np.float32)


def fbp_reconstruct(sino, angles_deg, output_size=None):
    """
    Filtered back-projection for a parallel-beam sinogram.

    Parameters
    ----------
    sino        : (n_angles × n_detectors) sinogram  [μ units, mm⁻¹]
    angles_deg  : 1-D array of projection angles (degrees)
    output_size : reconstructed image side length (default = n_detectors)

    Returns
    -------
    img_mu  : (output_size × output_size) float32 image in μ (mm⁻¹)
    img_hu  : same image converted to HU
    """
    n_angles, n_det = sino.shape
    N = output_size if output_size else n_det

    print(f"  [fbp] Sinogram shape: {sino.shape}  →  Image: {N}×{N}")

    # 1. Ramp filter
    sino_f = ramp_filter(sino)
    print(f"  [fbp] Ramp-filtered sinogram range: [{sino_f.min():.5f}, {sino_f.max():.5f}]")

    # 2. Back-project
    img = np.zeros((N, N), dtype=np.float64)
    half_d = (n_det - 1) / 2.0
    cx, cy = (N - 1) / 2.0, (N - 1) / 2.0

    # Pixel coordinate grids
    ys, xs = np.mgrid[0:N, 0:N]
    xs_c   = xs - cx
    ys_c   = ys - cy

    for i, theta_deg in enumerate(angles_deg):
        theta     = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Detector position t for each pixel
        t_pixels = xs_c * cos_t + ys_c * sin_t + half_d   # shape (N,N)

        # Clamp to valid detector range
        t_clamped = np.clip(t_pixels, 0, n_det - 1 - 1e-6)

        # Linear interpolation along detector axis
        t0 = np.floor(t_clamped).astype(int)
        t1 = t0 + 1
        t1 = np.minimum(t1, n_det - 1)
        ft = t_clamped - t0

        row_vals = sino_f[i][t0] * (1 - ft) + sino_f[i][t1] * ft

        img += row_vals

    # Normalization factor
    img *= (np.pi / n_angles)

    img_mu = img.astype(np.float32)
    img_hu = mu_to_hu(img_mu).astype(np.float32)

    print(f"  [fbp] Reconstructed μ range: [{img_mu.min():.5f}, {img_mu.max():.5f}]")
    print(f"  [fbp] Reconstructed HU range: [{img_hu.min():.1f}, {img_hu.max():.1f}]")
    return img_mu, img_hu


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 10 — REINSERT METAL  (NEW)
# ═══════════════════════════════════════════════════════════════════════════

def reinsert_metal(img_corrected_hu, img_original_hu, metal_mask):
    """
    Paste original metal pixel values back into the corrected image.
    This is done for MAR1, MAR2, and NMAR alike (paper §II.B last paragraph).

    The metal region itself is not correctable by interpolation; we simply
    restore the original HU values so the implant is visible.
    """
    print(f"  [reinsert_metal] Reinserting {int(metal_mask.sum())} metal pixels …")
    result = img_corrected_hu.copy()
    result[metal_mask == 1] = img_original_hu[metal_mask == 1]
    print(f"  [reinsert_metal] Final HU range: [{result.min():.1f}, {result.max():.1f}]")
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER NMAR FUNCTION  (orchestrates all steps)
# ═══════════════════════════════════════════════════════════════════════════

def run_nmar(img, metal_mask,
             n_angles=180,
             smooth_sigma=1.5,
             air_threshold=-700,
             bone_threshold=200,
             metal_sino_threshold=1e-3,
             t_eps=1e-2,
             verbose=True):
    """
    Full NMAR pipeline.

    Parameters
    ----------
    img           : input CT image in HU (N×N float32)
    metal_mask    : binary metal mask (uint8, 1=metal)
    n_angles      : number of projection angles
    smooth_sigma  : Gaussian smoothing for prior segmentation
    air_threshold : HU threshold for air class
    bone_threshold: HU threshold for bone class
    metal_sino_threshold : threshold to binarize metal sinogram → metal trace
    t_eps         : minimum prior sinogram value before normalization
    verbose       : print step-by-step progress

    Returns
    -------
    dict with all intermediate and final results
    """
    N = img.shape[0]

    # ── Step 2a: Forward project metal mask ─────────────────────────────
    if verbose: print("\n  ── Step 2a: Forward-project metal mask ──")
    sino_metal, angles = forward_project(metal_mask.astype(np.float32),
                                         n_angles=n_angles,
                                         n_detectors=int(np.ceil(N * np.sqrt(2))) | 1)
    metal_trace = get_metal_trace(sino_metal, threshold=metal_sino_threshold)

    # ── Step 3: Build prior image ────────────────────────────────────────
    if verbose: print("\n  ── Step 3: Build prior image ──")
    prior = build_prior_image(img, metal_mask,
                               smooth_sigma=smooth_sigma,
                               air_threshold=air_threshold,
                               bone_threshold=bone_threshold)

    # ── Steps 4 & 5: Forward project original and prior (in μ) ──────────
    if verbose: print("\n  ── Step 4: Forward-project original image (μ) ──")
    sino_orig,  _ = forward_project(hu_to_mu(img),   n_angles=n_angles,
                                    n_detectors=sino_metal.shape[1])

    if verbose: print("\n  ── Step 5: Forward-project prior image (μ) ──")
    sino_prior, _ = forward_project(hu_to_mu(prior), n_angles=n_angles,
                                    n_detectors=sino_metal.shape[1])

    # ── Step 6: Normalize ────────────────────────────────────────────────
    if verbose: print("\n  ── Step 6: Normalize sino_orig / sino_prior ──")
    sino_norm, obj_mask = nmar_normalize(sino_orig, sino_prior, t_eps=t_eps)

    # ── Step 7: Interpolate metal trace ──────────────────────────────────
    if verbose: print("\n  ── Step 7: Interpolate metal trace in normalized sinogram ──")
    sino_interp = interpolate_metal_trace(sino_norm, metal_trace)

    # ── Step 8: Denormalize ──────────────────────────────────────────────
    if verbose: print("\n  ── Step 8: Denormalize → p_corr = p_prior × sino_interp ──")
    sino_corr = nmar_denormalize(sino_interp, sino_prior)

    # ── Step 9: FBP reconstruct ──────────────────────────────────────────
    if verbose: print("\n  ── Step 9: FBP reconstruct corrected sinogram ──")
    img_corr_mu, img_corr_hu = fbp_reconstruct(sino_corr, angles, output_size=N)

    # ── Step 10: Reinsert metal ──────────────────────────────────────────
    if verbose: print("\n  ── Step 10: Reinsert metal pixels ──")
    img_final = reinsert_metal(img_corr_hu, img, metal_mask)

    if verbose:
        print("\n  ── NMAR Complete ──")
        print(f"     Input  HU range : [{img.min():.1f}, {img.max():.1f}]")
        print(f"     Output HU range : [{img_final.min():.1f}, {img_final.max():.1f}]")

    return {
        # images
        'img_original'   : img,
        'metal_mask'     : metal_mask,
        'prior'          : prior,
        'img_corr_mu'    : img_corr_mu,
        'img_corr_hu'    : img_corr_hu,
        'img_final'      : img_final,
        # sinograms
        'sino_orig'      : sino_orig,
        'sino_metal'     : sino_metal,
        'sino_prior'     : sino_prior,
        'sino_norm'      : sino_norm,
        'sino_interp'    : sino_interp,
        'sino_corr'      : sino_corr,
        # masks
        'metal_trace'    : metal_trace,
        'obj_mask'       : obj_mask,
        'angles'         : angles,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION FUNCTIONS  (all separate, never mixed with logic)
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, fname):
    """Helper: save fig and close it."""
    dirpath = os.path.dirname(fname)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] Saved → {fname}")

def plot_step1_metal_segmentation(img, metal_mask, title="", out_dir="."):
    """
    STEP 1 — Show original CT and the segmented metal mask side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Step 1 — Metal Segmentation  |  {title}", fontweight='bold')

    im0 = axes[0].imshow(img, cmap='gray', vmin=-1000, vmax=500)
    axes[0].set_title(f"Original CT (HU)\nrange [{img.min():.0f}, {img.max():.0f}]")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='HU', fraction=0.046)

    axes[1].imshow(metal_mask, cmap='hot')
    axes[1].set_title(f"Metal Mask\n{int(metal_mask.sum())} metal pixels")
    axes[1].axis('off')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step1_metal_seg_{_safe(title)}.png"))


def plot_step2_metal_sinogram(sino_metal, metal_trace, angles, title="", out_dir="."):
    """
    STEP 2 — Metal sinogram and the binary metal trace mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Step 2 — Metal Sinogram & Trace  |  {title}", fontweight='bold')

    extent = [0, sino_metal.shape[1], angles[-1], angles[0]]

    im0 = axes[0].imshow(sino_metal, cmap='hot', aspect='auto', extent=extent)
    axes[0].set_title(f"Metal Sinogram\nrange [{sino_metal.min():.4f}, {sino_metal.max():.4f}]")
    axes[0].set_xlabel("Detector channel"); axes[0].set_ylabel("Angle (°)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(metal_trace, cmap='binary', aspect='auto', extent=extent)
    axes[1].set_title(f"Metal Trace (binary)\n{int(metal_trace.sum())} pixels  "
                      f"({100*metal_trace.mean():.2f}% of sinogram)")
    axes[1].set_xlabel("Detector channel"); axes[1].set_ylabel("Angle (°)")

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step2_metal_sino_{_safe(title)}.png"))


def plot_step3_prior_image(img, metal_mask, prior, title="", out_dir="."):
    """
    STEP 3 — Prior image construction: original | mask | prior.
    Also shows difference image (original − prior) to highlight changes.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Step 3 — Prior Image Construction  |  {title}", fontweight='bold')

    kw = dict(cmap='gray', vmin=-1000, vmax=500)

    im0 = axes[0].imshow(img,   **kw); axes[0].set_title("Original CT (HU)")
    axes[0].axis('off'); plt.colorbar(im0, ax=axes[0], label='HU', fraction=0.046)

    axes[1].imshow(metal_mask, cmap='hot'); axes[1].set_title("Metal Mask"); axes[1].axis('off')

    im2 = axes[2].imshow(prior, **kw); axes[2].set_title("Prior Image (HU)")
    axes[2].axis('off'); plt.colorbar(im2, ax=axes[2], label='HU', fraction=0.046)

    diff = img.astype(np.float32) - prior
    diff[metal_mask == 1] = 0   # ignore metal region in diff
    im3 = axes[3].imshow(diff, cmap='RdBu_r', vmin=-500, vmax=500)
    axes[3].set_title("Difference (orig − prior)\n[metal pixels zeroed]")
    axes[3].axis('off'); plt.colorbar(im3, ax=axes[3], label='ΔHU', fraction=0.046)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step3_prior_{_safe(title)}.png"))


def plot_steps456_sinograms(sino_orig, sino_prior, sino_norm, metal_trace,
                             title="", out_dir="."):
    """
    STEPS 4–6 — Three sinograms: original, prior, normalized.
    Also shows a single-angle profile for easy comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Steps 4-5-6 — Sinograms & Profiles  |  {title}", fontweight='bold')

    # Choose a representative angle (middle)
    mid = sino_orig.shape[0] // 2

    def _sino_panel(ax, data, label, vmin=None, vmax=None):
        im = ax.imshow(data, cmap='gray', aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n[{data.min():.4f}, {data.max():.4f}]")
        ax.set_xlabel("Detector channel"); ax.set_ylabel("Angle (°)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    _sino_panel(axes[0,0], sino_orig,  "Step 4: Original p(θ,t)  [μ]")
    _sino_panel(axes[0,1], sino_prior, "Step 5: Prior p_prior(θ,t)  [μ]")
    _sino_panel(axes[0,2], sino_norm,  "Step 6: Normalized p/p_prior",
                vmin=0, vmax=3)

    # Profile plots (angle = mid)
    chans = np.arange(sino_orig.shape[1])
    mt_row = metal_trace[mid]

    for ax, data, label in zip(axes[1],
                                [sino_orig, sino_prior, sino_norm],
                                ["p(θ,t)", "p_prior(θ,t)", "p_norm(θ,t)"]):
        ax.plot(chans, data[mid], 'b-', lw=1, label=label)
        ax.fill_between(chans, data[mid].min(), data[mid].max(),
                        where=mt_row, alpha=0.3, color='red', label='metal trace')
        ax.set_title(f"Profile at angle {mid}°")
        ax.set_xlabel("Detector channel"); ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"steps456_sinograms_{_safe(title)}.png"))


def plot_step7_interpolation(sino_norm, sino_interp, metal_trace,
                              title="", out_dir="."):
    """
    STEP 7 — Interpolation of the metal trace.
    Shows: normalized sino | interpolated sino | difference | example profile.
    """
    diff = sino_interp - sino_norm
    mid  = sino_norm.shape[0] // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Step 7 — Metal Trace Interpolation  |  {title}", fontweight='bold')

    def _p(ax, data, label, vmin=None, vmax=None, cmap='gray'):
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n[{data.min():.4f}, {data.max():.4f}]")
        ax.set_xlabel("Detector channel"); ax.set_ylabel("Angle (°)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    _p(axes[0,0], sino_norm,   "Normalized (before interp)", vmin=0, vmax=2)
    _p(axes[0,1], sino_interp, "Interpolated (after interp)", vmin=0, vmax=2)
    _p(axes[1,0], np.abs(diff), "Absolute change |interp − norm|",
       cmap='hot', vmin=0, vmax=diff.std()*3)

    # Profile at mid angle
    chans  = np.arange(sino_norm.shape[1])
    mt_row = metal_trace[mid]
    axes[1,1].plot(chans, sino_norm[mid],   'b-',  lw=1.5, label='Before interp')
    axes[1,1].plot(chans, sino_interp[mid], 'r--', lw=1.5, label='After interp')
    axes[1,1].fill_between(chans, 0, 2, where=mt_row,
                            alpha=0.25, color='orange', label='metal trace')
    axes[1,1].set_title(f"Profile at angle idx {mid}")
    axes[1,1].set_xlabel("Detector channel"); axes[1,1].set_ylabel("Normalized value")
    axes[1,1].set_ylim(0, 2)
    axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step7_interpolation_{_safe(title)}.png"))


def plot_step8_denormalization(sino_interp, sino_prior, sino_corr, sino_orig,
                                metal_trace, title="", out_dir="."):
    """
    STEP 8 — Denormalization.
    Shows: corrected sinogram vs original, and residual in metal trace region.
    """
    mid = sino_orig.shape[0] // 2
    chans = np.arange(sino_orig.shape[1])
    mt_row = metal_trace[mid]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Step 8 — Denormalization  |  {title}", fontweight='bold')

    def _p(ax, data, label, vmin=None, vmax=None, cmap='gray'):
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(f"{label}\n[{data.min():.4f}, {data.max():.4f}]")
        ax.set_xlabel("Detector"); ax.set_ylabel("Angle (°)")
        plt.colorbar(im, ax=ax, fraction=0.046)

    _p(axes[0,0], sino_orig, "Original sino p(θ,t)  [μ]")
    _p(axes[0,1], sino_corr, "Corrected sino p_corr  [μ]")

    diff_corr = sino_corr - sino_orig
    _p(axes[1,0], diff_corr, "Difference p_corr − p_orig\n(should be ~0 outside trace)",
       cmap='RdBu_r', vmin=-diff_corr.std()*3, vmax=diff_corr.std()*3)

    # Profile comparison
    axes[1,1].plot(chans, sino_orig[mid], 'b-',  lw=1.5, label='Original')
    axes[1,1].plot(chans, sino_corr[mid], 'r--', lw=1.5, label='Corrected')
    axes[1,1].fill_between(chans, sino_orig[mid].min(), sino_orig[mid].max(),
                            where=mt_row, alpha=0.25, color='orange', label='metal trace')
    axes[1,1].set_title(f"Profile at angle idx {mid}")
    axes[1,1].set_xlabel("Detector"); axes[1,1].set_ylabel("μ (mm⁻¹)")
    axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step8_denorm_{_safe(title)}.png"))


def plot_step9_reconstruction(img_corr_mu, img_corr_hu, title="", out_dir="."):
    """
    STEP 9 — FBP reconstruction result (before metal reinsertion).
    Shows both μ image and HU image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Step 9 — FBP Reconstruction  |  {title}", fontweight='bold')

    im0 = axes[0].imshow(img_corr_mu, cmap='gray',
                          vmin=0, vmax=img_corr_mu[img_corr_mu > 0].mean() * 2)
    axes[0].set_title(f"Corrected image (μ, mm⁻¹)\n[{img_corr_mu.min():.5f}, {img_corr_mu.max():.5f}]")
    axes[0].axis('off'); plt.colorbar(im0, ax=axes[0], label='μ (mm⁻¹)', fraction=0.046)

    im1 = axes[1].imshow(img_corr_hu, cmap='gray', vmin=-1000, vmax=500)
    axes[1].set_title(f"Corrected image (HU)\n[{img_corr_hu.min():.1f}, {img_corr_hu.max():.1f}]")
    axes[1].axis('off'); plt.colorbar(im1, ax=axes[1], label='HU', fraction=0.046)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step9_fbp_{_safe(title)}.png"))


def plot_step10_final_comparison(img_original, img_final, metal_mask,
                                  title="", out_dir="."):
    """
    STEP 10 — Final comparison: original vs NMAR corrected vs difference.
    """
    diff = img_final.astype(np.float32) - img_original.astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Step 10 — Final Result  |  {title}", fontweight='bold')

    kw = dict(cmap='gray', vmin=-1000, vmax=500)

    im0 = axes[0].imshow(img_original, **kw)
    axes[0].set_title(f"Original CT (HU)\n[{img_original.min():.0f}, {img_original.max():.0f}]")
    axes[0].axis('off'); plt.colorbar(im0, ax=axes[0], label='HU', fraction=0.046)

    im1 = axes[1].imshow(img_final, **kw)
    axes[1].set_title(f"NMAR Corrected (HU)\n[{img_final.min():.0f}, {img_final.max():.0f}]")
    axes[1].axis('off'); plt.colorbar(im1, ax=axes[1], label='HU', fraction=0.046)

    std_d = diff[metal_mask == 0].std()
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-3*std_d, vmax=3*std_d)
    axes[2].set_title(f"Difference (NMAR − Original)\nstd={std_d:.1f} HU (non-metal)")
    axes[2].axis('off'); plt.colorbar(im2, ax=axes[2], label='ΔHU', fraction=0.046)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"step10_final_{_safe(title)}.png"))


def plot_full_pipeline_summary(results, title="", out_dir="."):
    """
    Summary figure: all key images and sinograms in one overview panel.
    Rows: images | sinograms
    Cols: original → prior → norm sino → interp sino → corrected → final
    """
    img     = results['img_original']
    prior   = results['prior']
    s_orig  = results['sino_orig']
    s_norm  = results['sino_norm']
    s_interp= results['sino_interp']
    s_corr  = results['sino_corr']
    img_fin = results['img_final']
    mt      = results['metal_trace']

    fig, axes = plt.subplots(2, 5, figsize=(26, 9))
    fig.suptitle(f"Full NMAR Pipeline Overview  |  {title}",
                 fontsize=13, fontweight='bold')

    img_kw  = dict(cmap='gray', vmin=-1000, vmax=500)
    sino_kw = dict(cmap='gray', aspect='auto')

    def _img(ax, data, t, **kw):
        im = ax.imshow(data, **kw)
        ax.set_title(t, fontsize=9); ax.axis('off')
        return im

    def _sino(ax, data, t, vmin=None, vmax=None, **kwargs):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, **sino_kw, **kwargs)
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Det", fontsize=7); ax.set_ylabel("°", fontsize=7)
        ax.tick_params(labelsize=6)
        return im

    # Row 0: images
    _img(axes[0,0], img,            "Original CT",             **img_kw)
    _img(axes[0,1], results['metal_mask'], "Metal Mask",       cmap='hot')
    _img(axes[0,2], prior,          "Prior Image",             **img_kw)
    _img(axes[0,3], results['img_corr_hu'], "Reconstructed (HU)", **img_kw)
    _img(axes[0,4], img_fin,        "NMAR Final",              **img_kw)

    # Row 1: sinograms
    _sino(axes[1,0], s_orig,    "Orig sino p(θ,t) [μ]")
    _sino(axes[1,1], results['sino_metal'], "Metal sino", cmap='hot')
    _sino(axes[1,2], s_norm,    "Normalized",        vmin=0, vmax=2)
    _sino(axes[1,3], s_interp,  "Interpolated",      vmin=0, vmax=2)
    _sino(axes[1,4], s_corr,    "Corrected p_corr")

    # Override cmap for metal sino panel
    axes[1,1].images[0].set_cmap('hot')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"pipeline_summary_{_safe(title)}.png"))


def plot_before_after_reinsertion(img_before, img_after, title="", out_dir="."):
    """
    Compare reconstructed image just before and after metal reinsertion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Before vs. After Metal Reinsertion  |  {title}", fontweight='bold')

    kw = dict(cmap='gray', vmin=-1000, vmax=500)

    im0 = axes[0].imshow(img_before, **kw)
    axes[0].set_title(f"Corrected (Before Reinsertion)\n[{img_before.min():.0f}, {img_before.max():.0f}] HU")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='HU', fraction=0.046)

    im1 = axes[1].imshow(img_after, **kw)
    axes[1].set_title(f"Final (After Reinsertion)\n[{img_after.min():.0f}, {img_after.max():.0f}] HU")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='HU', fraction=0.046)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, f"before_after_reinsertion_{_safe(title)}.png"))


def _safe(s):
    """Sanitize a string for use in filenames."""
    return (s.replace(" ", "_").replace("/", "_")
             .replace("(", "").replace(")", "")
             .replace("—", "").replace("|", "")
             .strip("_"))[:60]


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC STATS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def print_sinogram_stats(results):
    """Print a clean table of sinogram quality metrics."""
    mt  = results['metal_trace']
    obj = results['obj_mask'] if 'obj_mask' in results else (results['sino_prior'] > 1e-2)
    nm  = obj & ~mt   # non-metal, inside object

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │  Sinogram Quality Summary                               │")
    print("  ├──────────────────┬──────────┬──────────┬───────────────┤")
    print("  │ Sinogram         │   min    │   max    │   mean ± std  │")
    print("  ├──────────────────┼──────────┼──────────┼───────────────┤")

    for key, label in [('sino_orig',   'Original p(θ,t)'),
                        ('sino_prior',  'Prior p_prior'),
                        ('sino_norm',   'Normalized (all)'),
                        ('sino_interp', 'Interpolated'),
                        ('sino_corr',   'Corrected p_corr')]:
        d = results[key]
        print(f"  │ {label:<16} │ {d.min():8.4f} │ {d.max():8.4f} │ "
              f"{d.mean():7.4f}±{d.std():.4f} │")

    print("  ├──────────────────┴──────────┴──────────┴───────────────┤")
    print("  │  Normalized sino (non-metal, inside object):           │")
    if nm.any():
        v = results['sino_norm'][nm]
        print(f"  │    mean={v.mean():.4f}  std={v.std():.4f}  "
              f"range=[{v.min():.4f},{v.max():.4f}]          │")
        print(f"  │    (ideal: mean≈1.0, std<0.3)                          │")
    print("  └─────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════════════════
#  GECATSIM HELPERS  (same approach as beam_hardening.py)
# ═══════════════════════════════════════════════════════════════════════════

import phantom_definitions as pd
import ct_reconstruction as ctr


def build_common_ct(results_name, params=None):
    """Creates a standard CatSim scanner — identical to beam_hardening.py."""
    ct = xc.CatSim("../cfg/Phantom_Sample",
                   "../cfg/Scanner_Sample_generic",
                   "../cfg/Protocol_Sample_axial")
    ct = ctr.setup_clean_baseline(ct)
    ct.resultsName = results_name

    if params:
        if "fov"   in params: ct.recon.fov            = params["fov"]
        if "mA"    in params: ct.protocol.mA           = params["mA"]
        if "keV"   in params: ct.physics.monochromatic  = params["keV"]
        if "views" in params:
            ct.protocol.viewsPerRotation = params["views"]
            ct.protocol.viewCount        = params["views"]
            ct.protocol.stopViewId       = params["views"] - 1

    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0
    return ct


def run_scan_and_recon(ct):
    """Run CatSim simulation + reconstruction → return 2-D HU image."""
    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)

    imgFname = (f"{ct.resultsName}_"
                f"{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw")
    img = xc.rawread(imgFname,
                     [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize],
                     'float')
    return img[0, :, :]


def simulate_phantom(phantom_id, size, pixel_size, X, Y, params=None):
    """
    Generate one of the project phantoms and run a polychromatic
    (beam-hardening) CatSim scan — same physics as beam_hardening.py.

    Returns
    -------
    img_hu : (size × size) float32 array in HU
    """
    print(f"\n  [sim] Generating Phantom {phantom_id} …")
    if phantom_id == 1:
        pd.generate_phantom_1(size, pixel_size, X, Y)
    elif phantom_id == 2:
        pd.generate_phantom_2(size, pixel_size, X, Y)

    print(f"  [sim] Running polychromatic (beam-hardening) scan …")
    ct = build_common_ct(f"nmar_poly_p{phantom_id}", params)
    ct.physics.monochromatic = -1          # polychromatic → beam hardening
    img_hu = run_scan_and_recon(ct)
    print(f"  [sim] Reconstructed HU range: [{img_hu.min():.1f}, {img_hu.max():.1f}]")
    return img_hu


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — Run NMAR on combined phantom files
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main processing loop."""
    phantoms = {
        "p1": {
            "name": "Phantom 1 — Water Bowl + Iron Rod",
            "file": r"artifact_outputs\combined_p1_512x512x1.raw",
            "hu_threshold": 2500,
            "bone_threshold": 200,
        },
        "p2": {
            "name": "Phantom 2 — Plexiglas + Metal Cylinders",
            "file": r"artifact_outputs\combined_p2_512x512x1.raw",
            "hu_threshold": 3000,
            "bone_threshold": 500,
        }
    }

    for key, params in phantoms.items():
        print(f"\n{'='*60}")
        print(f"Processing: {params['name']}")
        print(f"  Loading: {params['file']}")

        # Load the combined artifact image
        img = load_combined_phantom(params['file'], size=512)
        if img is None:
            print(f"  SKIPPING: Could not load {params['file']}")
            continue

        print(f"  Image shape: {img.shape}")
        print(f"  HU range: [{img.min():.1f}, {img.max():.1f}]")

        # Step 1: Segment metal (pixels > 2500 HU are metal)
        metal_mask = segment_metal(img, hu_threshold=params['hu_threshold'])
        metal_pixels = int(metal_mask.sum())
        print(f"  Metal pixels: {metal_pixels} ({100*metal_pixels/img.size:.2f}%)")

        if metal_pixels == 0:
            print(f"  WARNING: No metal detected! Adjust threshold.")
            # Try lower threshold for Phantom 2 (amalgam might be lower HU)
            if "Plexiglas" in params['name']:
                metal_mask = segment_metal(img, hu_threshold=500)
                metal_pixels = int(metal_mask.sum())
                print(f"  Retry with threshold=500: {metal_pixels} metal pixels")

        results = run_nmar(img, metal_mask, n_angles=360, t_eps=1e-2, verbose=True)
        
        # Visualize
        print("  Generating figures …")
        safe_title = _safe(params['name'])
        plot_step1_metal_segmentation(img, metal_mask, title=params['name'], out_dir="nmar_outputs")
        plot_step2_metal_sinogram(results['sino_metal'], results['metal_trace'], results['angles'], title=params['name'], out_dir="nmar_outputs")
        plot_step3_prior_image(img, metal_mask, results['prior'], title=params['name'], out_dir="nmar_outputs")
        plot_steps456_sinograms(results['sino_orig'], results['sino_prior'], results['sino_norm'], results['metal_trace'], title=params['name'], out_dir="nmar_outputs")
        plot_step7_interpolation(results['sino_norm'], results['sino_interp'], results['metal_trace'], title=params['name'], out_dir="nmar_outputs")
        plot_step8_denormalization(results['sino_interp'], results['sino_prior'], results['sino_corr'], results['sino_orig'], results['metal_trace'], title=params['name'], out_dir="nmar_outputs")
        plot_step9_reconstruction(results['img_corr_mu'], results['img_corr_hu'], title=params['name'], out_dir="nmar_outputs")
        plot_step10_final_comparison(results['img_original'], results['img_final'], results['metal_mask'], title=params['name'], out_dir="nmar_outputs")
        plot_full_pipeline_summary(results, title=params['name'], out_dir="nmar_outputs")
        plot_before_after_reinsertion(results['img_corr_hu'], results['img_final'], title=params['name'], out_dir="nmar_outputs")

        # Summary
        print(f"\n  Summary for {params['name']}")
        print_sinogram_stats(results)


if __name__ == "__main__":
    print("="*60)
    print("NMAR Normalization on Combined Artifact Files")
    print("="*60)
    print("\nMake sure you have run run_both_phantoms() first to generate")
    print("the combined artifact files in artifact_outputs/")
    print("\nRunning NMAR...")
    
    main()