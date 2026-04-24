# Copyright 2024, GE Precision HealthCare. All rights reserved.
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path, rawread
import gecatsim.reconstruction.pyfiles.recon as recon
import os
import numpy as np

# --- MODULAR IMPORTS ---
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd
import ct_reconstruction as ctr


def run_motion_artifact(ct, size, pixel_size, X, Y, phantom_fn,
                        shift_mm=1.4, break_view=700):
    """
    Run a motion artifact simulation using a pre-configured ct object.

    Scans the phantom at two positions (rest and shifted), splices the
    sinograms at `break_view`, reconstructs, and returns the image array.

    Args:
        ct:          Configured CatSim object (from build_common_ct / _setup).
        size:        Phantom grid size in pixels.
        pixel_size:  mm per pixel.
        X, Y:        meshgrid arrays for phantom generation.
        phantom_fn:  Callable — pd.generate_phantom_1 or pd.generate_phantom_2.
        shift_mm:    Patient motion magnitude in mm.
        break_view:  View index at which the motion occurs.

    Returns:
        img: 2-D numpy array of the reconstructed slice.
    """
    shift_px = shift_mm / pixel_size
    n_views  = ct.protocol.viewCount
    n_cells  = ct.scanner.detectorColCount * ct.scanner.detectorRowCount

    # Scan A — rest position
    ct.resultsName = "motion_pos_a"
    phantom_fn(size, pixel_size, X, Y)
    ct.run_all()
    sino_a = rawread("motion_pos_a.prep", [n_views, n_cells], 'float')

    # Scan B — shifted position
    ct.resultsName = "motion_pos_b"
    phantom_fn(size, pixel_size, X, Y - shift_px)
    ct.run_all()
    sino_b = rawread("motion_pos_b.prep", [n_views, n_cells], 'float')

    # Splice & reconstruct
    bv      = min(break_view, n_views - 1)
    spliced = np.vstack([sino_a[:bv], sino_b[bv:]])
    ct.resultsName = "motion_artifact"
    spliced.astype(np.float32).tofile("motion_artifact.prep")
    ct.do_Recon = 1
    recon.recon(ct)

    fname = (f"motion_artifact_{ct.recon.imageSize}"
             f"x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw")
    img = xc.rawread(fname,
                     [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize],
                     'float')
    return img[0]


if __name__ == "__main__":
    my_path.add_search_path(".")
    ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")

    # --- 1. Apply Baseline ---
    ct = ctr.setup_clean_baseline(ct)

    # --- 2. Phantom Configuration ---
    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0

    # Grid calculations
    size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size

    # --- 3. Run via shared function ---
    img = run_motion_artifact(
        ct, size, pixel_size, X, Y,
        phantom_fn=pd.generate_phantom_1,
        shift_mm=1.4,
        break_view=700
    )

    # --- 4. Reconstruct & Plot ---
    print("\n--- Handing off to ct_reconstruction.py ---")
    ctr.perform_recon_and_plot(ct, plot_title="Motion Artifact")