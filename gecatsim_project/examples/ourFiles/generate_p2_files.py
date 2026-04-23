# Generates phantom 2 versions of motion and aliasing artifacts
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path, rawread
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd
import ct_reconstruction as ctr

def build_ct(results_name):
    ct = xc.CatSim(
        "../cfg/Phantom_Sample",
        "../cfg/Scanner_Sample_generic",
        "../cfg/Protocol_Sample_axial"
    )
    ct.resultsName = results_name
    ct = ctr.setup_clean_baseline(ct)
    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0
    return ct

if __name__ == "__main__":
    my_path.add_search_path(".")

    size = 512
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size

    # ── MOTION ARTIFACT — PHANTOM 2 ──────────────────────────────
    print("\n===== Generating Motion Artifact — Phantom 2 =====")

    ct = build_ct("pos_a_p2")
    n_views = ct.protocol.viewCount
    n_cells = ct.scanner.detectorColCount * ct.scanner.detectorRowCount

    shift_mm = 1.4
    shift_px = shift_mm / pixel_size
    break_view = 700

    # Position A
    print("Scanning Position A (Phantom 2)...")
    ct.resultsName = "pos_a_p2"
    pd.generate_phantom_2(size, pixel_size, X, Y)
    ct.run_all()
    sino_a = rawread("pos_a_p2.prep", [n_views, n_cells], 'float')

    # Position B (shifted)
    print("Scanning Position B (Phantom 2, shifted)...")
    ct.resultsName = "pos_b_p2"
    pd.generate_phantom_2(size, pixel_size, X, Y - shift_px)
    ct.run_all()
    sino_b = rawread("pos_b_p2.prep", [n_views, n_cells], 'float')

    # Splice
    motion_sino = np.zeros_like(sino_a)
    motion_sino[:break_view, :] = sino_a[:break_view, :]
    motion_sino[break_view:, :]  = sino_b[break_view:, :]
    motion_sino.astype(np.float32).tofile("motion_artifact_p2.prep")

    # Reconstruct
    print("Reconstructing motion artifact (Phantom 2)...")
    ct_recon = build_ct("motion_artifact_p2")
    ct_recon.do_Recon = 1
    recon.recon(ct_recon)
    print("Done → motion_artifact_p2_512x512x1.raw")

    # ── ALIASING ARTIFACT — PHANTOM 2 ────────────────────────────
    print("\n===== Generating Aliasing Artifact — Phantom 2 =====")

    ct2 = build_ct("aliasing_base_p2")
    pd.generate_phantom_2(size, pixel_size, X, Y)

    print("Running baseline scan (Phantom 2)...")
    ct2.resultsName = "aliasing_base_p2"
    ct2.run_all()

    sino = rawread("aliasing_base_p2.prep", [n_views, n_cells], 'float')

    # Detector under-sampling
    sino_det = np.copy(sino)
    sino_det[:, 1::2] = sino_det[:, 0::2]
    sino_det.astype(np.float32).tofile("aliasing_detector_p2.prep")

    # Reconstruct detector aliasing
    print("Reconstructing detector aliasing (Phantom 2)...")
    ct_det = build_ct("aliasing_detector_p2")
    ct_det.do_Recon = 1
    recon.recon(ct_det)
    print("Done → aliasing_detector_p2_512x512x1.raw")

    print("\n===== All Phantom 2 files generated successfully =====")