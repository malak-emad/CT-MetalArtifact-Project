# Copyright 2024, GE Precision HealthCare. All rights reserved.
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path, rawread
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- MODULAR IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd
import ct_reconstruction as ctr


def build_common_ct(results_name):
    """
    Create a CatSim object with the clean baseline settings
    and voxelized phantom configuration.
    """
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


def read_recon_image(ct):
    """
    Read reconstructed image from raw output file.
    """
    img_fname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(
        img_fname,
        [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize],
        'float'
    )
    return img[0, :, :]


def reconstruct_from_prep(ct, prep_filename, output_name):
    """
    Reconstruct from an already prepared sinogram file.
    """
    ct.resultsName = output_name

    expected_prep = f"{ct.resultsName}.prep"
    if prep_filename != expected_prep:
        data = np.fromfile(prep_filename, dtype=np.float32)
        data.tofile(expected_prep)

    ct.do_Recon = 1
    recon.recon(ct)

    return read_recon_image(ct)


def create_detector_undersampling(sino):
    """
    Simulate detector under-sampling by keeping every other detector
    and filling missing detector channels by nearest-neighbor copy.
    This avoids artificial dark bands caused by zeroing.
    """
    sino_ds = np.copy(sino)

    # Fill odd detector channels from previous even detector channel
    sino_ds[:, 1::2] = sino_ds[:, 0::2]

    return sino_ds


def create_view_undersampling(sino):
    """
    Simulate view under-sampling by keeping every 4th view
    and filling missing views by nearest-neighbor copy.
    """
    sino_vs = np.copy(sino)

    for i in range(sino.shape[0]):
        nearest_kept = (i // 4) * 4
        sino_vs[i, :] = sino[nearest_kept, :]

    return sino_vs


def plot_results(img_detector_alias, img_view_alias):
    """
    Plot aliasing artifacts from detector and view under-sampling.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_detector_alias, cmap='gray', vmin=-810, vmax=-190)
    plt.title("(a) Detector under-sampling")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_view_alias, cmap='gray', vmin=-950, vmax=-200)
    plt.title("(b) View under-sampling")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    my_path.add_search_path(".")

    # ---------------------------------------------------
    # 1) Build base CT object
    # ---------------------------------------------------
    ct = build_common_ct("aliasing_base")

    size = ct.recon.imageSize if hasattr(ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    fov_mm = 300.0
    pixel_size = fov_mm / size

    # ---------------------------------------------------
    # 2) Generate Phantom 1
    # ---------------------------------------------------
    pd.generate_phantom_1(size, pixel_size, X, Y)

    # ---------------------------------------------------
    # 3) Run clean scan to get original sinogram
    # ---------------------------------------------------
    print("\n--- Running baseline scan for Phantom 1 ---")
    ct.resultsName = "aliasing_base"
    ct.run_all()

    n_views = ct.protocol.viewCount
    n_cells = ct.scanner.detectorColCount * ct.scanner.detectorRowCount

    sino = rawread("aliasing_base.prep", [n_views, n_cells], 'float')

    # ---------------------------------------------------
    # 4) Create detector under-sampling artifact
    #    Approximate half-detector sampling
    # ---------------------------------------------------
    print("\n--- Creating detector under-sampling sinogram ---")
    sino_detector = create_detector_undersampling(sino)
    sino_detector.astype(np.float32).tofile("aliasing_detector.prep")

    # ---------------------------------------------------
    # 5) Create view under-sampling artifact
    #    Approximate quarter-view sampling
    # ---------------------------------------------------
    print("\n--- Creating view under-sampling sinogram ---")
    sino_view = create_view_undersampling(sino)
    sino_view.astype(np.float32).tofile("aliasing_view.prep")

    # ---------------------------------------------------
    # 6) Reconstruct both
    # ---------------------------------------------------
    print("\n--- Reconstructing detector under-sampling case ---")
    ct_detector = build_common_ct("aliasing_detector")
    img_detector_alias = reconstruct_from_prep(
        ct_detector,
        "aliasing_detector.prep",
        "aliasing_detector"
    )

    print("\n--- Reconstructing view under-sampling case ---")
    ct_view = build_common_ct("aliasing_view")
    img_view_alias = reconstruct_from_prep(
        ct_view,
        "aliasing_view.prep",
        "aliasing_view"
    )

    # ---------------------------------------------------
    # 7) Plot results
    # ---------------------------------------------------
    print("\n--- Plotting aliasing results ---")
    plot_results(img_detector_alias, img_view_alias)