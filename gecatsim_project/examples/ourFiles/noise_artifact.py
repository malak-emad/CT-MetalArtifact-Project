# Copyright 2024, GE Precision HealthCare. All rights reserved.
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path
import gecatsim.reconstruction.pyfiles.recon as recon
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
    Create a CatSim object with your clean baseline settings
    and the voxelized phantom configuration.
    """
    ct = xc.CatSim(
        "../cfg/Phantom_Sample",
        "../cfg/Scanner_Sample_generic",
        "../cfg/Protocol_Sample_axial"
    )

    ct.resultsName = results_name
    ct = ctr.setup_clean_baseline(ct)

    # Phantom configuration
    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0

    return ct


def read_recon_image(ct):
    """
    Read the reconstructed .raw image from disk.
    """
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(
        imgFname,
        [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize],
        'float'
    )
    return img[0, :, :]


def run_scan_and_recon(ct):
    """
    Run forward simulation + reconstruction and return reconstructed slice.
    """
    ct.run_all()

    # Ensure reconstruction exists
    ct.do_Recon = 1
    recon.recon(ct)

    return read_recon_image(ct)


def simulate_one_phantom(phantom_id, phantom_name, size, pixel_size, X, Y, noisy_mA=100):
    """
    Generate one phantom, run clean + noisy scans, and return all 3 images.
    """
    print(f"\n--- Running {phantom_name} clean baseline scan ---")

    # Generate phantom geometry
    if phantom_id == 1:
        pd.generate_phantom_1(size, pixel_size, X, Y)
    elif phantom_id == 2:
        pd.generate_phantom_2(size, pixel_size, X, Y)
    else:
        raise ValueError("phantom_id must be 1 or 2")

    # CLEAN
    ct_clean = build_common_ct(f"noise_clean_p{phantom_id}")
    ct_clean.physics.enableQuantumNoise = 0
    ct_clean.physics.enableElectronicNoise = 0
    clean_img = run_scan_and_recon(ct_clean)

    print(f"\n--- Running {phantom_name} noisy scan with Poisson / quantum noise ---")

    # NOISY
    ct_noisy = build_common_ct(f"noise_poisson_p{phantom_id}")
    ct_noisy.physics.enableQuantumNoise = 1
    ct_noisy.physics.enableElectronicNoise = 0
    ct_noisy.protocol.mA = noisy_mA
    noisy_img = run_scan_and_recon(ct_noisy)

    # DIFFERENCE
    diff_img = noisy_img - clean_img

    return clean_img, noisy_img, diff_img


def plot_results_both_phantoms(
    clean1, noisy1, diff1,
    clean2, noisy2, diff2,
    phantom1_name="Phantom 1",
    phantom2_name="Phantom 2"
):
    """
    Display clean reconstruction, noisy reconstruction, and difference map
    for both phantoms in a 2x3 layout.
    """
    plt.figure(figsize=(18, 10))

    # ----------------------------
    # Row 1: Phantom 1
    # ----------------------------
    plt.subplot(2, 3, 1)
    plt.imshow(clean1, cmap='gray', vmin=-1000, vmax=500)
    plt.title(f"Clean Reconstruction\n({phantom1_name})")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(noisy1, cmap='gray', vmin=-1000, vmax=500)
    plt.title(f"Poisson Noise Reconstruction\n({phantom1_name})")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(diff1, cmap='gray')
    plt.title(f"Difference Image\n(Noisy - Clean, {phantom1_name})")
    plt.axis('off')
    plt.colorbar()

    # plt.subplot(2, 3, 3)
    # v1 = np.percentile(np.abs(diff1), 99.8)
    # plt.imshow(diff1, cmap='gray', vmin=-v1, vmax=v1)
    # plt.title(f"Difference Image\n(Noisy - Clean, {phantom1_name})")
    # plt.axis('off')
    # plt.colorbar()

    # ----------------------------
    # Row 2: Phantom 2
    # ----------------------------
    plt.subplot(2, 3, 4)
    plt.imshow(clean2, cmap='gray', vmin=-1000, vmax=500)
    plt.title(f"Clean Reconstruction\n({phantom2_name})")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(noisy2, cmap='gray', vmin=-1000, vmax=500)
    plt.title(f"Poisson Noise Reconstruction\n({phantom2_name})")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    v2 = np.percentile(np.abs(diff2), 99.9)
    plt.imshow(diff2, cmap='gray', vmin=-v2, vmax=v2)
    plt.title(f"Difference Image\n(Noisy - Clean, {phantom2_name})")
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    my_path.add_search_path(".")

    # ---------------------------------------------------
    # 1) Create shared grid
    # ---------------------------------------------------
    temp_ct = xc.CatSim(
        "../cfg/Phantom_Sample",
        "../cfg/Scanner_Sample_generic",
        "../cfg/Protocol_Sample_axial"
    )
    temp_ct = ctr.setup_clean_baseline(temp_ct)

    size = temp_ct.recon.imageSize if hasattr(temp_ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    fov_mm = 300.0
    pixel_size = fov_mm / size

    # ---------------------------------------------------
    # 2) Run Phantom 1
    # ---------------------------------------------------
    clean1, noisy1, diff1 = simulate_one_phantom(
        phantom_id=1,
        phantom_name="Phantom 1",
        size=size,
        pixel_size=pixel_size,
        X=X,
        Y=Y,
        noisy_mA=100
    )

    # ---------------------------------------------------
    # 3) Run Phantom 2
    # ---------------------------------------------------
    clean2, noisy2, diff2 = simulate_one_phantom(
        phantom_id=2,
        phantom_name="Phantom 2",
        size=size,
        pixel_size=pixel_size,
        X=X,
        Y=Y,
        noisy_mA=100
    )

    # ---------------------------------------------------
    # 4) Plot both phantoms
    # ---------------------------------------------------
    print("\n--- Plotting results for both phantoms ---")
    plot_results_both_phantoms(
        clean1, noisy1, diff1,
        clean2, noisy2, diff2,
        phantom1_name="Phantom 1",
        phantom2_name="Phantom 2"
    )