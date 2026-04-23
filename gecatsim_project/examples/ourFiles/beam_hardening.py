# Copyright 2024, GE Precision HealthCare. All rights reserved.
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- MODULAR IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd
import ct_reconstruction as ctr

def build_common_ct(results_name):
    """Creates a standard scanner with anti-aliasing enabled."""
    ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    ct = ctr.setup_clean_baseline(ct)
    ct.resultsName = results_name


    # Standard Voxelized Phantom config
    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"

    # CRITICAL FIX: Add these lines to match aliasing_artifact.py
    ct.phantom.centerOffset = [0.0, 0.0, 0.0]
    ct.phantom.scale = 1.0
    
    return ct

def run_scan_and_recon(ct):
    """Runs the simulation and returns the reconstructed image."""
    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)
    
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
    return img[0, :, :]

def simulate_bh_one_phantom(phantom_id, size, pixel_size, X, Y):
    """Runs both Ideal and Beam Hardening scans for a specific phantom."""
    print(f"\n--- Running Phantom {phantom_id} ---")

    # 1. Generate the physical shape
    if phantom_id == 1:
        pd.generate_phantom_1(size, pixel_size, X, Y)
    elif phantom_id == 2:
        pd.generate_phantom_2(size, pixel_size, X, Y)

    # 2. Monochromatic (Ideal Physics)
    print(f"  -> Scanning Monochromatic (Ideal)...")
    ct_mono = build_common_ct(f"bh_mono_p{phantom_id}")
    ct_mono.physics.monochromatic = 111
    img_mono = run_scan_and_recon(ct_mono)

    # 3. Polychromatic (Beam Hardening Physics)
    print(f"  -> Scanning Polychromatic (Beam Hardening)...")
    ct_poly = build_common_ct(f"bh_poly_p{phantom_id}")
    ct_poly.physics.monochromatic = -1
    img_poly = run_scan_and_recon(ct_poly)

    return img_mono, img_poly

def plot_both_phantoms(mono1, poly1, mono2, poly2):
    """Plots a 2x2 grid comparing Ideal vs Beam Hardening for both phantoms."""
    plt.figure(figsize=(12, 10))

    # --- Phantom 1 Row ---
    plt.subplot(2, 2, 1)
    plt.imshow(mono1, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 1: Ideal\n(Water & Iron)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(poly1, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 1: Beam Hardening\n(Water & Iron)")
    plt.colorbar(label='HU')
    plt.axis('off')

    # --- Phantom 2 Row ---
    plt.subplot(2, 2, 3)
    plt.imshow(mono2, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 2: Ideal\n(Plexi & Silver)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(poly2, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 2: Beam Hardening\n(Plexi & Silver)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    my_path.add_search_path(".")
    
    # 1. Setup shared grid mathematics
    temp_ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    temp_ct = ctr.setup_clean_baseline(temp_ct)
    size = temp_ct.recon.imageSize if hasattr(temp_ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size
    
    # 2. Run Simulations
    mono1, poly1 = simulate_bh_one_phantom(1, size, pixel_size, X, Y)
    mono2, poly2 = simulate_bh_one_phantom(2, size, pixel_size, X, Y)
    
    # 3. Display Results
    print("\n--- Plotting Final Grid ---")
    plot_both_phantoms(mono1, poly1, mono2, poly2)