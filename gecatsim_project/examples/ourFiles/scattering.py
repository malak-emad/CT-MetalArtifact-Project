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

def build_common_ct(results_name, params=None):
    """Creates a standard scanner with anti-aliasing enabled."""
    ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    ct = ctr.setup_clean_baseline(ct)
    ct.resultsName = results_name

    if params:
        if "fov"   in params: ct.recon.fov                  = params["fov"]
        if "mA"    in params: ct.protocol.mA                = params["mA"]
        if "keV"   in params: ct.physics.monochromatic       = params["keV"]
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
    """Runs the simulation and returns the reconstructed image."""
    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)
    
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
    return img[0, :, :]

def simulate_scatter_one_phantom(phantom_id, size, pixel_size, X, Y, scatter_scale=1000.0, params=None):
    """Runs both Ideal and Scatter scans for a specific phantom."""
    print(f"\n--- Running Phantom {phantom_id} ---")

    # 1. Generate the physical shape
    if phantom_id == 1:
        pd.generate_phantom_1(size, pixel_size, X, Y)
    elif phantom_id == 2:
        pd.generate_phantom_2(size, pixel_size, X, Y)

    # 2. IDEAL (No Scatter, Monochromatic)
    print(f"  -> Scanning Ideal (No Scatter)...")
    ct_ideal = build_common_ct(f"scatter_ideal_p{phantom_id}", params)
    ct_ideal.physics.monochromatic = 100
    
    # Explicitly disable scatter (from baseline)
    ct_ideal.physics.scatterCallback = ""
    ct_ideal.physics.scatterScaleFactor = 0
    img_ideal = run_scan_and_recon(ct_ideal)

    # 3. SCATTER ARTIFACT
    print(f"  -> Scanning with Scatter Enabled...")
    ct_scatter = build_common_ct(f"scatter_artifact_p{phantom_id}", params)
    
    # CRITICAL: Keep monochromatic ON. If we turn it off, we mix beam hardening with scatter!
    ct_scatter.physics.monochromatic = 70 
    
    # CRITICAL CHANGE: Enable the scatter convolution model
    ct_scatter.physics.scatterCallback = "Scatter_ConvolutionModel"
    ct_scatter.physics.scatterScaleFactor = scatter_scale
    
    img_scatter = run_scan_and_recon(ct_scatter)

    return img_ideal, img_scatter

def plot_both_phantoms(ideal1, scatter1, ideal2, scatter2):
    """Plots a 2x2 grid comparing Ideal vs Scatter for both phantoms."""
    plt.figure(figsize=(12, 10))

    # --- Phantom 1 Row ---
    plt.subplot(2, 2, 1)
    plt.imshow(ideal1, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 1: Ideal\n(No Scatter)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(scatter1, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 1: Scatter Artifact\n(Cupping & Streaks)")
    plt.colorbar(label='HU')
    plt.axis('off')

    # --- Phantom 2 Row ---
    plt.subplot(2, 2, 3)
    plt.imshow(ideal2, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 2: Ideal\n(No Scatter)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(scatter2, cmap='gray', vmin=-1000, vmax=500)
    plt.title("Phantom 2: Scatter Artifact\n(Metal Streaking)")
    plt.colorbar(label='HU')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    my_path.add_search_path(".")
    
    # Setup shared grid mathematics
    temp_ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    temp_ct = ctr.setup_clean_baseline(temp_ct)
    size = temp_ct.recon.imageSize if hasattr(temp_ct.recon, "imageSize") else 512
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size
    
    # Run Simulations
    ideal1, scatter1 = simulate_scatter_one_phantom(1, size, pixel_size, X, Y)
    ideal2, scatter2 = simulate_scatter_one_phantom(2, size, pixel_size, X, Y)
    
    # Display Results
    plot_both_phantoms(ideal1, scatter1, ideal2, scatter2)