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
    
    # --- FIXES FOR ALIASING AND STARBURST STREAKS ---
    ct.protocol.viewsPerRotation = 1000 
    ct.protocol.viewCount = 1000        
    
    ct.physics.colSampleCount = 4       
    ct.physics.rowSampleCount = 2       
    ct.physics.srcXSampleCount = 4      
    ct.physics.srcYSampleCount = 2      
    ct.physics.viewSampleCount = 2
    
    ct.recon.kernelType = 'standard'    

    # Standard Voxelized Phantom config
    ct.phantom.filename = "my_phantom.json"
    ct.phantom.callback = "Phantom_Voxelized"
    ct.phantom.projectorCallback = "C_Projector_Voxelized"
    
    return ct

def run_scan_and_recon(ct, scan_type=""):
    """Runs the simulation, prints verification info, and returns the image."""
    
    # --- CODE-LEVEL VERIFICATION ---
    print(f"\n  [{scan_type} Verification] Preparing to run...")
    print(f"  -> Monochromatic Flag set to: {ct.physics.monochromatic}")
    if ct.physics.monochromatic == -1:
        spectrum_file = getattr(ct.protocol, 'spectrumFilename', 'Default Scanner Spectrum')
        print(f"  -> Polychromatic active. Using spectrum: {spectrum_file}")
    else:
        print(f"  -> Monochromatic active at {ct.physics.monochromatic} keV.")
    # -------------------------------

    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)
    
    imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
    img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
    return img[0, :, :]

def simulate_bh_one_phantom(phantom_id, size, pixel_size, X, Y):
    """Runs both Ideal and Beam Hardening scans for a specific phantom."""
    print(f"\n========================================")
    print(f"--- Running Phantom {phantom_id} ---")
    print(f"========================================")

    if phantom_id == 1:
        pd.generate_phantom_1(size, pixel_size, X, Y)
    elif phantom_id == 2:
        pd.generate_phantom_2(size, pixel_size, X, Y)

    # 1. Monochromatic (Ideal Physics)
    # Set to 120 keV to penetrate heavy metals and avoid photon starvation
    ct_mono = build_common_ct(f"bh_mono_p{phantom_id}")
    ct_mono.physics.monochromatic = 120 
    img_mono = run_scan_and_recon(ct_mono, scan_type="Ideal")

    # 2. Polychromatic (Beam Hardening Physics)
    # Set to -1 to force the use of the polychromatic spectrum
    ct_poly = build_common_ct(f"bh_poly_p{phantom_id}")
    ct_poly.physics.monochromatic = -1 
    img_poly = run_scan_and_recon(ct_poly, scan_type="Beam Hardening")

    return img_mono, img_poly

def plot_both_phantoms(mono1, poly1, mono2, poly2):
    """Plots a 2x2 grid comparing Ideal vs Beam Hardening for both phantoms."""
    plt.figure(figsize=(12, 10))

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

def plot_line_profile(img_mono, img_poly, size):
    """
    IMAGE-LEVEL VERIFICATION:
    Plots a 1D profile across the center of Phantom 1 (Water).
    """
    center_y = size // 2
    
    # Extract the middle row of pixels
    profile_mono = img_mono[center_y, :]
    profile_poly = img_poly[center_y, :]
    
    # --- DEBUG: Print the actual values to the terminal ---
    print(f"\n--- Debug: Center Pixel Values ---")
    print(f"Monochromatic Center Value: {profile_mono[size//2]}")
    print(f"Polychromatic Center Value: {profile_poly[size//2]}")
    # ------------------------------------------------------

    plt.figure(figsize=(10, 6))
    
    plt.plot(profile_mono, label='Monochromatic (Flat)', color='blue', linestyle='--')
    plt.plot(profile_poly, label='Polychromatic (Cupping)', color='red')
    
    plt.title("Verification: 1D Line Profile through Center of Water Phantom")
    plt.xlabel("Pixel Index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    
    # I REMOVED the plt.ylim() so the graph will auto-scale to fit your data perfectly!
    
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
    
    # 3. Display 2x2 Grid Results
    print("\n--- Plotting Final Grid ---")
    # plot_both_phantoms(mono1, poly1, mono2, poly2)
    
    # 4. Display Verification Line Profile
    print("\n--- Plotting Verification Line Profile ---")
    plot_line_profile(mono1, poly1, size)