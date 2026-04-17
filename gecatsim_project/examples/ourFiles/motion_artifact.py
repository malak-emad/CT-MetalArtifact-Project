# Copyright 2024, GE Precision HealthCare. All rights reserved.
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path, rawread
import os
import numpy as np

# --- MODULAR IMPORTS ---
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import phantom_definitions as pd
import ct_reconstruction as ctr  

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
    
    # Motion configuration
    shift_mm = 1.4  
    shift_px = shift_mm / pixel_size
    break_view = 700 
    
    n_views = ct.protocol.viewCount
    n_cells = ct.scanner.detectorColCount * ct.scanner.detectorRowCount

    # --- 3. Scan Position A (Normal) ---
    print("\n--- Scanning Position A ---")
    ct.resultsName = "pos_a"
    pd.generate_phantom_1(size, pixel_size, X, Y)
    ct.run_all()
    sino_a = rawread("pos_a.prep", [n_views, n_cells], 'float')

    # --- 4. Scan Position B (Shifted) ---
    print(f"\n--- Scanning Position B (Shifted {shift_mm}mm) ---")
    ct.resultsName = "pos_b"
    pd.generate_phantom_1(size, pixel_size, X, Y - shift_px)
    ct.run_all()
    sino_b = rawread("pos_b.prep", [n_views, n_cells], 'float')

    # --- 5. Splice Data (Create Artifact) ---
    print(f"\n--- Splicing Data at view {break_view} ---")
    motion_sino = np.zeros_like(sino_a)
    motion_sino[:break_view, :] = sino_a[:break_view, :]
    motion_sino[break_view:, :] = sino_b[break_view:, :]

    ct.resultsName = "motion_artifact"
    motion_sino.astype(np.float32).tofile(f"{ct.resultsName}.prep")

    # --- 6. Reconstruct & Plot ---
    print("\n--- Handing off to ct_reconstruction.py ---")
    
    ctr.perform_recon_and_plot(ct, plot_title="Motion Artifact")