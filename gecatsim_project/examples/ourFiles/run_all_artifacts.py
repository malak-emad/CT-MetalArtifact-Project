import numpy as np
import matplotlib.pyplot as plt
import gecatsim as xc
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gecatsim.pyfiles.CommonTools import my_path, rawread

import phantom_definitions as pd
import ct_reconstruction as ctr
import beam_hardening as bh
import scattering as sc
import noise_artifact as na
import motion_artifact
import aliasing_artifact as aa

OUTPUT_DIR = "artifact_outputs"

def move_outputs():
    """Move all .raw and .prep and .json files to artifact_outputs folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir("."):
        if f.endswith(".raw") or f.endswith(".prep") or f.endswith(".json") or f.endswith(".air") or f.endswith(".offset") or f.endswith(".scan"):
            dest = os.path.join(OUTPUT_DIR, f)
            shutil.move(f, dest)

def run_all_artifacts():
    my_path.add_search_path(".")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_ct = xc.CatSim("../cfg/Phantom_Sample", "../cfg/Scanner_Sample_generic", "../cfg/Protocol_Sample_axial")
    temp_ct = ctr.setup_clean_baseline(temp_ct)
    size = getattr(temp_ct.recon, "imageSize", 512)
    Y, X = np.ogrid[:size, :size]
    pixel_size = 300.0 / size
    for pid in [1, 2]:
        phantom_fn = pd.generate_phantom_1 if pid == 1 else pd.generate_phantom_2
        print(f"\n{'='*55}")
        print(f"Running all artifacts for Phantom {pid}...")
        print(f"  -> Beam Hardening...")
        bh.simulate_bh_one_phantom(pid, size, pixel_size, X, Y)
        print(f"  -> Scatter...")
        sc.simulate_scatter_one_phantom(pid, size, pixel_size, X, Y)
        print(f"  -> Noise...")
        na.simulate_one_phantom(pid, f"Phantom {pid}", size, pixel_size, X, Y, noisy_mA=100)
        print(f"  -> Motion...")
        ct_mot = aa.build_common_ct(f"motion_tmp_p{pid}")
        motion_artifact.run_motion_artifact(
            ct_mot, size, pixel_size, X, Y,
            phantom_fn=phantom_fn, shift_mm=1.4, break_view=700
        )
        if os.path.exists("motion_artifact_512x512x1.raw"):
            os.rename("motion_artifact_512x512x1.raw", f"motion_artifact_p{pid}_512x512x1.raw")
        print(f"  -> Aliasing...")
        phantom_fn(size, pixel_size, X, Y)
        ct_ali = aa.build_common_ct("aliasing_base")
        ct_ali.resultsName = "aliasing_base"
        ct_ali.run_all()
        n_views = ct_ali.protocol.viewCount
        n_cells = ct_ali.scanner.detectorColCount * ct_ali.scanner.detectorRowCount
        sino = rawread("aliasing_base.prep", [n_views, n_cells], 'float')
        sino_det = aa.create_detector_undersampling(sino)
        sino_det.astype(np.float32).tofile("aliasing_detector.prep")
        ct_d = aa.build_common_ct("aliasing_detector")
        aa.reconstruct_from_prep(ct_d, "aliasing_detector.prep", "aliasing_detector")
        if os.path.exists("aliasing_detector_512x512x1.raw"):
            os.rename("aliasing_detector_512x512x1.raw", f"aliasing_detector_p{pid}_512x512x1.raw")
        move_outputs()
    print("\n All artifact simulations complete. Files saved to artifact_outputs/")