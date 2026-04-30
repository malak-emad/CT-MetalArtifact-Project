# simulation_worker.py
import os, sys, traceback, shutil
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path, rawread
import phantom_definitions as pd

import aliasing_artifact
import beam_hardening
import motion_artifact
import noise_artifact
import scattering
import NMARcopy as nmar  

CACHE_DIR = Path("saved_results")
NMAR_OUT_DIR = Path("nmar_outputs")
ARTIFACT_OUT_DIR = Path("artifact_outputs")

CACHE_DIR.mkdir(exist_ok=True)
NMAR_OUT_DIR.mkdir(exist_ok=True)
ARTIFACT_OUT_DIR.mkdir(exist_ok=True)

def cache_save(key, arr, directory=CACHE_DIR): 
    np.save(directory / f"{key}.npy", arr)

def cache_exists(key, directory=CACHE_DIR): 
    return (directory / f"{key}.npy").exists()

def cache_load(key, directory=CACHE_DIR): 
    if not cache_exists(key, directory):
        return None
    data = np.load(directory / f"{key}.npy", allow_pickle=True)
    if data.shape == ():
        return data.item()        
    return data

class SimWorker(QThread):
    done, log, error = pyqtSignal(dict), pyqtSignal(str), pyqtSignal(str)

    def __init__(self, mode, params):
        super().__init__()
        self.mode = mode
        self.p = params

    def run(self):
        try:
            my_path.add_search_path(".")
            fn = getattr(self, f"_run_{self.mode.lower().replace(' ', '_')}")
            res = fn()
            self._cleanup_working_dir() 
            self.done.emit(res)
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _cleanup_working_dir(self):
        """Moves all generated GeCatSim files to artifact_outputs to keep the root clean."""
        patterns = ["*.raw", "*.prep", "*.air", "*.offset", "*.scan", "my_phantom.json"]
        for pattern in patterns:
            for file in Path(".").glob(pattern):
                try:
                    dest = ARTIFACT_OUT_DIR / file.name
                    if dest.exists(): dest.unlink() 
                    shutil.move(str(file), str(dest))
                except Exception:
                    pass 

    def _setup(self):
        ct = aliasing_artifact.build_common_ct("ui_temp", params=self.p)
        sz = getattr(ct.recon, "imageSize", 512)
        px = self.p["fov"] / sz
        Y, X = np.ogrid[:sz, :sz]
        return ct, sz, px, X, Y

    def _pfn(self):
        return pd.generate_phantom_1 if self.p["phantom_id"] == 1 else pd.generate_phantom_2

    def _cached(self, key, compute_func, directory=CACHE_DIR):
        if cache_exists(key, directory):
            self.log.emit(f"Loaded {key} from cache")
            return cache_load(key, directory)
        self.log.emit(f"Computing {key}...")
        val = compute_func()
        cache_save(key, val, directory)
        return val

    # -- Artifact Runners --
    def _run_aliasing(self):
        f = self.p.get("view_factor", 4)
        base = f"alias_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_vf{f}"
        
        def compute_aliasing():
            ct, sz, px, X, Y = self._setup()
            self._pfn()(sz, px, X, Y)
            ct.resultsName = "aliasing_base"; ct.run_all()
            sino = rawread("aliasing_base.prep", [ct.protocol.viewCount, ct.scanner.detectorColCount * ct.scanner.detectorRowCount], 'float')
            
            s_d = aliasing_artifact.create_detector_undersampling(sino)
            s_d.astype(np.float32).tofile("aliasing_detector.prep")
            img_d = aliasing_artifact.reconstruct_from_prep(aliasing_artifact.build_common_ct("aliasing_detector", params=self.p), "aliasing_detector.prep", "aliasing_detector")
            
            s_v = aliasing_artifact.create_view_undersampling(sino, factor=f)
            s_v.astype(np.float32).tofile("aliasing_view.prep")
            img_v = aliasing_artifact.reconstruct_from_prep(aliasing_artifact.build_common_ct("aliasing_view", params=self.p), "aliasing_view.prep", "aliasing_view")
            return {"detector": img_d, "view": img_v}

        res = self._cached(base, compute_aliasing)
        return {"type": "aliasing", **res}

    def _run_beam_hardening(self):
        base_mono = f"bh_mono_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_k{self.p['keV']}"
        base_poly = f"bh_poly_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_k{self.p['keV']}"
        
        ct, sz, px, X, Y = self._setup()
        mono, poly = beam_hardening.simulate_bh_one_phantom(self.p["phantom_id"], sz, px, X, Y, params=self.p)
        
        i_mono = self._cached(base_mono, lambda: mono)
        i_poly = self._cached(base_poly, lambda: poly)
        return {"type": "beam_hardening", "mono": i_mono, "poly": i_poly}

    def _run_motion(self):
        s, bv = self.p.get("shift_mm", 1.4), self.p.get("break_view", 700)
        base = f"motion_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_s{s}_b{bv}"
        
        def compute_motion():
            ct, sz, px, X, Y = self._setup()
            return motion_artifact.run_motion_artifact(ct, sz, px, X, Y, phantom_fn=self._pfn(), shift_mm=s, break_view=bv)
            
        img = self._cached(base, compute_motion)
        return {"type": "motion", "motion": img}

    def _run_noise(self):
        base = f"noise_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_mA{self.p['mA']}"
        
        def compute_noise():
            ct, sz, px, X, Y = self._setup()
            return noise_artifact.simulate_one_phantom(self.p["phantom_id"], f"P{self.p['phantom_id']}", sz, px, X, Y, noisy_mA=self.p["mA"], params=self.p)
            
        cln, nsy, _ = self._cached(base, compute_noise)
        return {"type": "noise", "clean": cln, "noisy": nsy, "diff": nsy - cln}

    def _run_scatter(self):
        sc_val = self.p.get("scatter_scale", 1000.0)
        base = f"scatter_p{self.p['phantom_id']}_f{self.p['fov']}_v{self.p['views']}_sc{sc_val}"
        
        def compute_scatter():
            ct, sz, px, X, Y = self._setup()
            return scattering.simulate_scatter_one_phantom(self.p["phantom_id"], sz, px, X, Y, scatter_scale=sc_val, params=self.p)
            
        idl, sc_art = self._cached(base, compute_scatter)
        return {"type": "scatter", "ideal": idl, "scatter": sc_art}

    def _run_combined(self):
        p_id, fov, views = self.p['phantom_id'], self.p['fov'], self.p['views']
        kev, mA = self.p['keV'], self.p['mA']
        sc = self.p.get('scatter_scale', 1000.0)
        vf = self.p.get('view_factor', 4)
        sh = self.p.get('shift_mm', 1.4)
        bv = self.p.get('break_view', 700)
        
        base_key = f"comb_p{p_id}_f{fov}_v{views}_k{kev}_mA{mA}_sc{sc}_vf{vf}_sh{sh}_bv{bv}"

        if cache_exists(base_key, CACHE_DIR):
            self.log.emit("Loaded Combined Artifacts from cache")
            res = cache_load(base_key, CACHE_DIR)
            return {"type": "combined", **res, "show_steps": self.p.get("show_steps", False)}

        self.log.emit("Computing Combined Artifacts...")
        images = {
            "Beam Hardening": self._run_beam_hardening()["poly"],
            "Scatter": self._run_scatter()["scatter"],
            "Noise": self._run_noise()["noisy"],
            "Motion": self._run_motion()["motion"],
            "Aliasing": self._run_aliasing()["detector"]
        }
        combined = np.mean(np.stack(list(images.values()), axis=0), axis=0)
        
        res_data = {"images": images, "combined": combined}
        cache_save(base_key, res_data, CACHE_DIR)
        
        return {"type": "combined", **res_data, "show_steps": self.p.get("show_steps", False)}

    def _run_nmar(self):
        pid, fov = self.p["phantom_id"], self.p["fov"]
        h_th, ang = self.p["hu_threshold"], self.p["n_angles"]
        mA, views, keV = self.p["mA"], self.p["views"], self.p["keV"]
        sc, vf = self.p.get("scatter_scale", 1000.0), self.p.get("view_factor", 4)
        sh, bv = self.p.get("shift_mm", 1.4), self.p.get("break_view", 700)

        base_key = f"nmar_p{pid}_f{fov}_v{views}_mA{mA}_k{keV}_sc{sc}_vf{vf}_sh{sh}_bv{bv}_th{h_th}_ang{ang}"

        if cache_exists(f"{base_key}_corrected", NMAR_OUT_DIR):
            self.log.emit("Loaded complete NMAR from cache")
            return {
                "type": "nmar",
                "original": cache_load(f"{base_key}_original", NMAR_OUT_DIR),
                "corrected": cache_load(f"{base_key}_corrected", NMAR_OUT_DIR),
                "sino_norm": cache_load(f"{base_key}_sino_norm", NMAR_OUT_DIR),
                "sino_corr": cache_load(f"{base_key}_sino_corr", NMAR_OUT_DIR),
                "metal_trace": cache_load(f"{base_key}_metal_trace", NMAR_OUT_DIR),
                "show_steps": self.p.get("show_steps", False)
            }

        self.log.emit("Building combined artifact image...")
        combined_res = self._run_combined()
        img_hu = combined_res["combined"]

        metal_mask = nmar.segment_metal(img_hu, hu_threshold=h_th)

        self.log.emit("Running NMAR pipeline...")
        results = nmar.run_nmar(img_hu, metal_mask, n_angles=ang, verbose=False)

        cache_save(f"{base_key}_original", results["img_original"], NMAR_OUT_DIR)
        cache_save(f"{base_key}_corrected", results["img_final"], NMAR_OUT_DIR)
        cache_save(f"{base_key}_sino_norm", results["sino_norm"], NMAR_OUT_DIR)
        cache_save(f"{base_key}_sino_corr", results["sino_corr"], NMAR_OUT_DIR)
        cache_save(f"{base_key}_metal_trace", results["metal_trace"], NMAR_OUT_DIR)

        return {
            "type": "nmar", "original": results["img_original"], "corrected": results["img_final"],
            "sino_norm": results["sino_norm"], "sino_corr": results["sino_corr"],
            "metal_trace": results["metal_trace"], "show_steps": self.p.get("show_steps", False)
        }