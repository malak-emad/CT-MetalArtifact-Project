# Copyright 2024, GE Precision HealthCare. All rights reserved.
import sys, os, traceback, json, hashlib
import numpy as np
from pathlib import Path

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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QFrame, QGridLayout,
    QProgressBar, QTextEdit, QSplitter, QSizePolicy, QPushButton,
    QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

CACHE_DIR = Path("saved_results")
NMAR_OUT_DIR = Path("nmar_outputs")
CACHE_DIR.mkdir(exist_ok=True)
NMAR_OUT_DIR.mkdir(exist_ok=True)

def cache_save(key, arr, directory=CACHE_DIR): 
    np.save(directory / f"{key}.npy", arr)

def cache_exists(key, directory=CACHE_DIR): 
    return (directory / f"{key}.npy").exists()

def cache_load(key, directory=CACHE_DIR): 
    return np.load(directory / f"{key}.npy") if cache_exists(key, directory) else None

# ── Style configuration ───────────────────────────────────────────────────
BG, PANEL, CARD = "#0b0f19", "#111827", "#1f2937"
BORDER, MUTED, ACCENT, GREEN = "#374151", "#9ca3af", "#3b82f6", "#10b981"
TEXT = "#f3f4f6"

QSS = f"""
* {{ font-family: 'Segoe UI', system-ui, sans-serif; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
#logo {{ color: {ACCENT}; font-size: 24px; font-weight: 800; letter-spacing: 2px; }}
#tagline {{ color: {MUTED}; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }}
#section {{ color: {ACCENT}; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 10px; }}
#param {{ color: {MUTED}; font-size: 12px; }}
#div {{ background: {BORDER}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; background: {PANEL}; }}
QTabBar::tab {{ background: {CARD}; color: {MUTED}; padding: 8px 18px; border-radius: 4px; margin-right: 2px; font-size: 12px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: #ffffff; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 10px; font-size: 13px; min-height: 24px; }}
QCheckBox {{ color: {TEXT}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {BORDER}; border-radius: 4px; background: {CARD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2563eb, stop:1 #1d4ed8); color: #ffffff; border: none; border-radius: 6px; padding: 12px; font-size: 13px; font-weight: bold; letter-spacing: 1px; }}
QPushButton:hover {{ background: #3b82f6; }}
QPushButton:disabled {{ background: {BORDER}; color: {MUTED}; }}
QTextEdit {{ background: #000000; border: 1px solid {BORDER}; border-radius: 6px; color: {GREEN}; font-family: 'Consolas', monospace; font-size: 11px; padding: 8px; }}
QProgressBar {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""

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
            self.done.emit(fn())
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

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
        base_mono = f"bh_mono_p{self.p['phantom_id']}_f{self.p['fov']}_k{self.p['keV']}"
        base_poly = f"bh_poly_p{self.p['phantom_id']}_f{self.p['fov']}_k{self.p['keV']}"
        
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
        base = f"noise_p{self.p['phantom_id']}_f{self.p['fov']}_mA{self.p['mA']}"
        
        def compute_noise():
            ct, sz, px, X, Y = self._setup()
            return noise_artifact.simulate_one_phantom(self.p["phantom_id"], f"P{self.p['phantom_id']}", sz, px, X, Y, noisy_mA=self.p["mA"], params=self.p)
            
        cln, nsy, _ = self._cached(base, compute_noise)
        return {"type": "noise", "clean": cln, "noisy": nsy, "diff": nsy - cln}

    def _run_scatter(self):
        sc_val = self.p.get("scatter_scale", 1000.0)
        base = f"scatter_p{self.p['phantom_id']}_f{self.p['fov']}_sc{sc_val}"
        
        def compute_scatter():
            ct, sz, px, X, Y = self._setup()
            return scattering.simulate_scatter_one_phantom(self.p["phantom_id"], sz, px, X, Y, scatter_scale=sc_val, params=self.p)
            
        idl, sc_art = self._cached(base, compute_scatter)
        return {"type": "scatter", "ideal": idl, "scatter": sc_art}

    def _run_combined(self):
        images = {
            "Beam Hardening": self._run_beam_hardening()["poly"],
            "Scatter": self._run_scatter()["scatter"],
            "Noise": self._run_noise()["noisy"],
            "Motion": self._run_motion()["motion"],
            "Aliasing": self._run_aliasing()["detector"]
        }
        combined = np.mean(np.stack(list(images.values()), axis=0), axis=0)
        return {"type": "combined", "images": images, "combined": combined, "show_all": self.p.get("show_all", False), "show_steps": self.p.get("show_steps", False)}

    def _run_nmar(self):
        pid, fov = self.p["phantom_id"], self.p["fov"]
        h_th, ang = self.p["hu_threshold"], self.p["n_angles"]
        
        base_key = f"nmar_p{pid}_fov{fov}_th{h_th}_ang{ang}"
        
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

        ct, sz, px, X, Y = self._setup()
        self.log.emit(f"Generating base phantom for NMAR...")
        img_hu = nmar.simulate_phantom(pid, sz, px, X, Y, params=self.p)
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

# ── Main Application UI ───────────────────────────────────────────────────
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Artifact Viewer")
        self.resize(1500, 900)
        self.setStyleSheet(QSS)
        self.worker = None

        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        # ── Global Sidebar ──
        sidebar = QWidget(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(320)
        self.s_lay = QVBoxLayout(sidebar); self.s_lay.setContentsMargins(20, 25, 20, 25); self.s_lay.setSpacing(10)

        logo = QLabel("CT·SIM"); logo.setObjectName("logo"); logo.setAlignment(Qt.AlignCenter); self.s_lay.addWidget(logo)
        tag = QLabel("Artifact Simulation"); tag.setObjectName("tagline"); tag.setAlignment(Qt.AlignCenter); self.s_lay.addWidget(tag)
        self.s_lay.addWidget(self._div())

        # Initialize all possible input widgets globally
        self.w_params = {
            "Phantom": QComboBox(),
            "Artifact": QComboBox(),
            "FOV (mm)": self._dspin(100, 700, 300, 10),
            "mA": self._spin(10, 1200, 800),
            "Views": self._spin(100, 3000, 1000),
            "keV": self._spin(40, 140, 70),
            "View Factor": self._spin(2, 16, 4),
            "Shift (mm)": self._dspin(0.1, 20, 1.4, 0.1),
            "Break View": self._spin(1, 999, 700),
            "Scatter Scale": self._dspin(10, 5000, 1000, 100),
            "HU Threshold": self._spin(500, 5000, 2500),
            "N Angles": self._spin(90, 720, 360),
            "Show Steps": QCheckBox("Show Detailed Steps"),
            "Show All Artifacts": QCheckBox("Show All Sub-Artifacts")
        }
        
        self.w_params["Phantom"].addItems(["Phantom 1 — Water & Iron", "Phantom 2 — Plexi & Silver"])
        self.w_params["Artifact"].addItems(["Aliasing", "Beam Hardening", "Motion", "Noise", "Scatter", "Combined"])
        self.w_params["Artifact"].currentTextChanged.connect(self._refresh_visibility)

        # Build dynamic grid for settings
        self.param_rows = {}
        self.grid = QGridLayout(); self.grid.setSpacing(8)
        
        row_idx = 0
        for name, wid in self.w_params.items():
            if isinstance(wid, QCheckBox):
                self.grid.addWidget(wid, row_idx, 0, 1, 2)
                self.param_rows[name] = {'lbl': wid, 'wid': wid} # Label is part of checkbox
            else:
                lbl = QLabel(name); lbl.setObjectName("param")
                self.grid.addWidget(lbl, row_idx, 0)
                self.grid.addWidget(wid, row_idx, 1)
                self.param_rows[name] = {'lbl': lbl, 'wid': wid}
            row_idx += 1

        self.s_lay.addLayout(self.grid)
        self.s_lay.addStretch()
        layout.addWidget(sidebar)

        # ── Right Viewer Area ──
        right_w = QWidget(); r_lay = QVBoxLayout(right_w); r_lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._refresh_visibility)
        r_lay.addWidget(self.tabs)

        self.figs = {}
        for tab_name in ["Artifacts", "NMAR"]:
            w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(15, 15, 15, 15)
            split = QSplitter(Qt.Vertical)
            
            fig = Figure(facecolor=BG)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.figs[tab_name] = {"fig": fig, "canvas": canvas}
            split.addWidget(canvas)

            log_w = QWidget(); log_w.setStyleSheet(f"background: {PANEL};")
            log_l = QVBoxLayout(log_w); log_l.setContentsMargins(15, 10, 15, 15)
            log_l.addWidget(QLabel(f"{tab_name.upper()} LOG", objectName="tagline"))
            
            log_box = QTextEdit(); log_box.setReadOnly(True); log_box.setMaximumHeight(120)
            log_l.addWidget(log_box)
            self.figs[tab_name]["log"] = log_box
            split.addWidget(log_w)
            
            split.setSizes([600, 120])
            l.addWidget(split, stretch=1)
            
            btn = QPushButton(f"▶ RUN {tab_name.upper()}")
            btn.clicked.connect(lambda _, m=tab_name: self._run(m))
            prog = QProgressBar(); prog.setRange(0, 0); prog.setVisible(False)
            l.addWidget(btn); l.addWidget(prog)
            
            self.figs[tab_name]["btn"] = btn
            self.figs[tab_name]["prog"] = prog
            self.tabs.addTab(w, tab_name)

        layout.addWidget(right_w, stretch=1)
        self._refresh_visibility() # Initialize view

    # ── Visibility Engine ─────────────────────────────────────────────────────
    def _refresh_visibility(self):
        """Dynamically shows/hides sidebar parameters based on active tab & mode."""
        tab = self.tabs.tabText(self.tabs.currentIndex())
        art = self.w_params["Artifact"].currentText()
        
        visible = {"Phantom"} 

        if tab == "Artifacts":
            visible.add("Artifact")
            if art == "Aliasing": visible.update(["FOV (mm)", "Views", "View Factor"])
            elif art == "Beam Hardening": visible.update(["FOV (mm)", "keV"])
            elif art == "Motion": visible.update(["FOV (mm)", "Views", "Shift (mm)", "Break View"])
            elif art == "Noise": visible.update(["FOV (mm)", "mA"])
            elif art == "Scatter": visible.update(["FOV (mm)", "Scatter Scale"])
            elif art == "Combined": visible.update(["FOV (mm)", "mA", "Views", "keV", "Show All Artifacts", "Show Steps"])
        else: # NMAR
            # Only show NMAR specific settings
            visible.update(["HU Threshold", "N Angles", "Show Steps"])

        for name, row in self.param_rows.items():
            is_vis = name in visible
            if name in ["Show Steps", "Show All Artifacts"]:
                row['wid'].setVisible(is_vis)
            else:
                row['lbl'].setVisible(is_vis)
                row['wid'].setVisible(is_vis)

    def _div(self): f = QFrame(); f.setObjectName("div"); f.setFixedHeight(1); return f
    def _spin(self, lo, hi, val): w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w
    def _dspin(self, lo, hi, val, step=1.0): w = QDoubleSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step); return w

    def _get_current_params(self):
        w = self.w_params
        return {
            "phantom_id": w["Phantom"].currentIndex() + 1,
            "fov": w["FOV (mm)"].value(), "mA": w["mA"].value(),
            "views": w["Views"].value(), "keV": w["keV"].value(),
            "view_factor": w["View Factor"].value(),
            "shift_mm": w["Shift (mm)"].value(), "break_view": w["Break View"].value(),
            "scatter_scale": w["Scatter Scale"].value(),
            "hu_threshold": w["HU Threshold"].value(), "n_angles": w["N Angles"].value(),
            "show_steps": w["Show Steps"].isChecked(),
            "show_all": w["Show All Artifacts"].isChecked()
        }

    # ── Execution & Plotting ──────────────────────────────────────────────────
    def _run(self, tab_name):
        p = self._get_current_params()
        mode = self.w_params["Artifact"].currentText() if tab_name == "Artifacts" else "NMAR"
        
        controls = self.figs[tab_name]
        controls["btn"].setEnabled(False)
        controls["prog"].setVisible(True)
        controls["log"].append(f"\n[{mode}] Initiating Execution...")

        self.worker = SimWorker(mode, p)
        self.worker.log.connect(lambda m: controls["log"].append(f"  {m}"))
        self.worker.error.connect(lambda e: controls["log"].append(f"ERROR: {e}"))
        self.worker.done.connect(lambda res: self._on_done(res, tab_name))
        self.worker.start()

    def _on_done(self, res, tab_name):
        controls = self.figs[tab_name]
        controls["btn"].setEnabled(True)
        controls["prog"].setVisible(False)
        controls["log"].append("Execution Complete.")
        
        fig, canvas = controls["fig"], controls["canvas"]
        fig.clear()

        def show(ax, img, title, vmin=None, vmax=None, cmap='gray'):
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(title, color=ACCENT, fontsize=10, pad=8)
            vmin = vmin if vmin is not None else float(np.percentile(img, 1))
            vmax = vmax if vmax is not None else float(np.percentile(img, 99))
            cb = fig.colorbar(ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax, fraction=0.04, pad=0.03)
            cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=8); plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

        t = res["type"]
        if t == "aliasing":
            ax = fig.subplots(1, 2)
            show(ax[0], res["detector"], "Detector Under-Sampling")
            show(ax[1], res["view"], "View Under-Sampling")
        elif t == "beam_hardening":
            ax = fig.subplots(1, 2)
            show(ax[0], res["mono"], "Monochromatic", -1000, 500)
            show(ax[1], res["poly"], "Polychromatic", -1000, 500)
        elif t == "motion":
            show(fig.subplots(1, 1), res["motion"], "Motion Artifact", -1000, 500)
        elif t == "noise":
            ax = fig.subplots(1, 3)
            show(ax[0], res["clean"], "Clean", -1000, 500)
            show(ax[1], res["noisy"], "Poisson Noise", -1000, 500)
            show(ax[2], res["diff"], "Difference", cmap='bwr')
        elif t == "scatter":
            ax = fig.subplots(1, 2)
            show(ax[0], res["ideal"], "No Scatter", -1000, 500)
            show(ax[1], res["scatter"], "Scatter Added", -1000, 500)
        elif t == "combined":
            if res.get("show_steps"):
                ax = fig.subplots(2, 3)
                imgs = list(res["images"].values()); lbls = list(res["images"].keys())
                for i in range(min(5, len(imgs))): show(ax[i//3][i%3], imgs[i], lbls[i], -1000, 500)
                show(ax[1][2], res["combined"], "Average Combined", -1000, 500)
            elif res.get("show_all"):
                ax = fig.subplots(2, 3)
                imgs = list(res["images"].values()); lbls = list(res["images"].keys())
                for i in range(5): show(ax[i//3][i%3], imgs[i], lbls[i], -1000, 500)
                show(ax[1][2], res["combined"], "Average Combined", -1000, 500)
            else:
                show(fig.subplots(1, 1), res["combined"], "Combined Artifacts", -1000, 500)
        elif t == "nmar":
            if res.get("show_steps"):
                ax = fig.subplots(2, 3)
                show(ax[0][0], res["original"], "Before NMAR", -1000, 500)
                show(ax[0][1], res["corrected"], "After NMAR", -1000, 500)
                show(ax[0][2], res["corrected"] - res["original"], "Difference Map", cmap='bwr')
                show(ax[1][0], res["sino_norm"], "Normalized Sinogram")
                show(ax[1][1], res["sino_corr"], "Corrected Sinogram")
                show(ax[1][2], res["metal_trace"].astype(np.float32), "Metal Trace", cmap='hot')
            else:
                ax = fig.subplots(1, 2)
                show(ax[0], res["original"], "Before NMAR", -1000, 500)
                show(ax[1], res["corrected"], "After NMAR (Corrected)", -1000, 500)

        fig.patch.set_facecolor(BG); fig.tight_layout()
        canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec_())