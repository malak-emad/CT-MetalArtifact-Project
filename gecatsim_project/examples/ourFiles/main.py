# Copyright 2024, GE Precision HealthCare. All rights reserved.
import sys, os, traceback
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path, rawread
import ct_reconstruction as ctr
import phantom_definitions as pd
import aliasing_artifact as aa
import beam_hardening as bh
import motion_artifact
import noise_artifact as na
import scattering as sc
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QFrame, QGridLayout,
    QProgressBar, QTextEdit, QSplitter, QSizePolicy, QPushButton, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

CACHE_DIR = Path("saved_results")
CACHE_DIR.mkdir(exist_ok=True)

def cache_save(key, arr):   np.save(CACHE_DIR / f"{key}.npy", arr)
def cache_exists(key):      return (CACHE_DIR / f"{key}.npy").exists()
def cache_load(key):        return np.load(CACHE_DIR / f"{key}.npy") if cache_exists(key) else None

BG, PANEL, CARD  = "#0b0f19", "#111827", "#1f2937"
BORDER, MUTED    = "#374151", "#9ca3af"
ACCENT, GREEN    = "#3b82f6", "#10b981"
TEXT             = "#f3f4f6"

QSS = f"""
* {{ font-family: 'Segoe UI', system-ui, sans-serif; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
#logo {{ color: {ACCENT}; font-size: 24px; font-weight: 800; letter-spacing: 2px; }}
#tagline {{ color: {MUTED}; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }}
#section {{ color: {ACCENT}; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-top: 10px; }}
#param {{ color: {MUTED}; font-size: 12px; }}
#div {{ background: {BORDER}; }}

QComboBox, QDoubleSpinBox, QSpinBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 10px; font-size: 13px; min-height: 24px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {BORDER}; selection-background-color: {ACCENT}; }}

QCheckBox {{ color: {TEXT}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {BORDER}; border-radius: 4px; background: {CARD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2563eb, stop:1 #1d4ed8);
    color: #ffffff; border: none; border-radius: 6px;
    padding: 12px; font-size: 13px; font-weight: bold; letter-spacing: 1px;
}}
QPushButton:hover {{ background: #3b82f6; }}
QPushButton:pressed {{ background: #1e40af; }}
QPushButton:disabled {{ background: {BORDER}; color: {MUTED}; }}

QTextEdit {{
    background: #000000; border: 1px solid {BORDER}; border-radius: 6px;
    color: {GREEN}; font-family: 'Consolas', monospace; font-size: 11px; padding: 8px;
}}
QProgressBar {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""

class Worker(QThread):
    done, log, error = pyqtSignal(dict), pyqtSignal(str), pyqtSignal(str)

    def __init__(self, artifact, params, keys):
        super().__init__()
        self.artifact = artifact
        self.p = params
        self.keys = keys

    def run(self):
        try:
            fn = getattr(self, "_run_" + self.artifact.lower().replace(" ", "_"))
            self.done.emit(fn())
        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _setup(self):
        my_path.add_search_path(".")
        ct = aa.build_common_ct("ui_temp", params=self.p)
        sz = getattr(ct.recon, "imageSize", 512)
        px = self.p["fov"] / sz
        Y, X = np.ogrid[:sz, :sz]
        return ct, sz, px, X, Y

    def _pfn(self): return pd.generate_phantom_1 if self.p["phantom_id"] == 1 else pd.generate_phantom_2

    def _run_aliasing(self):
        f = self.p.get("view_factor", 4)
        ct, sz, px, X, Y = self._setup()
        self._pfn()(sz, px, X, Y)
        self.log.emit("Running aliasing forward scan...")
        ct.resultsName = "aliasing_base"; ct.run_all()
        sino = rawread("aliasing_base.prep", [ct.protocol.viewCount, ct.scanner.detectorColCount * ct.scanner.detectorRowCount], 'float')
        
        self.log.emit("Reconstructing detector & view undersampling...")
        s_d = aa.create_detector_undersampling(sino); s_d.astype(np.float32).tofile("aliasing_detector.prep")
        img_d = aa.reconstruct_from_prep(aa.build_common_ct("aliasing_detector", params=self.p), "aliasing_detector.prep", "aliasing_detector")
        
        s_v = aa.create_view_undersampling(sino, factor=f); s_v.astype(np.float32).tofile("aliasing_view.prep")
        img_v = aa.reconstruct_from_prep(aa.build_common_ct("aliasing_view", params=self.p), "aliasing_view.prep", "aliasing_view")
        
        cache_save(self.keys[0], img_d); cache_save(self.keys[1], img_v)
        return {"type": "aliasing", "detector": img_d, "view": img_v}

    def _run_beam_hardening(self):
        ct_tmp, sz, px, X, Y = self._setup()
        self.log.emit("Simulating beam hardening...")
        i_mono, i_poly = bh.simulate_bh_one_phantom(self.p["phantom_id"], sz, px, X, Y, params=self.p)
        cache_save(self.keys[0], i_mono)
        cache_save(self.keys[1], i_poly)
        return {"type": "beam_hardening", "mono": i_mono, "poly": i_poly, "phantom_id": self.p["phantom_id"]}

    def _run_motion(self):
        shift, bv = self.p.get("shift_mm", 1.4), self.p.get("break_view", 700)
        ct, sz, px, X, Y = self._setup()
        self.log.emit(f"Simulating motion (shift={shift}mm)...")
        img = motion_artifact.run_motion_artifact(ct, sz, px, X, Y, phantom_fn=self._pfn(), shift_mm=shift, break_view=bv)
        cache_save(self.keys[0], img)
        return {"type": "motion", "motion": img}

    def _run_noise(self):
        ct_tmp, sz, px, X, Y = self._setup()
        self.log.emit("Simulating noise...")
        i_cln, i_nsy, _ = na.simulate_one_phantom(self.p["phantom_id"], f"Phantom {self.p['phantom_id']}", sz, px, X, Y, noisy_mA=self.p["mA"], params=self.p)
        cache_save(self.keys[0], i_cln)
        cache_save(self.keys[1], i_nsy)
        return {"type": "noise", "clean": i_cln, "noisy": i_nsy, "diff": i_nsy - i_cln}

    def _run_scatter(self):
        scale = self.p.get("scatter_scale", 1000.0)
        ct_tmp, sz, px, X, Y = self._setup()
        self.log.emit("Simulating scatter...")
        ideal, sc_art = sc.simulate_scatter_one_phantom(self.p["phantom_id"], sz, px, X, Y, scatter_scale=scale, params=self.p)
        cache_save(self.keys[0], ideal)
        cache_save(self.keys[1], sc_art)
        return {"type": "scatter", "ideal": ideal, "scatter": sc_art}

    def _run_combined(self):
        base = f"p{self.p['phantom_id']}_f{self.p['fov']}_m{self.p['mA']}_v{self.p['views']}_k{self.p['keV']}"
        images = {}

        # 1. Beam Hardening
        bh_key = f"bh_poly_{base}"
        if cache_exists(bh_key):
            images["Beam Hardening"] = cache_load(bh_key)
            self.log.emit("Beam Hardening loaded from cache")
        else:
            self.log.emit("▶ Computing missing Beam Hardening...")
            self.keys = [f"bh_mono_{base}", bh_key]
            images["Beam Hardening"] = self._run_beam_hardening()["poly"]

        # 2. Scatter
        sc = self.p.get("scatter_scale", 1000.0)
        sc_key = f"scatter_art_{base}_sc{sc}"
        if cache_exists(sc_key):
            images["Scatter"] = cache_load(sc_key)
            self.log.emit("Scatter loaded from cache")
        else:
            self.log.emit("▶ Computing missing Scatter...")
            self.keys = [f"scatter_idl_{base}", sc_key]
            images["Scatter"] = self._run_scatter()["scatter"]

        # 3. Noise
        nsy_key = f"noise_nsy_{base}"
        if cache_exists(nsy_key):
            images["Noise"] = cache_load(nsy_key)
            self.log.emit("Noise loaded from cache")
        else:
            self.log.emit("▶ Computing missing Noise...")
            self.keys = [f"noise_cln_{base}", nsy_key]
            images["Noise"] = self._run_noise()["noisy"]

        # 4. Motion
        s, b = self.p.get("shift_mm", 1.4), self.p.get("break_view", 700)
        mot_key = f"motion_{base}_s{s}_b{b}"
        if cache_exists(mot_key):
            images["Motion"] = cache_load(mot_key)
            self.log.emit("Motion loaded from cache")
        else:
            self.log.emit("▶ Computing missing Motion...")
            self.keys = [mot_key]
            images["Motion"] = self._run_motion()["motion"]

        # 5. Aliasing
        f = self.p.get("view_factor", 4)
        alias_key = f"alias_det_{base}_vf{f}"
        if cache_exists(alias_key):
            images["Aliasing"] = cache_load(alias_key)
            self.log.emit("Aliasing loaded from cache")
        else:
            self.log.emit("▶ Computing missing Aliasing...")
            self.keys = [alias_key, f"alias_view_{base}_vf{f}"]
            images["Aliasing"] = self._run_aliasing()["detector"]

        self.log.emit("All artifacts ready. Generating combined image...")
        combined = np.mean(np.stack(list(images.values()), axis=0), axis=0)
        return {
            "type": "combined", 
            "images": images, 
            "combined": combined, 
            "phantom_id": self.p["phantom_id"],
            "show_all": self.p.get("show_all", False)  
        }


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Artifact Viewer")
        self.resize(1400, 850); self.setStyleSheet(QSS)
        self.worker = None; self.spec_widgets = {}

        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)

        # Viewer (Right Side)
        viewer_w = QWidget(); v_lay = QVBoxLayout(viewer_w); v_lay.setContentsMargins(0,0,0,0)
        split = QSplitter(Qt.Vertical)
        
        self.fig = Figure(facecolor=BG)
        self.canvas = FigureCanvas(self.fig); self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        split.addWidget(self.canvas)

        log_w = QWidget(); log_w.setStyleSheet(f"background: {PANEL};")
        l_lay = QVBoxLayout(log_w); l_lay.setContentsMargins(15, 10, 15, 15)
        l_lbl = QLabel("SIMULATION LOG"); l_lbl.setObjectName("tagline"); l_lay.addWidget(l_lbl)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(120)
        l_lay.addWidget(self.log_box); split.addWidget(log_w)
        
        split.setSizes([700, 150]); v_lay.addWidget(split)

        # Sidebar (Left Side)
        sidebar = QWidget(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(300)
        s_lay = QVBoxLayout(sidebar); s_lay.setContentsMargins(20, 25, 20, 25); s_lay.setSpacing(10)
        
        logo = QLabel("CT·SIM"); logo.setObjectName("logo"); logo.setAlignment(Qt.AlignCenter); s_lay.addWidget(logo)
        tag = QLabel("Artifact Simulation"); tag.setObjectName("tagline"); tag.setAlignment(Qt.AlignCenter); s_lay.addWidget(tag)
        s_lay.addWidget(self._div())

        self.combo_artifact = QComboBox(); self.combo_artifact.addItems(["Aliasing", "Beam Hardening", "Motion", "Noise", "Scatter", "Combined"])
        self.combo_artifact.currentTextChanged.connect(self._refresh_spec)
        self.combo_phantom  = QComboBox(); self.combo_phantom.addItems(["Phantom 1 — Water & Iron", "Phantom 2 — Plexi & Silver"])
        s_lay.addWidget(self._sec("Configuration")); s_lay.addWidget(self.combo_artifact); s_lay.addWidget(self.combo_phantom)
        
        self.spin_fov   = self._dspin(100, 700, 300, 10)
        self.spin_mA    = self._spin(10, 1200, 800)
        self.spin_views = self._spin(100, 3000, 1000)
        self.spin_keV   = self._spin(40, 140, 70)
        
        g = QGridLayout(); g.setSpacing(8)
        for r, (lbl, w) in enumerate([("FOV (mm)", self.spin_fov), ("mA", self.spin_mA), ("Views", self.spin_views), ("keV", self.spin_keV)]):
            pl = QLabel(lbl); pl.setObjectName("param")
            g.addWidget(pl, r, 0); g.addWidget(w, r, 1)
        s_lay.addWidget(self._sec("Scanner Params")); s_lay.addLayout(g)

        s_lay.addWidget(self._sec("Artifact Params"))
        self.spec_grid = QGridLayout(); self.spec_grid.setSpacing(8); s_lay.addLayout(self.spec_grid)
        self._refresh_spec()
        s_lay.addStretch()

        s_lay.addWidget(self._div())
        self.btn = QPushButton("▶ RUN SIMULATION")
        self.btn.clicked.connect(self._run)
        s_lay.addWidget(self.btn)
        
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setVisible(False)
        s_lay.addWidget(self.progress)

        layout.addWidget(sidebar); layout.addWidget(viewer_w, stretch=1)

    def _div(self): f = QFrame(); f.setObjectName("div"); f.setFixedHeight(1); return f
    def _sec(self, t): l = QLabel(t); l.setObjectName("section"); return l
    def _spin(self, lo, hi, val): w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w
    def _dspin(self, lo, hi, val, step=1.0): w = QDoubleSpinBox(); w.setRange(lo, hi); w.setValue(val); w.setSingleStep(step); return w

    def _refresh_spec(self):
        while self.spec_grid.count():
            item = self.spec_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.spec_widgets.clear()
        
        a = self.combo_artifact.currentText()
        specs = []
        if a == "Aliasing": specs = [("View Factor", self._spin(2, 16, 4))]
        elif a == "Motion": specs = [("Shift (mm)", self._dspin(0.1, 20, 1.4, 0.1)), ("Break View", self._spin(1, 999, 700))]
        elif a == "Scatter": specs = [("Scatter Scale", self._dspin(10, 5000, 1000, 100))]
        elif a == "Combined": 
            cb = QCheckBox("");
            cb.setChecked(False) # Default is False (Single large image)
            specs = [("Show All Artifacts", cb)]
        
        for r, (lbl, w) in enumerate(specs):
            pl = QLabel(lbl); pl.setObjectName("param")
            self.spec_grid.addWidget(pl, r, 0); self.spec_grid.addWidget(w, r, 1)
            self.spec_widgets[lbl] = w

    def _params(self):
        p = {
            "fov": self.spin_fov.value(), "mA": self.spin_mA.value(),
            "views": self.spin_views.value(), "keV": self.spin_keV.value(),
            "phantom_id": self.combo_phantom.currentIndex() + 1,
        }
        sw = self.spec_widgets
        if "View Factor" in sw: p["view_factor"] = sw["View Factor"].value()
        if "Shift (mm)" in sw: p["shift_mm"] = sw["Shift (mm)"].value()
        if "Break View" in sw: p["break_view"] = sw["Break View"].value()
        if "Scatter Scale" in sw: p["scatter_scale"] = sw["Scatter Scale"].value()
        if "Show All Artifacts" in sw: p["show_all"] = sw["Show All Artifacts"].isChecked()
        return p

    def _get_expected_cache_keys(self, a, p):
        base = f"p{p['phantom_id']}_f{p['fov']}_m{p['mA']}_v{p['views']}_k{p['keV']}"
        
        if a == "Aliasing":       return [f"alias_det_{base}_vf{p.get('view_factor',4)}", f"alias_view_{base}_vf{p.get('view_factor',4)}"]
        elif a == "Beam Hardening": return [f"bh_mono_{base}", f"bh_poly_{base}"]
        elif a == "Motion":       return [f"motion_{base}_s{p.get('shift_mm',1.4)}_b{p.get('break_view',700)}"]
        elif a == "Noise":        return [f"noise_cln_{base}", f"noise_nsy_{base}"]
        elif a == "Scatter":      return [f"scatter_idl_{base}", f"scatter_art_{base}_sc{p.get('scatter_scale',1000.0)}"]
        return []

    def _reconstruct_from_cache(self, a, p, keys):
        pid = p["phantom_id"]
        if a == "Aliasing":       return {"type": "aliasing", "detector": cache_load(keys[0]), "view": cache_load(keys[1])}
        elif a == "Beam Hardening": return {"type": "beam_hardening", "mono": cache_load(keys[0]), "poly": cache_load(keys[1]), "phantom_id": pid}
        elif a == "Motion":       return {"type": "motion", "motion": cache_load(keys[0])}
        elif a == "Noise":
            c, n = cache_load(keys[0]), cache_load(keys[1])
            return {"type": "noise", "clean": c, "noisy": n, "diff": n - c}
        elif a == "Scatter":      return {"type": "scatter", "ideal": cache_load(keys[0]), "scatter": cache_load(keys[1])}

    def _run(self):
        a = self.combo_artifact.currentText()
        p = self._params()

        keys = self._get_expected_cache_keys(a, p)
        if a != "Combined" and keys and all(cache_exists(k) for k in keys):
            self.log_box.append(f"\n INSTANT LOAD")
            result = self._reconstruct_from_cache(a, p, keys)
            self._plot(result)
            return

        self.log_box.append(f"\n COMPUTING [{a}] FOV={p['fov']} mA={p['mA']} Views={p['views']} {p['keV']}keV")
        self.btn.setEnabled(False)
        self.progress.setVisible(True)
        
        self.worker = Worker(a, p, keys)
        self.worker.log.connect(lambda m: self.log_box.append(f"  {m}"))
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(lambda e: self.log_box.append(f"✖ ERROR: {e}"))
        self.worker.start()

    def _on_done(self, result):
        self.btn.setEnabled(True)
        self.progress.setVisible(False)
        self.log_box.append("Process Complete.")
        self._plot(result)

    def _plot(self, res):
        self.fig.clear()
        
        def show(ax, img, title, vmin=None, vmax=None, cmap='gray'):
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(title, color=ACCENT, fontsize=10, pad=8, fontfamily="Segoe UI")
            vmin = vmin if vmin is not None else float(np.percentile(img, 1))
            vmax = vmax if vmax is not None else float(np.percentile(img, 99))
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=8)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

        t = res["type"]
        if t == "aliasing":
            ax = self.fig.subplots(1, 2); show(ax[0], res["detector"], "Detector Under-Sampling"); show(ax[1], res["view"], "View Under-Sampling")
        elif t == "beam_hardening":
            ax = self.fig.subplots(1, 2); show(ax[0], res["mono"], "Monochromatic", -1000, 500); show(ax[1], res["poly"], "Polychromatic", -1000, 500)
        elif t == "motion":
            show(self.fig.subplots(1, 1), res["motion"], "Motion Artifact", -1000, 500)
        elif t == "noise":
            ax = self.fig.subplots(1, 3); show(ax[0], res["clean"], "Clean", -1000, 500); show(ax[1], res["noisy"], "Poisson Noise", -1000, 500); show(ax[2], res["diff"], "Difference", cmap='bwr')
        elif t == "scatter":
            ax = self.fig.subplots(1, 2); show(ax[0], res["ideal"], "No Scatter", -1000, 500); show(ax[1], res["scatter"], "Scatter Added", -1000, 500)
        elif t == "combined":
            if res.get("show_all", False):
                imgs, lbls = list(res["images"].values()), list(res["images"].keys())
                ax = self.fig.subplots(2, 3)
                for i in range(5): show(ax[i // 3][i % 3], imgs[i], lbls[i], -1000, 500)
                show(ax[1][2], res["combined"], "Average Combined", -1000, 500)
            else:
                show(self.fig.subplots(1, 1), res["combined"], "Combined Artifacts (Average of All 5)", -1000, 500)

        self.fig.patch.set_facecolor(BG); self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec_())