# Copyright 2024, GE Precision HealthCare. All rights reserved.
"""
CT Artifact Simulation Panel
Imports functions directly from the existing artifact modules.
Results cached to saved_results/ — run once, browse instantly.
"""

import sys
import os
import numpy as np
from pathlib import Path

# ── Add script folder to path so sibling modules are importable ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Artifact module imports (your existing files) ─────────────────────────────
import gecatsim as xc
from gecatsim.pyfiles.CommonTools import my_path, rawread
import ct_reconstruction as ctr
import phantom_definitions as pd
import aliasing_artifact as aa
import beam_hardening as bh
import motion_artifact
import noise_artifact as na
import scattering as sc

# ── PyQt5 ─────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QFrame, QGridLayout, QProgressBar, QTextEdit, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ── Matplotlib ─────────────────────────────────────────────────────────────────
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CACHE_DIR = Path("saved_results")
CACHE_DIR.mkdir(exist_ok=True)

def cache_save(key, arr):   np.save(CACHE_DIR / f"{key}.npy", arr)
def cache_load(key):
    p = CACHE_DIR / f"{key}.npy"
    return np.load(p) if p.exists() else None

# ══════════════════════════════════════════════════════════════════════════════
# STYLESHEET
# ══════════════════════════════════════════════════════════════════════════════
BG, PANEL, CARD  = "#07090f", "#0e1117", "#131820"
BORDER, MUTED    = "#1e2736", "#5a6a80"
ACCENT, GREEN    = "#4fc3f7", "#43d68a"
TEXT             = "#d0dce8"

QSS = f"""
* {{ font-family:'Segoe UI','Arial',sans-serif; color:{TEXT}; }}
QMainWindow, QWidget {{ background:{BG}; }}
#sidebar  {{ background:{PANEL}; border-right:1px solid {BORDER}; }}
#logo     {{ color:{ACCENT}; font-size:22px; font-weight:700; letter-spacing:3px; }}
#tagline  {{ color:{MUTED};  font-size:9px;  letter-spacing:2px; }}
#section  {{ color:{MUTED};  font-size:9px;  letter-spacing:2px; margin-top:4px; }}
#param    {{ color:{MUTED};  font-size:11px; }}
#div      {{ background:{BORDER}; }}
#cache    {{ color:{MUTED};  font-size:10px; }}
#cache_ok {{ color:{GREEN};  font-size:10px; }}

QComboBox, QDoubleSpinBox, QSpinBox {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:5px;
    padding:5px 9px; font-size:12px; min-height:26px; color:{TEXT};
}}
QComboBox::drop-down {{ border:none; width:18px; }}
QComboBox QAbstractItemView {{
    background:{CARD}; border:1px solid {BORDER};
    selection-background-color:#1a3a5c; color:{TEXT};
}}
QPushButton {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #1a6ad4, stop:1 #1053a8);
    color:#fff; border:none; border-radius:6px;
    padding:10px; font-size:12px; font-weight:600;
    letter-spacing:1px; min-height:38px;
}}
QPushButton:hover   {{ background:#2278e8; }}
QPushButton:pressed {{ background:#0e4391; }}
QPushButton:disabled{{ background:{BORDER}; color:{MUTED}; }}
QPushButton#cached  {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #1a6433, stop:1 #124d27);
}}
QPushButton#cached:hover {{ background:#22854a; }}
QTextEdit {{
    background:#04060a; border:1px solid {BORDER}; border-radius:4px;
    color:{GREEN}; font-family:'Consolas','Courier New',monospace;
    font-size:10px; padding:6px;
}}
QProgressBar {{
    background:{BORDER}; border:none; border-radius:3px; height:4px;
}}
QProgressBar::chunk {{ background:{ACCENT}; border-radius:3px; }}
QScrollBar:vertical {{ background:{BG}; width:8px; }}
QScrollBar::handle:vertical {{
    background:{BORDER}; border-radius:4px; min-height:20px;
}}
"""

# ══════════════════════════════════════════════════════════════════════════════
# WORKER — calls YOUR existing module functions
# ══════════════════════════════════════════════════════════════════════════════
class Worker(QThread):
    done  = pyqtSignal(dict)
    log   = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, artifact, params):
        super().__init__()
        self.artifact = artifact
        self.p = params

    def L(self, msg): self.log.emit(msg)

    def run(self):
        try:
            fn_name = "_run_" + self.artifact.lower().replace(" ", "_")
            result  = getattr(self, fn_name)()
            self.done.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")

    # ── shared setup used by all runners ──────────────────────────────────────
    def _setup(self):
        """Return (ct, size, pixel_size, X, Y) with UI params applied."""
        my_path.add_search_path(".")
        ct = aa.build_common_ct("ui_temp", params=self.p)  # pass UI params directly

        size       = getattr(ct.recon, "imageSize", 512)
        pixel_size = self.p["fov"] / size
        Y, X       = np.ogrid[:size, :size]
        return ct, size, pixel_size, X, Y

    def _phantom_fn(self):
        return pd.generate_phantom_1 if self.p["phantom_id"] == 1 else pd.generate_phantom_2

    # ── ALIASING ──────────────────────────────────────────────────────────────
    def _run_aliasing(self):
        pid    = self.p["phantom_id"]
        factor = self.p.get("view_factor", 4)
        k_det  = f"alias_det_p{pid}_fov{self.p['fov']}_v{self.p['views']}"
        k_view = f"alias_view_p{pid}_fov{self.p['fov']}_v{self.p['views']}_f{factor}"

        ct, size, px, X, Y = self._setup()
        self._phantom_fn()(size, px, X, Y)
        self.L("Forward scan...")
        ct.resultsName = "aliasing_base"
        ct.run_all()
        n_views = ct.protocol.viewCount
        n_cells = ct.scanner.detectorColCount * ct.scanner.detectorRowCount
        sino = rawread("aliasing_base.prep", [n_views, n_cells], 'float')

        self.L("Detector under-sampling...")
        sino_det = aa.create_detector_undersampling(sino)
        sino_det.astype(np.float32).tofile("aliasing_detector.prep")
        ct_d = aa.build_common_ct("aliasing_detector", params=self.p)
        img_det = aa.reconstruct_from_prep(ct_d, "aliasing_detector.prep", "aliasing_detector")
        cache_save(k_det, img_det)

        self.L("View under-sampling...")
        sino_view = aa.create_view_undersampling(sino, factor=factor)
        sino_view.astype(np.float32).tofile("aliasing_view.prep")
        ct_v = aa.build_common_ct("aliasing_view", params=self.p)
        img_view = aa.reconstruct_from_prep(ct_v, "aliasing_view.prep", "aliasing_view")
        cache_save(k_view, img_view)

        return {"type": "aliasing", "detector": img_det, "view": img_view}

    # ── BEAM HARDENING ────────────────────────────────────────────────────────
    def _run_beam_hardening(self):
        pid    = self.p["phantom_id"]
        k_mono = f"bh_mono_p{pid}_fov{self.p['fov']}_kev{self.p['keV']}"
        k_poly = f"bh_poly_p{pid}_fov{self.p['fov']}"

        my_path.add_search_path(".")
        ct_tmp, size, px, X, Y = self._setup()
        self.L("Running beam hardening simulation...")
        img_mono, img_poly = bh.simulate_bh_one_phantom(pid, size, px, X, Y, params=self.p)
        cache_save(k_mono, img_mono)
        cache_save(k_poly, img_poly)

        return {"type": "beam_hardening",
                "mono": img_mono, "poly": img_poly, "phantom_id": pid}

    # ── MOTION ────────────────────────────────────────────────────────────────
    def _run_motion(self):
        pid      = self.p["phantom_id"]
        shift_mm = self.p.get("shift_mm", 1.4)
        bv       = self.p.get("break_view", 700)
        key      = f"motion_p{pid}_fov{self.p['fov']}_s{shift_mm}_b{bv}"

        my_path.add_search_path(".")
        ct, size, px, X, Y = self._setup()
        self.L(f"Running motion simulation (shift={shift_mm}mm, break@{bv})...")
        img = motion_artifact.run_motion_artifact(
            ct, size, px, X, Y,
            phantom_fn=self._phantom_fn(),
            shift_mm=shift_mm,
            break_view=bv,
        )
        cache_save(key, img)

        return {"type": "motion", "motion": img}

    # ── NOISE ─────────────────────────────────────────────────────────────────
    def _run_noise(self):
        pid     = self.p["phantom_id"]
        k_clean = f"noise_clean_p{pid}_fov{self.p['fov']}"
        k_noisy = f"noise_noisy_p{pid}_fov{self.p['fov']}_mA{self.p['mA']}"

        my_path.add_search_path(".")
        ct_tmp, size, px, X, Y = self._setup()
        self.L("Running noise simulation...")
        name = "Phantom 1" if pid == 1 else "Phantom 2"
        img_clean, img_noisy, _ = na.simulate_one_phantom(
            pid, name, size, px, X, Y, noisy_mA=self.p["mA"], params=self.p
        )
        cache_save(k_clean, img_clean)
        cache_save(k_noisy, img_noisy)

        return {"type": "noise", "clean": img_clean,
                "noisy": img_noisy, "diff": img_noisy - img_clean}

    # ── SCATTER ───────────────────────────────────────────────────────────────
    def _run_scatter(self):
        pid   = self.p["phantom_id"]
        scale = self.p.get("scatter_scale", 1000.0)
        k_i   = f"scatter_ideal_p{pid}_fov{self.p['fov']}_kev{self.p['keV']}"
        k_s   = f"scatter_art_p{pid}_fov{self.p['fov']}_sc{scale}"

        my_path.add_search_path(".")
        ct_tmp, size, px, X, Y = self._setup()
        self.L("Running scatter simulation...")
        img_ideal, img_scatter = sc.simulate_scatter_one_phantom(
            pid, size, px, X, Y, scatter_scale=scale, params=self.p
        )
        cache_save(k_i, img_ideal)
        cache_save(k_s, img_scatter)

        return {"type": "scatter", "ideal": img_ideal, "scatter": img_scatter}

    # ── COMBINED — average all 5 artifact keys for this phantom from cache ────
    def _run_combined(self):
        pid = self.p["phantom_id"]
        # Map: display label → cache key prefix to search for
        wanted = {
            "Beam Hardening (poly)": f"bh_poly_p{pid}_",
            "Scatter":               f"scatter_art_p{pid}_",
            "Noise (noisy)":         f"noise_noisy_p{pid}_",
            "Motion":                f"motion_p{pid}_",
            "Aliasing (detector)":   f"alias_det_p{pid}_",
        }
        images, missing = {}, []
        for label, prefix in wanted.items():
            hits = sorted(CACHE_DIR.glob(f"{prefix}*.npy"))
            if hits:
                images[label] = np.load(hits[-1])   # most recent match
                self.L(f"✓ {label} — loaded from saved results")
            else:
                missing.append(label)
        if missing:
            raise RuntimeError(
                f"Missing saved results for: {', '.join(missing)}\n"
                "Run each artifact type first so they get saved to saved_results/."
            )
        combined = np.mean(np.stack(list(images.values()), axis=0), axis=0)
        return {"type": "combined", "images": images, "combined": combined, "phantom_id": pid}


# ══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CANVAS
# ══════════════════════════════════════════════════════════════════════════════
class Canvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class App(QMainWindow):
    ARTIFACTS = ["Aliasing", "Beam Hardening", "Motion", "Noise", "Scatter", "Combined"]
    PHANTOMS  = ["Phantom 1 — Water & Iron", "Phantom 2 — Plexi & Silver"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Artifact Viewer")
        self.resize(1380, 840)
        self.setStyleSheet(QSS)
        self.worker       = None
        self.spec_widgets = {}

        # Build UI — viewer first (creates self.canvas, self.log_box), then sidebar.
        root = QWidget(); self.setCentralWidget(root)
        row  = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0); row.setSpacing(0)

        viewer  = self._build_viewer()   # creates self.canvas, self.log_box
        sidebar = self._build_sidebar()  # safe to call now

        row.addWidget(sidebar)
        row.addWidget(viewer, stretch=1)

    # ── sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        side = QWidget(); side.setObjectName("sidebar"); side.setFixedWidth(280)
        v = QVBoxLayout(side); v.setContentsMargins(20, 20, 20, 20); v.setSpacing(8)

        for txt, obj in [("CT·SIM", "logo"), ("ARTIFACT ANALYSIS SUITE", "tagline")]:
            l = QLabel(txt); l.setObjectName(obj); l.setAlignment(Qt.AlignCenter)
            v.addWidget(l)
        v.addWidget(self._div())

        # Artifact & Phantom selectors
        for label, attr, items in [
            ("Artifact", "combo_artifact", self.ARTIFACTS),
            ("Phantom",  "combo_phantom",  self.PHANTOMS),
        ]:
            v.addWidget(self._sec(label))
            cb = QComboBox(); cb.addItems(items)
            setattr(self, attr, cb); v.addWidget(cb)
        self.combo_artifact.currentTextChanged.connect(self._on_artifact_changed)

        v.addWidget(self._div())

        # Shared scanner parameters
        v.addWidget(self._sec("Scanner Parameters"))
        g = QGridLayout(); g.setColumnStretch(1, 1); g.setSpacing(6)
        self.spin_fov   = self._dspin(100, 700, 300, 10)
        self.spin_mA    = self._spin(10, 1200, 800)
        self.spin_views = self._spin(100, 3000, 1000)
        self.spin_keV   = self._spin(40, 140, 70)
        for r, (lbl, w) in enumerate([
            ("FOV (mm)", self.spin_fov), ("mA",    self.spin_mA),
            ("Views",    self.spin_views), ("keV", self.spin_keV),
        ]):
            pl = QLabel(lbl); pl.setObjectName("param")
            g.addWidget(pl, r, 0); g.addWidget(w, r, 1)
        v.addLayout(g)

        v.addWidget(self._div())

        # Artifact-specific parameters
        v.addWidget(self._sec("Artifact Parameters"))
        self.spec_grid = QGridLayout()
        self.spec_grid.setColumnStretch(1, 1); self.spec_grid.setSpacing(6)
        v.addLayout(self.spec_grid)
        self._refresh_spec()          # log_box now exists, safe to call

        v.addStretch()

        # ── Loaded Scenario dropdown ──────────────────────────────────────────
        v.addWidget(self._sec("Loaded Scenario"))
        self.combo_scenario = QComboBox()
        self.combo_scenario.currentIndexChanged.connect(self._on_scenario_selected)
        v.addWidget(self.combo_scenario)
        self._refresh_scenario_list()   # populate from saved_results/

        v.addWidget(self._div())

        self.btn = QPushButton("▶  RUN SIMULATION")
        self.btn.clicked.connect(self._run); v.addWidget(self.btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0); self.progress.setVisible(False)
        v.addWidget(self.progress)

        return side

    # ── scenario dropdown helpers ─────────────────────────────────────────────
    def _refresh_scenario_list(self):
        """Repopulate the Loaded Scenario dropdown from saved_results/*.npy"""
        self.combo_scenario.blockSignals(True)
        self.combo_scenario.clear()
        self.combo_scenario.addItem("— none —", userData=None)
        for p in sorted(CACHE_DIR.glob("*.npy")):
            self.combo_scenario.addItem(p.stem, userData=str(p))
        self.combo_scenario.blockSignals(False)

    def _on_scenario_selected(self, idx):
        path = self.combo_scenario.itemData(idx)
        if not path:
            return
        arr = np.load(path)
        # Infer a display type from the filename stem
        stem = Path(path).stem
        if stem.startswith("bh_mono"):
            result = {"type": "beam_hardening", "mono": arr,
                      "poly": arr, "phantom_id": 1}
        elif stem.startswith("bh_poly"):
            result = {"type": "beam_hardening", "mono": arr,
                      "poly": arr, "phantom_id": 1}
        elif stem.startswith("scatter_art"):
            result = {"type": "scatter", "ideal": arr, "scatter": arr}
        elif stem.startswith("scatter_ideal"):
            result = {"type": "scatter", "ideal": arr, "scatter": arr}
        elif stem.startswith("noise_noisy"):
            result = {"type": "noise", "clean": arr, "noisy": arr,
                      "diff": np.zeros_like(arr)}
        elif stem.startswith("noise_clean"):
            result = {"type": "noise", "clean": arr, "noisy": arr,
                      "diff": np.zeros_like(arr)}
        elif stem.startswith("motion"):
            result = {"type": "motion", "motion": arr}
        elif stem.startswith("alias_det"):
            result = {"type": "aliasing", "detector": arr, "view": arr}
        elif stem.startswith("alias_view"):
            result = {"type": "aliasing", "detector": arr, "view": arr}
        else:
            result = {"type": "motion", "motion": arr}  # fallback: show as single image
        self.log_box.append(f"\n📂 Loaded scenario: {Path(path).stem}")
        self._plot(result)

    # ── viewer ────────────────────────────────────────────────────────────────
    def _build_viewer(self):
        w = QWidget(); v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        split = QSplitter(Qt.Vertical)

        self.canvas = Canvas()
        split.addWidget(self.canvas)

        log_w = QWidget(); log_w.setStyleSheet(f"background:{PANEL};")
        lv = QVBoxLayout(log_w); lv.setContentsMargins(10, 6, 10, 10)
        ll = QLabel("SIMULATION LOG"); ll.setObjectName("tagline"); lv.addWidget(ll)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(130); lv.addWidget(self.log_box)
        split.addWidget(log_w)
        split.setSizes([680, 160]); v.addWidget(split)
        return w

    # ── small widget factories ────────────────────────────────────────────────
    def _div(self):
        f = QFrame(); f.setObjectName("div"); f.setFixedHeight(1); return f
    def _sec(self, t):
        l = QLabel(t); l.setObjectName("section"); return l
    def _spin(self, lo, hi, val):
        w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w
    def _dspin(self, lo, hi, val, step=1.0):
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setValue(val)
        w.setSingleStep(step); return w

    # ── artifact-specific param panel ─────────────────────────────────────────
    def _clear_spec(self):
        while self.spec_grid.count():
            item = self.spec_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.spec_widgets.clear()

    def _add_spec(self, row, label, widget):
        pl = QLabel(label); pl.setObjectName("param")
        self.spec_grid.addWidget(pl, row, 0)
        self.spec_grid.addWidget(widget, row, 1)
        self.spec_widgets[label] = widget

    def _refresh_spec(self):
        self._clear_spec()
        a = self.combo_artifact.currentText()
        if a == "Aliasing":
            self._add_spec(0, "View Factor",   self._spin(2, 16, 4))
        elif a == "Motion":
            self._add_spec(0, "Shift (mm)",    self._dspin(0.1, 20, 1.4, 0.1))
            self._add_spec(1, "Break View",    self._spin(1, 999, 700))
        elif a == "Scatter":
            self._add_spec(0, "Scatter Scale", self._dspin(10, 5000, 1000, 100))

    def _on_artifact_changed(self):
        self._refresh_spec()

    # ── collect UI params ─────────────────────────────────────────────────────
    def _params(self):
        a, sw = self.combo_artifact.currentText(), self.spec_widgets
        p = {
            "fov":        self.spin_fov.value(),
            "mA":         self.spin_mA.value(),
            "views":      self.spin_views.value(),
            "keV":        self.spin_keV.value(),
            "phantom_id": self.combo_phantom.currentIndex() + 1,
        }
        if "View Factor"   in sw: p["view_factor"]   = sw["View Factor"].value()
        if "Shift (mm)"    in sw: p["shift_mm"]      = sw["Shift (mm)"].value()
        if "Break View"    in sw: p["break_view"]    = sw["Break View"].value()
        if "Scatter Scale" in sw: p["scatter_scale"] = sw["Scatter Scale"].value()
        return p

    # ── run ───────────────────────────────────────────────────────────────────
    def _run(self):
        a, p = self.combo_artifact.currentText(), self._params()
        self.log_box.append(
            f"\n▶ [{a}]  FOV={p['fov']} mm  mA={p['mA']}  "
            f"Views={p['views']}  {p['keV']} keV  Phantom {p['phantom_id']}"
        )
        self.btn.setEnabled(False); self.progress.setVisible(True)
        self.worker = Worker(a, p)
        self.worker.log.connect(lambda m: self.log_box.append(f"  {m}"))
        self.worker.done.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, result):
        self.btn.setEnabled(True); self.progress.setVisible(False)
        self.log_box.append("✓ Done — rendering")
        self._plot(result)
        self._refresh_scenario_list()   # update dropdown with newly saved files

    def _on_error(self, msg):
        self.btn.setEnabled(True); self.progress.setVisible(False)
        self.log_box.append(f"✖ ERROR:\n{msg}")

    # ── plot ──────────────────────────────────────────────────────────────────
    def _plot(self, result):
        fig = self.canvas.fig; fig.clear()

        def show(ax, img, title, vmin=None, vmax=None, cmap='gray', clabel='HU'):
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(title, color=ACCENT, fontsize=9, pad=5, fontfamily="monospace")
            vmin = vmin if vmin is not None else float(np.percentile(img, 1))
            vmax = vmax if vmax is not None else float(np.percentile(img, 99))
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cb.set_label(clabel, color=MUTED, fontsize=7)
            cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
            cb.outline.set_edgecolor(BORDER)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

        TITLES = {
            "aliasing":       "ALIASING ARTIFACTS",
            "beam_hardening": "BEAM HARDENING ARTIFACT",
            "motion":         "MOTION ARTIFACT",
            "noise":          "NOISE ARTIFACT",
            "scatter":        "SCATTER ARTIFACT",
            "combined":       "COMBINED — ALL 5 ARTIFACTS",
        }
        t = result["type"]

        if t == "aliasing":
            axes = fig.subplots(1, 2)
            show(axes[0], result["detector"], "(a) Detector Under-Sampling")
            show(axes[1], result["view"],      "(b) View Under-Sampling")

        elif t == "beam_hardening":
            lbl  = "Water & Iron" if result["phantom_id"] == 1 else "Plexi & Silver"
            axes = fig.subplots(1, 2)
            show(axes[0], result["mono"], f"Ideal — Monochromatic ({lbl})",        -1000, 500)
            show(axes[1], result["poly"], f"Polychromatic — Beam Hardening ({lbl})", -1000, 500)

        elif t == "motion":
            show(fig.subplots(1, 1), result["motion"],
                 "Motion Artifact — Sinogram Splice", -1000, 500)

        elif t == "noise":
            axes = fig.subplots(1, 3)
            show(axes[0], result["clean"], "Clean (No Noise)",  -1000, 500)
            show(axes[1], result["noisy"], "Poisson Noise",      -1000, 500)
            v = float(np.percentile(np.abs(result["diff"]), 99))
            show(axes[2], result["diff"],  "Difference Map", -v, v, cmap='bwr', clabel='ΔHU')

        elif t == "scatter":
            axes = fig.subplots(1, 2)
            show(axes[0], result["ideal"],   "Ideal — No Scatter", -1000, 500)
            show(axes[1], result["scatter"], "Scatter Artifact",    -1000, 500)

        elif t == "combined":
            imgs   = list(result["images"].values())
            labels = list(result["images"].keys())
            combined = result["combined"]
            axes = fig.subplots(2, 3)
            for i in range(5):
                ax = axes[i // 3][i % 3]
                show(ax, imgs[i], labels[i], -1000, 500)
            show(axes[1][2], combined, "Combined\n(Average of All 5)", -1000, 500)

        fig.suptitle(TITLES.get(t, ""), color=ACCENT, fontsize=12,
                     fontfamily="monospace", fontweight="bold", y=1.01)
        fig.patch.set_facecolor(BG)
        fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App(); win.show()
    sys.exit(app.exec_())