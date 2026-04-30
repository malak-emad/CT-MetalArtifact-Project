# Copyright 2024, GE Precision HealthCare. All rights reserved.
import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QFrame, QGridLayout,
    QProgressBar, QTextEdit, QSplitter, QSizePolicy, QPushButton,
    QCheckBox, QTabWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ui_styles import BG, PANEL, CARD, BORDER, MUTED, ACCENT, LOG_TEXT, TEXT, QSS
from simulation_worker import SimWorker


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Artifact Viewer")
        self.resize(1500, 900)
        self.setStyleSheet(QSS)
        self.worker = None
        self.last_res = {"Artifacts": None, "NMAR": None} 

        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        # ── Global Sidebar ──
        sidebar = QWidget(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(320)
        self.s_lay = QVBoxLayout(sidebar); self.s_lay.setContentsMargins(20, 25, 20, 25); self.s_lay.setSpacing(10)

        logo = QLabel("CT·SIM"); logo.setObjectName("logo"); logo.setAlignment(Qt.AlignCenter); self.s_lay.addWidget(logo)
        tag = QLabel("Artifact Simulation"); tag.setObjectName("tagline"); tag.setAlignment(Qt.AlignCenter); self.s_lay.addWidget(tag)
        self.s_lay.addWidget(self._div())

        self.w_params = {
            "Phantom": QComboBox(),
            "Artifact": QComboBox(),
            "_Section_Combined": QLabel("Combined Artifacts Parameters"), 
            "FOV (mm)": self._dspin(100, 700, 300, 10),
            "mA": self._spin(10, 1200, 800),
            "Views": self._spin(100, 3000, 1000),
            "keV": self._spin(40, 140, 70),
            "View Factor": self._spin(2, 16, 4),
            "Shift (mm)": self._dspin(0.1, 20, 1.4, 0.1),
            "Break View": self._spin(1, 999, 700),
            "Scatter Scale": self._dspin(10, 5000, 1000, 100),
            "_Section_NMAR": QLabel("NMAR Parameters"), 
            "HU Threshold": self._spin(500, 5000, 2500),
            "N Angles": self._spin(90, 720, 360),
            "Show Steps": QCheckBox("Show Detailed Steps")
        }
        
        self.w_params["Phantom"].addItems(["Phantom 1 — Water & Iron", "Phantom 2 — Plexi & Silver"])
        self.w_params["Artifact"].addItems(["Aliasing", "Beam Hardening", "Motion", "Noise", "Scatter", "Combined"])
        self.w_params["Artifact"].currentTextChanged.connect(self._refresh_visibility)
        
        self.w_params["Show Steps"].stateChanged.connect(self._replot_instant)

        self.param_rows = {}
        self.grid = QGridLayout(); self.grid.setSpacing(8)
        
        row_idx = 0
        for name, wid in self.w_params.items():
            if isinstance(wid, QCheckBox) or name.startswith("_Section_"):
                if name.startswith("_Section_"):
                    wid.setObjectName("section")
                    wid.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
                self.grid.addWidget(wid, row_idx, 0, 1, 2)
                self.param_rows[name] = {'lbl': wid, 'wid': wid} 
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

            log_w = QWidget(); log_w.setStyleSheet(f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px;")
            log_l = QVBoxLayout(log_w); log_l.setContentsMargins(15, 10, 15, 15)
            log_l.addWidget(QLabel(f"{tab_name.upper()} LOG", objectName="tagline"))
            
            log_box = QTextEdit(); log_box.setReadOnly(True); log_box.setMaximumHeight(120)
            log_box.setStyleSheet("border: none; background: transparent;") 
            log_l.addWidget(log_box)
            self.figs[tab_name]["log"] = log_box
            split.addWidget(log_w)
            
            split.setSizes([600, 120])
            l.addWidget(split, stretch=1)
            
            btn = QPushButton(f"RUN {tab_name.upper()}")
            btn.clicked.connect(lambda _, m=tab_name: self._run(m))
            prog = QProgressBar(); prog.setRange(0, 0); prog.setVisible(False)
            l.addWidget(btn); l.addWidget(prog)
            
            self.figs[tab_name]["btn"] = btn
            self.figs[tab_name]["prog"] = prog
            self.tabs.addTab(w, tab_name)

        layout.addWidget(right_w, stretch=1)
        self._refresh_visibility() 

    def _replot_instant(self):
        """Instantly redraws the plot when a checkbox is toggled without running the thread."""
        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        if self.last_res.get(tab_name):
            self.last_res[tab_name]["show_steps"] = self.w_params["Show Steps"].isChecked()
            self._on_done(self.last_res[tab_name], tab_name, instant_update=True)

    def _refresh_visibility(self):
        """Dynamically shows/hides sidebar parameters based on active tab & mode."""
        tab = self.tabs.tabText(self.tabs.currentIndex())
        art = self.w_params["Artifact"].currentText()
        
        visible = {"Phantom"} 

        if tab == "Artifacts":
            visible.add("Artifact")
            if art == "Aliasing": visible.update(["FOV (mm)", "Views", "View Factor"])
            elif art == "Beam Hardening": visible.update(["FOV (mm)", "Views", "keV"])
            elif art == "Motion": visible.update(["FOV (mm)", "Views", "Shift (mm)", "Break View"])
            elif art == "Noise": visible.update(["FOV (mm)", "Views", "mA"])
            elif art == "Scatter": visible.update(["FOV (mm)", "Views", "Scatter Scale"])
            elif art == "Combined": visible.update(["FOV (mm)", "mA", "Views", "keV", "View Factor", "Shift (mm)", "Break View", "Scatter Scale", "Show Steps"])
        else: # NMAR Tab
            visible.update([
                "_Section_Combined", "FOV (mm)", "mA", "Views", "keV", "View Factor", "Shift (mm)", "Break View", "Scatter Scale",
                "_Section_NMAR", "HU Threshold", "N Angles", "Show Steps"
            ])

        for name, row in self.param_rows.items():
            is_vis = name in visible
            if name in ["Show Steps"] or name.startswith("_Section_"):
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
            "show_steps": w["Show Steps"].isChecked()
        }

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

    def _on_done(self, res, tab_name, instant_update=False):
        self.last_res[tab_name] = res  
        controls = self.figs[tab_name]
        controls["btn"].setEnabled(True)
        controls["prog"].setVisible(False)
        
        if not instant_update:
            controls["log"].append("Execution Complete.")
        
        fig, canvas = controls["fig"], controls["canvas"]
        fig.clear()

        def show(ax, img, title, vmin=None, vmax=None, cmap='gray'):
            ax.set_facecolor(BG); ax.axis("off")
            ax.set_title(title, color=ACCENT, fontsize=10, pad=8, weight='bold')
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