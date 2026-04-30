BG, PANEL, CARD = "#EBE3D5", "#DFD6C8", "#EBE3D5"
BORDER, MUTED, ACCENT, LOG_TEXT = "#D6CFC6", "#7A7285", "#4E6766", "#1E152A"
TEXT = "#1E152A"

QSS = f"""
* {{ font-family: 'Segoe UI', system-ui, sans-serif; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
#logo {{ color: {ACCENT}; font-size: 24px; font-weight: 800; letter-spacing: 2px; }}
#tagline {{ color: {MUTED}; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; font-weight: bold; }}
#section {{ color: {ACCENT}; font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-top: 15px; margin-bottom: 5px; }}
#param {{ color: {TEXT}; font-size: 12px; font-weight: 500; }}
#div {{ background: {BORDER}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 4px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 8px 18px; border-radius: 4px; margin-right: 2px; font-size: 12px; font-weight: 600; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: {BG}; border-color: {ACCENT}; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 10px; font-size: 13px; min-height: 24px; color: {TEXT}; }}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{ border: 1px solid {ACCENT}; }}
QCheckBox {{ color: {TEXT}; font-size: 12px; spacing: 8px; font-weight: 500; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER}; border-radius: 3px; background: {CARD}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QPushButton {{ background: {ACCENT}; color: {BG}; border: none; border-radius: 4px; padding: 12px; font-size: 13px; font-weight: bold; letter-spacing: 1px; }}
QPushButton:hover {{ background: #3b504f; }}
QPushButton:disabled {{ background: {BORDER}; color: {MUTED}; }}
QTextEdit {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; color: {LOG_TEXT}; font-family: 'Consolas', monospace; font-size: 11px; padding: 8px; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 4px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
"""