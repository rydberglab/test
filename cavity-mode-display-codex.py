"""
Interactive Fabry-Perot cavity visualizer using PyQt + Matplotlib.

Features
--------
- Panel 1: side-view Gaussian beam envelope inside the cavity.
- Panel 4: g1-g2 cavity stability diagram.
- R1, R2, and L controls use a 1000 mm span around a user-set center.
- Center for each of R1/R2/L is editable next to its slider.
- Additional sliders for wavelength, n_center, and fixed y-axis range.
"""

from __future__ import annotations

import sys
import numpy as np

# Qt binding: prefer PyQt6, fallback to PyQt5.
try:
    from PyQt6.QtCore import Qt, pyqtSignal as Signal, QSignalBlocker
    from PyQt6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    HORIZONTAL = Qt.Orientation.Horizontal
except ImportError:
    try:
        from PyQt5.QtCore import Qt, pyqtSignal as Signal, QSignalBlocker
        from PyQt5.QtWidgets import (
            QApplication,
            QDoubleSpinBox,
            QFrame,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QSlider,
            QVBoxLayout,
            QWidget,
        )

        HORIZONTAL = Qt.Horizontal
    except ImportError as exc:
        raise SystemExit(
            "PyQt is required. Install PyQt6 or PyQt5, then rerun."
        ) from exc

from matplotlib.figure import Figure
from matplotlib.patches import Polygon

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


# ---------------- Cavity mode solver ----------------
def compute_cavity_mode(
    R1: float,
    R2: float,
    L: float,
    wavelength: float,
    n_center: float,
) -> dict:
    """Compute TEM00 mode for a two-mirror cavity using ABCD matrices."""
    if R1 <= 0 or R2 <= 0:
        raise ValueError("Mirror ROC must be positive.")
    if L <= 0:
        raise ValueError("Cavity length must be positive.")
    if n_center <= 0:
        raise ValueError("Center refractive index must be positive.")

    lambda_medium = wavelength / n_center

    def prop(d: float) -> np.ndarray:
        return np.array([[1.0, d], [0.0, 1.0]])

    def mirror(R: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [-2.0 / R, 1.0]])

    M = mirror(R1) @ prop(L) @ mirror(R2) @ prop(L)
    A, C, D = M[0, 0], M[1, 0], M[1, 1]

    g1, g2 = 1.0 - L / R1, 1.0 - L / R2

    if abs((A + D) / 2.0) > 1.0 + 1e-9:
        raise ValueError(
            f"Unstable cavity: g1={g1:.4f}, g2={g2:.4f}, g1*g2={g1 * g2:.4f}."
        )
    if abs(C) < 1e-14:
        raise ValueError("Near-planar cavity (C ~ 0): Gaussian mode is not confined.")

    disc = max(0.0, 4.0 - (A + D) ** 2)
    q_m1 = complex((A - D) / (2.0 * C), np.sqrt(disc) / (2.0 * abs(C)))
    if q_m1.imag < 0:
        q_m1 = q_m1.conjugate()

    z = np.linspace(0.0, L, 2000)
    inv_q = 1.0 / (q_m1 + z)
    w_z = np.sqrt(-lambda_medium / (np.pi * inv_q.imag))

    idx_w0 = int(np.argmin(w_z))

    return {
        "z": z,
        "w_z": w_z,
        "w0": w_z[idx_w0],
        "z_waist": z[idx_w0],
        "zR": np.pi * w_z[idx_w0] ** 2 / lambda_medium,
        "w_M1": w_z[0],
        "w_M2": w_z[-1],
        "g1": g1,
        "g2": g2,
    }


class LabeledSlider(QWidget):
    """A row with label, slider, and numeric spinbox."""

    value_changed = Signal(float)

    def __init__(
        self,
        name: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 2,
        unit: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scale = int(round(1.0 / step))
        self.unit = unit

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(130)

        self.slider = QSlider(HORIZONTAL)
        self.slider.setRange(int(round(minimum * self.scale)), int(round(maximum * self.scale)))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setMinimumWidth(95)
        if unit:
            self.spin.setSuffix(f" {unit}")

        row.addWidget(self.name_label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.set_value(value, emit=False)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float, emit: bool = True) -> None:
        val_int = int(round(value * self.scale))
        val_int = int(np.clip(val_int, self.slider.minimum(), self.slider.maximum()))
        val = val_int / self.scale

        with QSignalBlocker(self.slider), QSignalBlocker(self.spin):
            self.slider.setValue(val_int)
            self.spin.setValue(val)

        if emit:
            self.value_changed.emit(val)

    def _on_slider(self, value_int: int) -> None:
        value = value_int / self.scale
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.value_changed.emit(value)

    def _on_spin(self, value: float) -> None:
        value_int = int(round(value * self.scale))
        with QSignalBlocker(self.slider):
            self.slider.setValue(value_int)
        self.value_changed.emit(float(value))


class CenteredRangeSlider(QWidget):
    """Slider with editable center and fixed span in mm."""

    value_changed = Signal(float)

    def __init__(
        self,
        name: str,
        value_mm: float,
        center_mm: float,
        span_mm: float = 1000.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.span_mm = span_mm
        self.half_span = 0.5 * span_mm
        self._min_mm = 1.0
        self._max_mm = 1001.0

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(130)

        self.slider = QSlider(HORIZONTAL)

        self.value_label = QLabel()
        self.value_label.setMinimumWidth(84)

        center_label = QLabel("Center")

        self.center_spin = QDoubleSpinBox()
        self.center_spin.setRange(1.0, 1_000_000.0)
        self.center_spin.setDecimals(1)
        self.center_spin.setSingleStep(10.0)
        self.center_spin.setMinimumWidth(90)
        self.center_spin.setSuffix(" mm")

        self.range_label = QLabel()
        self.range_label.setMinimumWidth(140)

        row.addWidget(self.name_label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        row.addWidget(center_label)
        row.addWidget(self.center_spin)
        row.addWidget(self.range_label)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.center_spin.valueChanged.connect(self._on_center_changed)

        self._set_center(center_mm)
        self.set_value_mm(value_mm, emit=False)

    def value_mm(self) -> float:
        return float(self.slider.value())

    def _bounds_from_center(self, center_mm: float) -> tuple[float, float]:
        min_mm = center_mm - self.half_span
        max_mm = center_mm + self.half_span
        if min_mm < 1.0:
            max_mm += 1.0 - min_mm
            min_mm = 1.0
        if max_mm <= min_mm:
            max_mm = min_mm + 1.0
        return min_mm, max_mm

    def _set_center(self, center_mm: float) -> None:
        min_mm, max_mm = self._bounds_from_center(center_mm)
        self._min_mm = min_mm
        self._max_mm = max_mm

        with QSignalBlocker(self.slider):
            self.slider.setRange(int(round(min_mm)), int(round(max_mm)))

        with QSignalBlocker(self.center_spin):
            self.center_spin.setValue(center_mm)

        self.range_label.setText(f"[{min_mm:.0f}, {max_mm:.0f}] mm")

    def set_value_mm(self, value_mm: float, emit: bool = True) -> None:
        val = int(round(np.clip(value_mm, self._min_mm, self._max_mm)))
        with QSignalBlocker(self.slider):
            self.slider.setValue(val)
        self._sync_value_label()
        if emit:
            self.value_changed.emit(float(val))

    def _sync_value_label(self) -> None:
        self.value_label.setText(f"{self.slider.value():.0f} mm")

    def _on_slider_changed(self, value_mm: int) -> None:
        self._sync_value_label()
        self.value_changed.emit(float(value_mm))

    def _on_center_changed(self, center_mm: float) -> None:
        prev_val = self.value_mm()
        self._set_center(float(center_mm))
        self.set_value_mm(prev_val, emit=True)


class CavityMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fabry-Perot Cavity Viewer (PyQt)")
        self.resize(1450, 900)

        self._build_ui()
        self._apply_style()
        self.update_plot()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        self.figure = Figure(figsize=(12, 7), facecolor="#f8f9fb")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax1 = self.figure.add_subplot(1, 2, 1)
        self.ax4 = self.figure.add_subplot(1, 2, 2)

        root_layout.addWidget(self.toolbar)
        root_layout.addWidget(self.canvas, 1)

        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        geometry_box = QGroupBox("Geometry (mm)")
        geo_layout = QVBoxLayout(geometry_box)
        geo_layout.setSpacing(6)

        self.ctrl_r1 = CenteredRangeSlider("R1", value_mm=500.0, center_mm=500.0, span_mm=1000.0)
        self.ctrl_r2 = CenteredRangeSlider("R2", value_mm=500.0, center_mm=500.0, span_mm=1000.0)
        self.ctrl_l = CenteredRangeSlider("Length L", value_mm=400.0, center_mm=400.0, span_mm=1000.0)

        geo_layout.addWidget(self.ctrl_r1)
        geo_layout.addWidget(self.ctrl_r2)
        geo_layout.addWidget(self.ctrl_l)

        optics_box = QGroupBox("Optics and Display")
        opt_layout = QVBoxLayout(optics_box)
        opt_layout.setSpacing(6)

        self.ctrl_wavelength = LabeledSlider(
            "Wavelength",
            minimum=400.0,
            maximum=2000.0,
            value=1064.0,
            step=1.0,
            decimals=0,
            unit="nm",
        )
        self.ctrl_n_center = LabeledSlider(
            "n_center",
            minimum=1.00,
            maximum=3.00,
            value=1.00,
            step=0.01,
            decimals=2,
            unit="",
        )
        self.ctrl_y_max = LabeledSlider(
            "Y max",
            minimum=5,
            maximum=50.00,
            value=5.00,
            step=0.01,
            decimals=2,
            unit="mm",
        )

        opt_layout.addWidget(self.ctrl_wavelength)
        opt_layout.addWidget(self.ctrl_n_center)
        opt_layout.addWidget(self.ctrl_y_max)

        controls_layout.addWidget(geometry_box)
        controls_layout.addWidget(optics_box)
        root_layout.addWidget(controls_frame)

        self.ctrl_r1.value_changed.connect(self.update_plot)
        self.ctrl_r2.value_changed.connect(self.update_plot)
        self.ctrl_l.value_changed.connect(self.update_plot)
        self.ctrl_wavelength.value_changed.connect(self.update_plot)
        self.ctrl_n_center.value_changed.connect(self.update_plot)
        self.ctrl_y_max.value_changed.connect(self.update_plot)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #cfd7e6;
                border-radius: 8px;
                margin-top: 10px;
                background: #fdfdfd;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ccd4e0;
                height: 8px;
                background: #eef2f8;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2f6fde;
                border: 1px solid #275cbc;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            """
        )

    def update_plot(self, _value: float | None = None) -> None:
        R1 = self.ctrl_r1.value_mm() * 1e-3
        R2 = self.ctrl_r2.value_mm() * 1e-3
        L = self.ctrl_l.value_mm() * 1e-3
        wavelength = self.ctrl_wavelength.value() * 1e-9
        n_center = self.ctrl_n_center.value()
        y_max_mm = self.ctrl_y_max.value()

        g1 = 1.0 - L / R1
        g2 = 1.0 - L / R2

        mode = None
        error_text = None
        try:
            mode = compute_cavity_mode(R1=R1, R2=R2, L=L, wavelength=wavelength, n_center=n_center)
        except ValueError as exc:
            error_text = str(exc)

        self._draw_panel_1(mode, error_text, R1, R2, L, wavelength, n_center, y_max_mm)
        self._draw_panel_4(g1, g2)

        self.figure.suptitle("Fabry-Perot Cavity Interactive View (Panels 1 and 4)", fontsize=13, weight="bold")
        self.figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        self.canvas.draw_idle()

    def _draw_panel_1(
        self,
        mode: dict | None,
        error_text: str | None,
        R1: float,
        R2: float,
        L: float,
        wavelength: float,
        n_center: float,
        y_max_mm: float,
    ) -> None:
        ax = self.ax1
        ax.clear()

        mm = 1e3
        L_mm = L * mm
        pad_mm = max(8.0, 0.08 * L_mm)
        y_lim = max(0.05, y_max_mm*1.1)

        if mode is not None:
            z_mm = mode["z"] * mm
            w_mm = mode["w_z"] * mm
        else:
            z_mm = np.array([0.0, L_mm])
            w_mm = np.array([0.0, 0.0])

        ax.axvspan(-pad_mm, 0.0, color="#eaf4ff", alpha=0.35)
        ax.axvspan(0.0, L_mm, color="#fff6dc", alpha=0.30)
        ax.axvspan(L_mm, L_mm + pad_mm, color="#eaf4ff", alpha=0.35)

        if mode is not None:
            ax.fill_between(z_mm, w_mm, -w_mm, alpha=0.24, color="steelblue")
            ax.plot(z_mm, w_mm, color="steelblue", lw=2)
            ax.plot(z_mm, -w_mm, color="steelblue", lw=2)

            z_w_mm = mode["z_waist"] * mm
            ax.axvline(z_w_mm, color="crimson", ls="--", lw=1.3)

        mirror_h = max(0.05, y_max_mm)
        mirror_w = max(0.6, 0.02 * L_mm)

        self._add_curved_mirror(
            ax=ax,
            z_vertex_mm=0.0,
            roc_mm=R1 * mm,
            half_height_mm=mirror_h,
            thickness_mm=mirror_w,
            side="left",
        )
        self._add_curved_mirror(
            ax=ax,
            z_vertex_mm=L_mm,
            roc_mm=R2 * mm,
            half_height_mm=mirror_h,
            thickness_mm=mirror_w,
            side="right",
        )

        ax.set_xlim(-pad_mm, L_mm + pad_mm)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True, alpha=0.28)

    def _add_curved_mirror(
        self,
        ax,
        z_vertex_mm: float,
        roc_mm: float,
        half_height_mm: float,
        thickness_mm: float,
        side: str,
    ) -> None:
        roc_mm = max(float(roc_mm), 1e-6)
        half_height_mm = max(float(half_height_mm), 1e-6)
        thickness_mm = max(float(thickness_mm), 1e-3)

        # Circular sag profile for a spherical mirror surface.
        y_abs_max = min(half_height_mm, 0.999 * roc_mm)
        if y_abs_max <= 1e-6:
            return

        y = np.linspace(-y_abs_max, y_abs_max, 300)
        sag = roc_mm - np.sqrt(np.maximum(roc_mm * roc_mm - y * y, 0.0))

        if side == "left":
            x_front = z_vertex_mm + sag
            x_back = x_front - thickness_mm
        else:
            x_front = z_vertex_mm - sag
            x_back = x_front + thickness_mm

        verts = np.column_stack(
            [np.concatenate([x_front, x_back[::-1]]), np.concatenate([y, y[::-1]])]
        )
        ax.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor="#4f5d6b",
                edgecolor="#3c4854",
                linewidth=0.8,
                alpha=0.90,
            )
        )

    def _draw_panel_4(self, g1: float, g2: float) -> None:
        ax = self.ax4
        ax.clear()

        g_lin = np.linspace(-1.6, 1.6, 400)
        G1, G2 = np.meshgrid(g_lin, g_lin)
        g_prod = G1 * G2

        ax.contourf(
            G1,
            G2,
            np.clip(g_prod, -0.05, 1.05),
            levels=[-0.04, 0.0, 1.0, 1.04],
            colors=["#ffd6d6", "#d5efd5", "#ffd6d6"],
            alpha=0.70,
        )
        ax.contour(G1, G2, g_prod, levels=[0.0, 1.0], colors=["darkgreen"], linewidths=2)

        ax.plot(g1, g2, "r*", ms=14, label=f"Current ({g1:.3f}, {g2:.3f})")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axvline(0.0, color="k", lw=0.6)

        stable = 0.0 <= g1 * g2 <= 1.0
        ax.text(
            0.5,
            0.08,
            "STABLE" if stable else "UNSTABLE",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="darkgreen" if stable else "firebrick",
            weight="bold",
        )

        ax.set_title("Panel 4: Cavity Stability Diagram")
        ax.set_xlabel("g1 = 1 - L/R1")
        ax.set_ylabel("g2 = 1 - L/R2")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = CavityMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
