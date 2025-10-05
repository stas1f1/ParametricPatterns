"""PyQt5 GUI for parametric pattern application."""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox,
                             QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
import numpy as np
import svgwrite

from src.pattern_engine import PatternEngine


class PatternCanvas(QWidget):
    """Canvas widget for displaying patterns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.patterns = {}  # Dictionary of pattern_name: points
        self.setMinimumSize(800, 600)
        self.zoom = 1.0
        self.offset_x = 50
        self.offset_y = 50

    def set_patterns(self, patterns):
        """Update patterns to display."""
        self.patterns = patterns
        self.update()  # Trigger repaint

    def paintEvent(self, event):
        """Paint the patterns."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(250, 250, 250))

        if not self.patterns:
            # Show placeholder text
            painter.setPen(QColor(150, 150, 150))
            font = QFont('Arial', 12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter,
                           "Load pattern images to begin")
            return

        # Calculate layout positions for three patterns
        canvas_width = self.width()
        canvas_height = self.height()

        # Layout: Back and Front side by side at top, Sleeve centered below
        positions = {
            'back': (canvas_width * 0.25, canvas_height * 0.3),
            'front': (canvas_width * 0.75, canvas_height * 0.3),
            'sleeve': (canvas_width * 0.5, canvas_height * 0.75),
        }

        # Colors for different patterns
        colors = {
            'back': QColor(50, 100, 200),
            'front': QColor(200, 50, 100),
            'sleeve': QColor(50, 200, 100),
        }

        # Draw each pattern
        for pattern_name, points in self.patterns.items():
            if pattern_name not in positions:
                continue

            color = colors.get(pattern_name, QColor(0, 0, 0))
            center_x, center_y = positions[pattern_name]

            self._draw_pattern(painter, points, center_x, center_y, color, pattern_name)

    def _draw_pattern(self, painter, points, center_x, center_y, color, label):
        """
        Draw a single pattern.

        Args:
            painter: QPainter instance
            points: Array of (x, y) points
            center_x, center_y: Center position on canvas
            color: QColor for the pattern
            label: Pattern name label
        """
        if len(points) == 0:
            return

        points = np.array(points)

        # Calculate bounds
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        width = max_x - min_x
        height = max_y - min_y

        # Calculate scale to fit nicely (target ~150-200 pixels)
        target_size = 150
        scale = min(target_size / width, target_size / height) if width > 0 and height > 0 else 1.0

        # Center the pattern
        pattern_center_x = (min_x + max_x) / 2
        pattern_center_y = (min_y + max_y) / 2

        # Draw pattern outline
        pen = QPen(color, 2)
        painter.setPen(pen)

        for i in range(len(points)):
            x1 = (points[i][0] - pattern_center_x) * scale + center_x
            y1 = (points[i][1] - pattern_center_y) * scale + center_y

            next_i = (i + 1) % len(points)
            x2 = (points[next_i][0] - pattern_center_x) * scale + center_x
            y2 = (points[next_i][1] - pattern_center_y) * scale + center_y

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw label
        painter.setPen(QColor(0, 0, 0))
        font = QFont('Arial', 10, QFont.Bold)
        painter.setFont(font)
        label_y = center_y + (height * scale / 2) + 20
        painter.drawText(int(center_x - 30), int(label_y), label.upper())


class ParameterSlider(QWidget):
    """Widget for a parameter slider with label and value display."""

    def __init__(self, name, label, min_val, max_val, default_val, decimal_places=2, parent=None):
        super().__init__(parent)
        self.name = name
        self.decimal_places = decimal_places
        self.min_val = min_val
        self.max_val = max_val

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)

        # Label and value display
        header_layout = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setFont(QFont('Arial', 10))
        self.value_label = QLabel(self._format_value(default_val))
        self.value_label.setFont(QFont('Arial', 9))
        self.value_label.setStyleSheet("color: #666;")
        header_layout.addWidget(self.label)
        header_layout.addStretch()
        header_layout.addWidget(self.value_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)  # High resolution
        self.slider.setValue(self._value_to_slider(default_val))
        self.slider.valueChanged.connect(self._on_slider_change)

        layout.addLayout(header_layout)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def _value_to_slider(self, value):
        """Convert real value to slider position."""
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return int(normalized * 1000)

    def _slider_to_value(self, slider_pos):
        """Convert slider position to real value."""
        normalized = slider_pos / 1000
        return self.min_val + normalized * (self.max_val - self.min_val)

    def _format_value(self, value):
        """Format value for display."""
        return f"{value:.{self.decimal_places}f}"

    def _on_slider_change(self, position):
        """Handle slider value change."""
        value = self._slider_to_value(position)
        self.value_label.setText(self._format_value(value))

    def get_value(self):
        """Get current value."""
        return self._slider_to_value(self.slider.value())

    def set_value(self, value):
        """Set slider value."""
        self.slider.setValue(self._value_to_slider(value))


class PatternGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.engine = None
        self.sliders = {}
        self.init_ui()

        # Try to load patterns
        self.load_engine()

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle('Parametric Pattern Editor')
        self.setGeometry(100, 100, 1200, 700)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left panel - Controls
        left_panel = self._create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel - Canvas
        self.canvas = PatternCanvas()
        main_layout.addWidget(self.canvas, stretch=3)

    def _create_control_panel(self):
        """Create the control panel with sliders."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel('Pattern Parameters')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)

        # Sliders group
        sliders_group = QGroupBox()
        sliders_layout = QVBoxLayout()
        sliders_group.setLayout(sliders_layout)

        # Parameter definitions
        params = [
            ('length', 'Length of Garment', 0.5, 1.5, 1.0, 2),
            ('width', 'Width of Garment', 0.5, 1.5, 1.0, 2),
            ('armpit_level', 'Armpit Level', -50, 50, 0, 0),
            ('shoulder_level', 'Shoulder Level', -50, 50, 0, 0),
            ('distortion', 'Edge Distortion', 0, 10, 0, 1),
        ]

        # Create sliders
        for name, label, min_val, max_val, default, decimals in params:
            slider = ParameterSlider(name, label, min_val, max_val, default, decimals)
            slider.slider.valueChanged.connect(self.on_parameter_change)
            self.sliders[name] = slider
            sliders_layout.addWidget(slider)

        layout.addWidget(sliders_group)
        layout.addStretch()

        # Buttons
        button_layout = QVBoxLayout()

        reset_btn = QPushButton('Reset to Default')
        reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(reset_btn)

        export_svg_btn = QPushButton('Export SVG')
        export_svg_btn.clicked.connect(self.export_svg)
        button_layout.addWidget(export_svg_btn)

        layout.addLayout(button_layout)

        return panel

    def load_engine(self):
        """Load the pattern engine."""
        try:
            self.engine = PatternEngine()
            self.update_display()
        except Exception as e:
            print(f"Error loading patterns: {e}")
            print("Please run image conversion first or check that pattern files exist.")

    def get_current_params(self):
        """Get current parameter values from sliders."""
        return {name: slider.get_value() for name, slider in self.sliders.items()}

    def on_parameter_change(self):
        """Handle parameter change from any slider."""
        self.update_display()

    def update_display(self):
        """Update the pattern display with current parameters."""
        if self.engine is None:
            return

        params = self.get_current_params()

        try:
            transformed_patterns = self.engine.get_all_transformed_patterns(params)
            self.canvas.set_patterns(transformed_patterns)
        except Exception as e:
            print(f"Error updating display: {e}")

    def reset_parameters(self):
        """Reset all parameters to default."""
        if self.engine is None:
            return

        defaults = self.engine.get_default_params()
        for name, value in defaults.items():
            if name in self.sliders:
                self.sliders[name].set_value(value)

        self.update_display()

    def export_svg(self):
        """Export current patterns to SVG file."""
        if self.engine is None:
            return

        params = self.get_current_params()
        transformed_patterns = self.engine.get_all_transformed_patterns(params)

        # Create SVG (A0 size: 841 x 1189 mm)
        dwg = svgwrite.Drawing('pattern_export.svg', size=('841mm', '1189mm'),
                              viewBox=f'0 0 8410 11890')  # 10 pixels per mm

        # Layout positions (in 0.1mm units)
        positions = {
            'back': (2100, 3000),
            'front': (6300, 3000),
            'sleeve': (4200, 8000),
        }

        colors = {
            'back': 'blue',
            'front': 'red',
            'sleeve': 'green',
        }

        for pattern_name, points in transformed_patterns.items():
            if pattern_name not in positions:
                continue

            points = np.array(points)
            center_x, center_y = positions[pattern_name]

            # Scale from pixels to 0.1mm (assuming 10 pixels per cm = 1 pixel per mm)
            scale = 10  # 1 pixel = 1mm, 1mm = 10 units in SVG

            # Convert points to SVG coordinates
            svg_points = [(p[0] * scale + center_x, p[1] * scale + center_y) for p in points]

            # Draw path
            path_data = f'M {svg_points[0][0]},{svg_points[0][1]}'
            for p in svg_points[1:]:
                path_data += f' L {p[0]},{p[1]}'
            path_data += ' Z'

            dwg.add(dwg.path(d=path_data, fill='none', stroke=colors[pattern_name],
                            stroke_width=5))

            # Add label
            dwg.add(dwg.text(pattern_name.upper(), insert=(center_x, center_y - 500),
                           font_size=200, fill='black'))

        dwg.save()
        print("Exported to pattern_export.svg")


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    window = PatternGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()
