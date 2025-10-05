# Parametric Pattern Application

A Python GUI application for parametric customization of clothing patterns. Modify pattern dimensions through interactive sliders with real-time visualization.

## Features

- **Parametric Control**: Adjust length, width, armpit level, shoulder level, and edge distortion
- **Real-time Visualization**: See changes instantly as you adjust parameters
- **SVG Export**: Export modified patterns to A0-sized SVG for printing
- **Three Pattern Pieces**: Back, Front, and Sleeve patterns

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your pattern images in the project root:
   - `BasePattern_Back.png`
   - `BasePattern_Front.png`
   - `BasePattern_Sleeve.png`

   (Images should be 768x768 pixels, black outlines on white background)

## Usage

### First-time Setup

Convert pattern images to curve data:
```bash
python main.py --convert
```

This creates curve representations in `BasePatternCurves/` directory.

### Run the Application

```bash
python main.py
```

### GUI Controls

**Sliders:**
- **Length of Garment**: Scale pattern vertically (0.5x - 1.5x)
- **Width of Garment**: Scale pattern horizontally (0.5x - 1.5x)
- **Armpit Level**: Adjust armhole position (-50 to +50 pixels)
- **Shoulder Level**: Adjust shoulder seam position (-50 to +50 pixels)
- **Edge Distortion**: Add aesthetic variation to edges (0-10 amplitude)

**Buttons:**
- **Reset to Default**: Restore original pattern dimensions
- **Export SVG**: Save current pattern to `pattern_export.svg` (A0 size)

## File Structure

```
ParametricPatterns/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── BasePattern_*.png            # Original pattern images (user-provided)
├── BasePatternCurves/           # Generated curve data
│   ├── back_curves.json
│   ├── front_curves.json
│   └── sleeve_curves.json
├── src/
│   ├── image_to_curves.py      # Image-to-curve conversion
│   ├── pattern_engine.py       # Pattern transformation engine
│   ├── curve_operations.py     # Curve manipulation utilities
│   └── gui.py                  # PyQt5 GUI
└── utils/
    └── geometry.py             # Geometric helper functions
```

## Technical Details

- **Image Processing**: OpenCV for edge detection and contour extraction
- **Curve Fitting**: SciPy B-splines for smooth parametric curves
- **GUI**: PyQt5 for responsive interface
- **Export**: SVGWrite for A0-sized pattern exports
- **Units**: Works in centimeters (10 pixels = 1 cm)

## Requirements

- Python 3.7+
- opencv-python
- numpy
- scipy
- PyQt5
- svgwrite
- Pillow
- scikit-image

See `requirements.txt` for version details.

## Troubleshooting

**"No curve data found"**: Run `python main.py --convert` first to generate curve data from images.

**"Image not found"**: Ensure pattern PNG files are in the project root directory.

**Display issues**: Patterns are auto-scaled to fit the canvas. Check that pattern images have clear black outlines on white background.
