# Parametric Clothing Pattern Application

## Project Overview
A Python GUI application for parametric customization of clothing patterns. The application allows users to modify pattern dimensions through interactive sliders, with real-time visualization of the modified pattern pieces.

## Pattern Components
The application works with three pattern pieces:
- **Back**: BasePattern_Back.png
- **Front**: BasePattern_Front.png
- **Sleeve**: BasePattern_Sleeve.png

## Adjustable Parameters
1. **Length of garment**: Vertical scaling/adjustment of the pattern body
2. **Width of garment**: Horizontal scaling/adjustment of the pattern body
3. **Level of armpit**: Vertical positioning of the armhole (allows lowering)
4. **Level of shoulder**: Vertical positioning of the shoulder seam (allows lowering)
5. **Distortion of edges**: Creates uneven/organic seam lines (random or controlled variation)

## File Structure
```
ParametricPatterns/
├── task.md                          # This file
├── main.py                          # Main application entry point
├── BasePattern_Back.png             # Original back pattern image
├── BasePattern_Front.png            # Original front pattern image
├── BasePattern_Sleeve.png           # Original sleeve pattern image
├── BasePatternCurves/               # Converted curve data storage
│   ├── back_curves.json             # Back pattern as parametric curves
│   ├── front_curves.json            # Front pattern as parametric curves
│   └── sleeve_curves.json           # Sleeve pattern as parametric curves
├── src/
│   ├── __init__.py
│   ├── image_to_curves.py           # Image to curve conversion module
│   ├── pattern_engine.py            # Pattern transformation engine
│   ├── curve_operations.py          # Curve manipulation utilities
│   └── gui.py                       # GUI implementation
├── utils/
│   ├── __init__.py
│   └── geometry.py                  # Geometric helper functions
└── requirements.txt                 # Python dependencies
```

## Algorithm Workflow

### Phase 1: Initial Conversion (One-time Setup)
1. **Edge Detection**: Use computer vision to detect pattern edges from PNG images
   - Convert images to grayscale
   - Apply edge detection (Canny, contour detection)
   - Identify closed contours representing pattern pieces

2. **Curve Fitting**: Convert detected edges to parametric curves
   - Extract edge points
   - Fit B-spline or Bézier curves to edge points
   - Identify key control points (corners, seams, curves)
   - Label critical points (shoulder, armpit, hem, side seam, etc.)

3. **Data Storage**: Save curve representations as JSON
   - Store control points
   - Store curve types and parameters
   - Store semantic labels for key points
   - Store original dimensions for reference

### Phase 2: Pattern Modification (Real-time)
1. **Parameter Input**: Receive slider values from GUI
2. **Transformation Logic**: Apply geometric transformations based on parameters
   - **Length adjustment**: Scale vertically, preserve shoulder/neck region
   - **Width adjustment**: Scale horizontally, maintain proportions
   - **Armpit level**: Move armpit control points vertically
   - **Shoulder level**: Move shoulder seam control points vertically
   - **Edge distortion**: Add controlled noise/variation to curve control points

3. **Curve Regeneration**: Rebuild curves with modified control points
4. **Rendering**: Display modified patterns in real-time

## GUI Requirements

### Layout
```
+----------------------------------+------------------------------------------+
|          CONTROLS                |              PATTERN DISPLAY             |
|                                  |                                          |
| Length:        [====|====]       |     +-------+      +-------+             |
| Width:         [===|=====]       |     | BACK  |      | FRONT|              |
| Armpit Level:  [======|==]       |     |       |      |       |             |
| Shoulder:      [=======|=]       |     |       |      |       |             |
| Distortion:    [|========]       |     +-------+      +-------+             |
|                                  |                                          |
| [Reset] [Export SVG] [Export PDF]|            +-------+                     |
|                                  |            | SLEEVE|                     |
|                                  |            |       |                     |
|                                  |            +-------+                     |
+----------------------------------+------------------------------------------+
```

### Components
1. **Left Panel (Control Panel)**:
   - 5 sliders with labels and value displays
   - Reset button to restore default values
   - Export buttons (SVG, PDF formats)
   - Optional: numerical input fields next to sliders

2. **Right Panel (Display Panel)**:
   - Canvas showing all three pattern pieces
   - Real-time update on slider change
   - Zoom/pan capabilities
   - Option to toggle grid/measurements
   - Option to show original pattern overlay

### GUI Framework Options
- **PyQt5/PyQt6**: Professional, feature-rich, cross-platform
- **Tkinter**: Built-in, simpler, lighter
- **Kivy**: Modern, touch-friendly

Recommended: **PyQt5/PyQt6** for best balance of features and rendering capabilities

## Technical Dependencies
```
opencv-python      # Image processing, edge detection
numpy              # Numerical operations
scipy              # Curve fitting, interpolation
matplotlib         # Curve visualization (optional)
PyQt5 / PyQt6      # GUI framework
svgwrite           # SVG export
reportlab          # PDF export (optional)
Pillow             # Image handling
scikit-image       # Additional image processing
```

## Key Challenges & Considerations

1. **Semantic Understanding**:
   - How to identify which parts of the pattern correspond to which features (shoulder, armpit, etc.)?
   - May require manual labeling of key points during initial conversion
   - Consider interactive tool for marking key points on first run

2. **Transformation Constraints**:
   - How should width changes affect different parts (body vs. neckline)?
   - Should armpit and shoulder adjustments affect surrounding curves?
   - How to maintain seam alignment between front/back pieces?

3. **Edge Distortion**:
   - What type of distortion? (Perlin noise, sinusoidal, random?)
   - Should distortion be symmetrical on matching seams?
   - Frequency and amplitude control?

4. **Curve Representation**:
   - B-splines offer smooth, continuous curves
   - Bézier curves provide intuitive control points
   - Polylines simpler but less smooth

5. **Real-time Performance**:
   - Curve recalculation must be fast (<100ms)
   - Rendering optimization for smooth slider interaction
   - Consider caching intermediate results

## Questions for Clarification

1. **Pattern Image Format**:
   - Are the patterns black outlines on white background?- Yes
   - Are there internal markings (darts, notches) to preserve? - No
   - What resolution are the images? - 768 x 768

2. **Parameter Ranges**:
   - What are the min/max values for each parameter (e.g., length ±20%) - Customize for now, might fix later
   - Should there be default/recommended values? - the ones that create the initial images

3. **Seam Alignment**:
   - Do front and back pieces need to maintain matching seam lengths? - Try to keep them compatible
   - Should sleeve armhole match body armhole automatically? - No, I want some configurations to be less anatomical (e.g. oversized) than others 

4. **Edge Distortion Details**:
   - Should distortion be applied to all edges or specific edges only? - all
   - Is this for aesthetic purposes or to simulate fabric behavior? - aesthetic
   - Should the distortion pattern be regeneratable/seedable? - No 

5. **Export Requirements**:
   - Should exports include measurements/annotations? - Yes
   - Required paper size (A4, A0, custom)? - A0
   - Should exports show seam allowances? - No

6. **Measurement Units**:
   - Work in pixels, centimeters, or inches? - cm
   - Should the app display actual dimensions? - no, only show cm size while keeping render size comfortable to view

7. **Initial Conversion**:
   - Should the conversion be automatic or semi-automatic with user guidance? - automatic
   - Do you have preference for curve fitting method? - no, use best to fit

8. **Multi-size Support**:
   - Will there be different base patterns for different sizes? - no
   - Or should one base pattern scale to all sizes? - Do not consider various sizing, the work is being done on single size at a time.
