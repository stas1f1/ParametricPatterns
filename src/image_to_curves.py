"""Convert pattern images to parametric curves."""
import cv2
import numpy as np
import json
from pathlib import Path
from utils.geometry import fit_bspline, evaluate_bspline, find_extrema


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays and types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj


class PatternConverter:
    """Converts pattern images to curve representations."""

    def __init__(self, image_path, pixels_per_cm=10):
        """
        Initialize converter.

        Args:
            image_path: Path to pattern image (PNG)
            pixels_per_cm: Scale factor for cm conversion
        """
        self.image_path = Path(image_path)
        self.pixels_per_cm = pixels_per_cm
        self.image = None
        self.contours = []
        self.curve_data = None

    def load_image(self):
        """Load and preprocess image."""
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return gray

    def detect_edges(self, gray_image, blur_kernel=5, canny_low=50, canny_high=150):
        """
        Detect edges in the pattern.

        Args:
            gray_image: Grayscale image
            blur_kernel: Gaussian blur kernel size
            canny_low: Canny low threshold
            canny_high: Canny high threshold

        Returns:
            Binary edge image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        return edges

    def find_contours(self, edges):
        """
        Find contours from edge image.

        Args:
            edges: Binary edge image

        Returns:
            List of contours
        """
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area to remove noise
        min_area = 1000
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Sort by area (largest first)
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

        self.contours = filtered_contours
        return filtered_contours

    def identify_key_points(self, contour_points):
        """
        Identify semantic key points on the pattern.

        Args:
            contour_points: Array of contour points

        Returns:
            Dictionary of labeled key points
        """
        extrema = find_extrema(contour_points)

        # Height of the pattern
        height = extrema['bottom']['point'][1] - extrema['top']['point'][1]

        # Estimate key levels (these are heuristics, may need adjustment per pattern type)
        shoulder_y = extrema['top']['point'][1] + height * 0.15
        armpit_y = extrema['top']['point'][1] + height * 0.35

        # Find points closest to these levels
        shoulder_candidates = []
        armpit_candidates = []

        for i, point in enumerate(contour_points):
            if abs(point[1] - shoulder_y) < height * 0.05:
                shoulder_candidates.append({'idx': i, 'point': point})
            if abs(point[1] - armpit_y) < height * 0.05:
                armpit_candidates.append({'idx': i, 'point': point})

        # Select leftmost and rightmost for shoulders
        if shoulder_candidates:
            shoulder_candidates = sorted(shoulder_candidates, key=lambda p: p['point'][0])
            shoulder_left = shoulder_candidates[0] if shoulder_candidates else None
            shoulder_right = shoulder_candidates[-1] if len(shoulder_candidates) > 1 else None
        else:
            shoulder_left = shoulder_right = None

        # Select leftmost and rightmost for armpits
        if armpit_candidates:
            armpit_candidates = sorted(armpit_candidates, key=lambda p: p['point'][0])
            armpit_left = armpit_candidates[0] if armpit_candidates else None
            armpit_right = armpit_candidates[-1] if len(armpit_candidates) > 1 else None
        else:
            armpit_left = armpit_right = None

        return {
            'top': extrema['top'],
            'bottom': extrema['bottom'],
            'left': extrema['left'],
            'right': extrema['right'],
            'shoulder_left': shoulder_left,
            'shoulder_right': shoulder_right,
            'armpit_left': armpit_left,
            'armpit_right': armpit_right,
        }

    def convert_to_curves(self, contour, num_control_points=50):
        """
        Convert contour to curve representation.

        Args:
            contour: OpenCV contour
            num_control_points: Number of control points for spline

        Returns:
            Dictionary with curve data
        """
        # Extract points from contour
        points = contour.reshape(-1, 2).astype(float)

        # Fit B-spline
        spline = fit_bspline(points, num_control_points=num_control_points)

        # Identify key points
        key_points = self.identify_key_points(points)

        # Evaluate spline for storage
        evaluated_points = evaluate_bspline(spline, num_points=200)

        return {
            'original_points': convert_numpy_to_list(points),
            'spline_tck': convert_numpy_to_list(spline['tck']),
            'spline_type': spline['type'],
            'key_points': convert_numpy_to_list(key_points),
            'evaluated_points': convert_numpy_to_list(evaluated_points),
            'pixels_per_cm': self.pixels_per_cm,
            'image_size': list(self.image.shape[:2])
        }

    def process(self):
        """
        Full processing pipeline.

        Returns:
            Curve data dictionary
        """
        # Load and process image
        gray = self.load_image()
        edges = self.detect_edges(gray)
        contours = self.find_contours(edges)

        if not contours:
            raise ValueError("No contours found in image")

        # Use largest contour
        main_contour = contours[0]

        # Convert to curves
        self.curve_data = self.convert_to_curves(main_contour)

        return self.curve_data

    def save_curves(self, output_path):
        """
        Save curve data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        if self.curve_data is None:
            raise ValueError("No curve data to save. Run process() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.curve_data, f, indent=2)

        return output_path


def convert_all_patterns(pattern_dir='.', output_dir='BasePatternCurves', pixels_per_cm=10):
    """
    Convert all base patterns to curves.

    Args:
        pattern_dir: Directory containing pattern images
        output_dir: Directory to save curve data
        pixels_per_cm: Scale factor

    Returns:
        Dictionary mapping pattern names to curve data
    """
    pattern_dir = Path(pattern_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = {
        'back': 'BasePattern_Back.png',
        'front': 'BasePattern_Front.png',
        'sleeve': 'BasePattern_Sleeve.png'
    }

    results = {}

    for name, filename in patterns.items():
        image_path = pattern_dir / filename

        if not image_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Converting {filename}...")

        converter = PatternConverter(image_path, pixels_per_cm=pixels_per_cm)
        curve_data = converter.process()

        output_path = output_dir / f'{name}_curves.json'
        converter.save_curves(output_path)

        results[name] = curve_data
        print(f"  Saved to {output_path}")

    return results


if __name__ == '__main__':
    # Run conversion
    convert_all_patterns()
