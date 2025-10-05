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

    def detect_corners(self, points, angle_threshold=130, min_distance=80, window=30):
        """
        Detect sharp corner points (actual pattern corners).

        Args:
            points: Array of contour points
            angle_threshold: Angle threshold in degrees (lower = sharper corner)
            min_distance: Minimum distance between corners in pixels
            window: Window size for angle calculation

        Returns:
            List of corner point indices
        """
        points = np.array(points)
        n = len(points)
        angles = []

        # Calculate angles at each point
        for i in range(n):
            # Get neighboring points with larger window
            p1 = points[(i - window) % n]
            p2 = points[i]
            p3 = points[(i + window) % n]

            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append({'idx': i, 'angle': angle, 'point': p2})

        # Sort by angle to get sharpest corners first
        angles = sorted(angles, key=lambda a: a['angle'])

        # Select only sharp corners that are far apart
        corners = []
        for corner in angles:
            if corner['angle'] >= angle_threshold:
                break  # No more sharp corners

            too_close = False
            for existing in corners:
                dist = np.linalg.norm(corner['point'] - existing['point'])
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                corners.append(corner)

        # Sort by position along contour
        corners = sorted(corners, key=lambda c: c['idx'])

        return corners

    def is_segment_straight(self, points, tolerance=5):
        """
        Check if a segment is approximately straight.

        Args:
            points: Array of points in the segment
            tolerance: Maximum deviation from straight line in pixels

        Returns:
            True if segment is straight, False otherwise
        """
        if len(points) < 3:
            return True

        # Fit a line from start to end
        start = points[0]
        end = points[-1]

        # Calculate distance from each point to the line
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1:
            return True

        line_vec_norm = line_vec / line_len

        max_dist = 0
        for point in points[1:-1]:
            # Vector from start to point
            point_vec = point - start
            # Project onto line
            projection = np.dot(point_vec, line_vec_norm) * line_vec_norm
            # Distance from point to line
            dist = np.linalg.norm(point_vec - projection)
            max_dist = max(max_dist, dist)

        return max_dist <= tolerance

    def convert_to_curves(self, contour, num_control_points=50):
        """
        Convert contour to curve representation using corner-based splines.

        Args:
            contour: OpenCV contour
            num_control_points: Number of control points for spline (deprecated, uses corners)

        Returns:
            Dictionary with curve data
        """
        # Extract points from contour
        points = contour.reshape(-1, 2).astype(float)

        # Detect corner points
        corners = self.detect_corners(points)

        if len(corners) < 3:
            raise ValueError(f"Not enough corners detected: {len(corners)}. Need at least 3.")

        # Create segments between corners - either lines or curves
        segments = []
        n_corners = len(corners)

        for i in range(n_corners):
            start_idx = corners[i]['idx']
            end_idx = corners[(i + 1) % n_corners]['idx']

            # Extract points for this segment
            if end_idx > start_idx:
                segment_points = points[start_idx:end_idx+1]
            else:
                # Wrap around
                segment_points = np.vstack([points[start_idx:], points[:end_idx+1]])

            # Check if segment is straight or curved
            is_straight = self.is_segment_straight(segment_points)

            if is_straight:
                # Store as a line
                segments.append({
                    'type': 'line',
                    'start_corner': i,
                    'end_corner': (i + 1) % n_corners,
                    'start_point': convert_numpy_to_list(segment_points[0]),
                    'end_point': convert_numpy_to_list(segment_points[-1])
                })
            else:
                # Store as a curve - use just a few sample points
                # Downsample to ~5 control points
                n_samples = min(5, len(segment_points))
                indices = np.linspace(0, len(segment_points)-1, n_samples, dtype=int)
                control_pts = segment_points[indices]

                segments.append({
                    'type': 'curve',
                    'start_corner': i,
                    'end_corner': (i + 1) % n_corners,
                    'control_points': convert_numpy_to_list(control_pts)
                })

        return {
            'corners': convert_numpy_to_list([{
                'point': c['point'].tolist(),
                'angle': c['angle']
            } for c in corners]),
            'segments': segments,
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
