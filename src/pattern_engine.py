"""Pattern transformation engine for parametric modifications."""
import numpy as np
from pathlib import Path
from src.curve_operations import CurveManipulator, reconstruct_spline_from_points
from utils.geometry import scale_points, translate_points, add_noise_to_curve


class PatternEngine:
    """Manages pattern transformations based on parameters."""

    def __init__(self, curves_dir='BasePatternCurves'):
        """
        Initialize pattern engine.

        Args:
            curves_dir: Directory containing curve JSON files
        """
        self.curves_dir = Path(curves_dir)
        self.patterns = {}
        self.load_patterns()

        # Default parameter ranges
        self.param_ranges = {
            'length': {'min': 0.5, 'max': 1.5, 'default': 1.0},
            'width': {'min': 0.5, 'max': 1.5, 'default': 1.0},
            'armpit_level': {'min': -50, 'max': 50, 'default': 0},  # pixels
            'shoulder_level': {'min': -50, 'max': 50, 'default': 0},  # pixels
            'distortion': {'min': 0, 'max': 10, 'default': 0},  # amplitude
        }

    def load_patterns(self):
        """Load all pattern curve files."""
        pattern_files = {
            'back': self.curves_dir / 'back_curves.json',
            'front': self.curves_dir / 'front_curves.json',
            'sleeve': self.curves_dir / 'sleeve_curves.json',
        }

        for name, filepath in pattern_files.items():
            if filepath.exists():
                self.patterns[name] = CurveManipulator(filepath)
            else:
                print(f"Warning: {filepath} not found")

    def reset_all(self):
        """Reset all patterns to original."""
        for pattern in self.patterns.values():
            pattern.reset()

    def apply_transformations(self, pattern_name, params):
        """
        Apply transformations to a pattern.

        Args:
            pattern_name: 'back', 'front', or 'sleeve'
            params: Dictionary of parameter values
                {
                    'length': 1.0,
                    'width': 1.0,
                    'armpit_level': 0,
                    'shoulder_level': 0,
                    'distortion': 0
                }

        Returns:
            Array of transformed points
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Pattern '{pattern_name}' not found")

        pattern = self.patterns[pattern_name]

        # Start with original data
        pattern.reset()

        # Check distortion parameter
        distortion = params.get('distortion', 0)

        # Get evaluated points - apply distortion at evaluation if we have segment-based data
        if distortion > 0:
            points = pattern.evaluate(
                num_points=200,
                apply_distortion=True,
                distortion_params={'amplitude': distortion, 'frequency': 8}
            )
        else:
            points = pattern.evaluate(num_points=200)

        # Get bounds and key points for reference
        bounds = pattern.get_bounds()
        key_points = pattern.get_key_points()

        # Calculate center for scaling operations
        center_x = bounds['min_x'] + bounds['width'] / 2
        center_y = bounds['min_y'] + bounds['height'] / 2
        center = np.array([center_x, center_y])

        # 1. Apply length scaling (vertical)
        length_scale = params.get('length', 1.0)
        if length_scale != 1.0:
            # Scale from top (preserve shoulder position)
            top_y = bounds['min_y']
            points = self._scale_from_point(points, 1.0, length_scale, pivot_x=center_x, pivot_y=top_y)

        # 2. Apply width scaling (horizontal)
        width_scale = params.get('width', 1.0)
        if width_scale != 1.0:
            points = scale_points(points, scale_x=width_scale, scale_y=1.0, origin=center)

        # 3. Apply armpit level adjustment
        armpit_offset = params.get('armpit_level', 0)
        if armpit_offset != 0 and key_points.get('armpit_left') or key_points.get('armpit_right'):
            points = self._adjust_region(points, bounds, 'armpit', armpit_offset)

        # 4. Apply shoulder level adjustment
        shoulder_offset = params.get('shoulder_level', 0)
        if shoulder_offset != 0 and (key_points.get('shoulder_left') or key_points.get('shoulder_right')):
            points = self._adjust_region(points, bounds, 'shoulder', shoulder_offset)

        # Update pattern with new spline
        new_tck = reconstruct_spline_from_points(points)
        pattern.set_tck(new_tck)

        return points

    def _scale_from_point(self, points, scale_x, scale_y, pivot_x, pivot_y):
        """
        Scale points from a specific pivot point.

        Args:
            points: Array of points
            scale_x, scale_y: Scale factors
            pivot_x, pivot_y: Pivot point coordinates

        Returns:
            Scaled points
        """
        pivot = np.array([pivot_x, pivot_y])
        translated = points - pivot
        scaled = translated * [scale_x, scale_y]
        return scaled + pivot

    def _adjust_region(self, points, bounds, region_type, offset):
        """
        Adjust a specific region (armpit or shoulder) vertically.

        Args:
            points: Array of points
            bounds: Bounds dictionary
            region_type: 'armpit' or 'shoulder'
            offset: Vertical offset in pixels (positive = down)

        Returns:
            Adjusted points
        """
        # Define region ranges (as fraction of height from top)
        if region_type == 'armpit':
            region_start = 0.25
            region_end = 0.45
        elif region_type == 'shoulder':
            region_start = 0.05
            region_end = 0.25
        else:
            return points

        top_y = bounds['min_y']
        height = bounds['height']

        # Calculate actual y-coordinates for region
        region_y_start = top_y + height * region_start
        region_y_end = top_y + height * region_end

        # Apply smooth transition using cosine interpolation
        adjusted_points = points.copy()
        for i, point in enumerate(points):
            y = point[1]

            if region_y_start <= y <= region_y_end:
                # Calculate interpolation factor (0 to 1 across region)
                t = (y - region_y_start) / (region_y_end - region_y_start)
                # Smooth interpolation (cosine)
                smooth_t = (1 - np.cos(t * np.pi)) / 2
                # Apply offset with smooth transition
                adjusted_points[i, 1] += offset * smooth_t

        return adjusted_points

    def get_all_transformed_patterns(self, params):
        """
        Get all patterns with transformations applied.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary mapping pattern names to transformed points
        """
        results = {}
        for name in self.patterns.keys():
            results[name] = self.apply_transformations(name, params)
        return results

    def get_param_ranges(self):
        """Get parameter ranges for UI."""
        return self.param_ranges

    def get_default_params(self):
        """Get default parameter values."""
        return {name: info['default'] for name, info in self.param_ranges.items()}
