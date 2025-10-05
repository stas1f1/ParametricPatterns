"""Operations for manipulating parametric curves."""
import numpy as np
from scipy import interpolate
import json
from pathlib import Path


class CurveManipulator:
    """Handles loading and manipulating curve data."""

    def __init__(self, curve_file):
        """
        Initialize with curve data file.

        Args:
            curve_file: Path to JSON curve data file
        """
        self.curve_file = Path(curve_file)
        self.original_data = None
        self.current_data = None
        self.load()

    def load(self):
        """Load curve data from JSON."""
        with open(self.curve_file, 'r') as f:
            self.original_data = json.load(f)

        # Deep copy for current data
        self.current_data = json.loads(json.dumps(self.original_data))

    def reset(self):
        """Reset to original curve data."""
        self.current_data = json.loads(json.dumps(self.original_data))

    def _reconstruct_points_from_segments(self, apply_distortion=False, distortion_params=None):
        """
        Reconstruct full curve points from corner-based segments.

        Args:
            apply_distortion: Whether to apply distortion effect
            distortion_params: Dictionary with 'amplitude' and 'frequency' for distortion

        Returns:
            Array of (x, y) points representing the complete curve
        """
        corners = self.current_data.get('corners', [])
        segments = self.current_data.get('segments', [])

        if not corners or not segments:
            raise ValueError("Invalid curve format: missing corners or segments")

        all_points = []
        corner_indices = []  # Track which points are corners

        for segment in segments:
            # Mark the start of this segment as a corner
            corner_indices.append(len(all_points))

            if segment['type'] == 'line':
                # For lines, just use start and end points
                start = np.array(segment['start_point'])
                end = np.array(segment['end_point'])
                # Interpolate between start and end for smooth curve
                line_points = np.linspace(start, end, 20)
                all_points.extend(line_points[:-1])  # Exclude last to avoid duplicates

            elif segment['type'] == 'curve':
                # For curves, use control points to create smooth curve
                control_pts = np.array(segment['control_points'])

                # Fit spline through control points
                if len(control_pts) >= 4:
                    tck, u = interpolate.splprep([control_pts[:, 0], control_pts[:, 1]], k=3, s=0)
                elif len(control_pts) == 3:
                    tck, u = interpolate.splprep([control_pts[:, 0], control_pts[:, 1]], k=2, s=0)
                else:
                    # Too few points, just use them directly
                    all_points.extend(control_pts[:-1])
                    continue

                # Evaluate spline
                u_new = np.linspace(0, 1, 20)
                x, y = interpolate.splev(u_new, tck)
                curve_points = np.column_stack([x, y])
                all_points.extend(curve_points[:-1])  # Exclude last to avoid duplicates

        # Close the curve by adding the first point at the end
        if all_points:
            all_points.append(all_points[0])

        points = np.array(all_points)

        # Apply distortion if requested
        if apply_distortion and distortion_params:
            from utils.geometry import add_noise_to_curve_with_corners
            points = add_noise_to_curve_with_corners(
                points,
                corner_indices=corner_indices,
                amplitude=distortion_params.get('amplitude', 0),
                frequency=distortion_params.get('frequency', 8)
            )

        return points

    def get_tck(self):
        """
        Get spline tck tuple from current data.

        Returns:
            Tuple of (t, c, k) for scipy splev
        """
        # Check if old format (single spline_tck) or new format (segments)
        if 'spline_tck' in self.current_data:
            # Old format
            tck_list = self.current_data['spline_tck']
            # Convert lists back to arrays
            t = np.array(tck_list[0])
            c = [np.array(tck_list[1][0]), np.array(tck_list[1][1])]
            k = tck_list[2]
            return (t, c, k)
        else:
            # New format - reconstruct from segments
            points = self._reconstruct_points_from_segments()
            return reconstruct_spline_from_points(points)

    def set_tck(self, tck):
        """
        Update spline tck in current data.

        Args:
            tck: Tuple of (t, c, k)
        """
        t, c, k = tck
        self.current_data['spline_tck'] = [
            t.tolist(),
            [c[0].tolist(), c[1].tolist()],
            k
        ]

    def evaluate(self, num_points=200, apply_distortion=False, distortion_params=None):
        """
        Evaluate current spline.

        Args:
            num_points: Number of points to evaluate
            apply_distortion: Whether to apply distortion effect
            distortion_params: Dictionary with 'amplitude' and 'frequency' for distortion

        Returns:
            Array of (x, y) points
        """
        # Check if using new format (segments) - if so, use segment reconstruction with distortion
        if 'segments' in self.current_data and apply_distortion:
            return self._reconstruct_points_from_segments(
                apply_distortion=apply_distortion,
                distortion_params=distortion_params
            )

        tck = self.get_tck()
        u = np.linspace(0, 1, num_points)
        x, y = interpolate.splev(u, tck)
        points = np.column_stack([x, y])

        # Apply distortion if requested
        if apply_distortion and distortion_params:
            from utils.geometry import add_noise_to_curve
            points = add_noise_to_curve(
                points,
                amplitude=distortion_params.get('amplitude', 0),
                frequency=distortion_params.get('frequency', 8)
            )

        return points

    def get_key_points(self):
        """Get key points dictionary."""
        return self.current_data.get('key_points', {})

    def get_bounds(self):
        """
        Get bounding box of current curve.

        Returns:
            Dictionary with min/max x and y
        """
        points = self.evaluate()
        return {
            'min_x': np.min(points[:, 0]),
            'max_x': np.max(points[:, 0]),
            'min_y': np.min(points[:, 1]),
            'max_y': np.max(points[:, 1]),
            'width': np.max(points[:, 0]) - np.min(points[:, 0]),
            'height': np.max(points[:, 1]) - np.min(points[:, 1]),
        }


def reconstruct_spline_from_points(points):
    """
    Create a new spline from modified points.

    Args:
        points: Array of (x, y) points

    Returns:
        tck tuple
    """
    points = np.array(points)

    # Create parametric representation
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    u = distances / distances[-1]

    # Fit spline
    tck, _ = interpolate.splprep([points[:, 0], points[:, 1]], u=u, s=len(points)*0.5, k=3)

    return tck
