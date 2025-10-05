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

    def get_tck(self):
        """
        Get spline tck tuple from current data.

        Returns:
            Tuple of (t, c, k) for scipy splev
        """
        tck_list = self.current_data['spline_tck']
        # Convert lists back to arrays
        t = np.array(tck_list[0])
        c = [np.array(tck_list[1][0]), np.array(tck_list[1][1])]
        k = tck_list[2]
        return (t, c, k)

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

    def evaluate(self, num_points=200):
        """
        Evaluate current spline.

        Args:
            num_points: Number of points to evaluate

        Returns:
            Array of (x, y) points
        """
        tck = self.get_tck()
        u = np.linspace(0, 1, num_points)
        x, y = interpolate.splev(u, tck)
        return np.column_stack([x, y])

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
