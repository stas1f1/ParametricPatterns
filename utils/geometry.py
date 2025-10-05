"""Geometric helper functions for curve operations."""
import numpy as np
from scipy import interpolate


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def smooth_contour(points, smoothing_factor=3):
    """
    Smooth a contour using B-spline interpolation.

    Args:
        points: Array of (x, y) points
        smoothing_factor: Degree of smoothing (higher = smoother)

    Returns:
        Smoothed array of points
    """
    if len(points) < 4:
        return points

    # Ensure closed contour
    points = np.array(points)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Create parametric representation
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) / distances[-1]

    # Fit spline
    try:
        tck, u = interpolate.splprep([points[:, 0], points[:, 1]], u=distances, s=smoothing_factor, per=True)
        u_new = np.linspace(0, 1, len(points) * 2)
        x_new, y_new = interpolate.splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except:
        return points


def fit_bspline(points, num_control_points=None):
    """
    Fit a B-spline curve to points.

    Args:
        points: Array of (x, y) points
        num_control_points: Number of control points (None = auto)

    Returns:
        Dictionary with tck (spline parameters) and control points
    """
    points = np.array(points)

    # Create parametric representation
    distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    distances = distances / distances[-1]

    # Determine smoothing based on number of points
    if num_control_points is None:
        smoothing = len(points) * 0.5
    else:
        smoothing = len(points) / num_control_points

    # Fit spline
    tck, u = interpolate.splprep([points[:, 0], points[:, 1]], u=distances, s=smoothing, k=3)

    return {
        'tck': tck,
        'u': u,
        'type': 'bspline'
    }


def evaluate_bspline(spline_data, num_points=100):
    """
    Evaluate a B-spline at specified number of points.

    Args:
        spline_data: Dictionary from fit_bspline
        num_points: Number of points to evaluate

    Returns:
        Array of (x, y) points
    """
    u_new = np.linspace(0, 1, num_points)
    x, y = interpolate.splev(u_new, spline_data['tck'])
    return np.column_stack([x, y])


def find_extrema(points):
    """
    Find extreme points (min/max x and y).

    Args:
        points: Array of (x, y) points

    Returns:
        Dictionary with indices and values of extrema
    """
    points = np.array(points)

    return {
        'top': {'idx': np.argmin(points[:, 1]), 'point': points[np.argmin(points[:, 1])]},
        'bottom': {'idx': np.argmax(points[:, 1]), 'point': points[np.argmax(points[:, 1])]},
        'left': {'idx': np.argmin(points[:, 0]), 'point': points[np.argmin(points[:, 0])]},
        'right': {'idx': np.argmax(points[:, 0]), 'point': points[np.argmax(points[:, 0])]},
    }


def scale_points(points, scale_x=1.0, scale_y=1.0, origin=None):
    """
    Scale points around an origin.

    Args:
        points: Array of (x, y) points
        scale_x: X scale factor
        scale_y: Y scale factor
        origin: Origin point (None = centroid)

    Returns:
        Scaled points
    """
    points = np.array(points)

    if origin is None:
        origin = np.mean(points, axis=0)

    translated = points - origin
    scaled = translated * [scale_x, scale_y]
    return scaled + origin


def translate_points(points, dx=0, dy=0):
    """Translate points by dx, dy."""
    points = np.array(points)
    return points + [dx, dy]


def add_noise_to_curve(points, amplitude=1.0, frequency=10, seed=None):
    """
    Add sinusoidal/random distortion to curve for aesthetic effect.

    Args:
        points: Array of (x, y) points
        amplitude: Maximum displacement in pixels
        frequency: Number of oscillations along the curve
        seed: Random seed (None = random)

    Returns:
        Distorted points
    """
    if seed is not None:
        np.random.seed(seed)

    points = np.array(points)
    num_points = len(points)

    # Calculate normals at each point
    normals = np.zeros_like(points)
    for i in range(num_points):
        if i == 0:
            tangent = points[1] - points[-1]
        elif i == num_points - 1:
            tangent = points[0] - points[-2]
        else:
            tangent = points[i+1] - points[i-1]

        # Normal is perpendicular to tangent
        tangent_normalized = tangent / (np.linalg.norm(tangent) + 1e-8)
        normals[i] = [-tangent_normalized[1], tangent_normalized[0]]

    # Generate distortion pattern
    t = np.linspace(0, 2 * np.pi * frequency, num_points)
    distortion = amplitude * (np.sin(t) + 0.3 * np.random.randn(num_points))

    # Apply distortion along normals
    distorted = points + normals * distortion[:, np.newaxis]

    return distorted
