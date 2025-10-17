import numpy as np
from scipy.spatial import ConvexHull

def extract_features(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mean_z = np.mean(z)
    std_z = np.std(z)
    min_z = np.min(z)
    max_z = np.max(z)
    height_range = max_z - min_z
    bbox_volume = (np.ptp(x) * np.ptp(y) * height_range)
    n_points = len(points)
    density = n_points / bbox_volume if bbox_volume > 0 else 0
    try:
        hull = ConvexHull(points[:, :2])
        area_xy = hull.volume
    except:
        area_xy = 0
    height_to_area = height_range / area_xy if area_xy > 0 else 0
    return [mean_z, std_z, min_z, max_z, height_range, bbox_volume, n_points, density, area_xy, height_to_area]
