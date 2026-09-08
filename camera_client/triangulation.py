"""Pure geometry and linear algebra functions for multi-ray triangulation
and statistical consistency checks."""

import numpy as np


def closest_point_on_ray(origin, direction, point):
    """Project a point onto a ray, returning the closest point on the ray.

    Args:
        origin: (3,) ray origin
        direction: (3,) unit ray direction
        point: (3,) point to project

    Returns:
        (3,) closest point on the ray
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    t = direction @ (point - origin)
    return origin + t * direction


def least_squares_intersection(rays):
    """Find the point minimizing sum of squared distances to all rays (SVD).

    Each ray is (origin, direction). Solves the linear least squares problem:
    for each ray i: (I - e_i e_i^T)(P - k_i) = 0

    Args:
        rays: list of (origin, direction) tuples, len >= 2
              origin: (3,) array, direction: (3,) unit vector

    Returns:
        (3,) approximate intersection point, or None if degenerate
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in rays:
        origin = np.asarray(origin, dtype=np.float64)
        e = np.asarray(direction, dtype=np.float64)
        P = np.eye(3) - np.outer(e, e)
        A += P
        b += P @ origin
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def fuse_covariances(covariances):
    """Fuse covariance matrices via information fusion.

    Sigma_fused = (Sigma_1^-1 + ... + Sigma_n^-1)^-1

    Args:
        covariances: list of (3, 3) covariance matrices

    Returns:
        (3, 3) fused covariance matrix
    """
    info_matrix = np.zeros((3, 3))
    for cov in covariances:
        info_matrix += np.linalg.inv(cov)
    return np.linalg.inv(info_matrix)


def information_fusion(points, covariances):
    """Fuse N point estimates with their covariances.

    P_fused = Sigma_fused * sum(Sigma_i^-1 * P_i)
    Sigma_fused = (sum(Sigma_i^-1))^-1

    Args:
        points: list of (3,) arrays
        covariances: list of (3, 3) covariance matrices

    Returns:
        (P_fused, Sigma_fused) — fused point (3,) and covariance (3, 3)
    """
    info_matrix = np.zeros((3, 3))
    info_vector = np.zeros(3)
    for p, cov in zip(points, covariances):
        inv_cov = np.linalg.inv(cov)
        info_matrix += inv_cov
        info_vector += inv_cov @ np.asarray(p)
    sigma_fused = np.linalg.inv(info_matrix)
    p_fused = sigma_fused @ info_vector
    return p_fused, sigma_fused


def mahalanobis_distance(p1, cov1, p2, cov2):
    """Squared Mahalanobis distance between two measurements.

    delta^2 = (p1 - p2)^T (cov1 + cov2)^-1 (p1 - p2)

    Args:
        p1: (3,) first point
        cov1: (3, 3) covariance for p1
        p2: (3,) second point
        cov2: (3, 3) covariance for p2

    Returns:
        Squared Mahalanobis distance (scalar)
    """
    diff = np.asarray(p1) - np.asarray(p2)
    cov_sum = np.asarray(cov1) + np.asarray(cov2)
    return float(diff @ np.linalg.inv(cov_sum) @ diff)
