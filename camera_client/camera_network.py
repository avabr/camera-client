import numpy as np
from camera_client.error_model import CameraSpatialCovariance
from camera_client import triangulation


def _points_in_polygon(points_xy, polygon_xy):
    """Fully vectorized ray casting point-in-polygon test.

    Args:
        points_xy: (N, 2) array of point coordinates
        polygon_xy: (M, 2) array of polygon vertices

    Returns:
        (N,) boolean array
    """
    px = points_xy[:, 0][:, np.newaxis]  # (N, 1)
    py = points_xy[:, 1][:, np.newaxis]  # (N, 1)

    # Edge vertices: i -> j for all edges
    xi = polygon_xy[:, 0]  # (M,)
    yi = polygon_xy[:, 1]  # (M,)
    xj = np.roll(xi, 1)    # (M,)
    yj = np.roll(yi, 1)    # (M,)

    # Broadcasting: (N, 1) vs (M,) -> (N, M)
    # Suppress divide-by-zero for horizontal edges (yj == yi); masked out by the first condition
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)

    # Odd number of crossings = inside
    return np.bitwise_xor.reduce(crosses, axis=1)


class NetworkCovariance:
    """Result of CameraNetwork.get_covariance().

    Access per-camera covariance lists by camera_id:
        result[1177]  — list of N covariance matrices (or None) for camera 1177

    Access fused covariance:
        result.fused  — list of N fused covariance matrices (or None)
    """

    def __init__(self, cameras, fused):
        self._cameras = cameras  # dict {camera_id: [cov_or_none, ...]}
        self.fused = fused       # list [cov_or_none, ...]

    def __getitem__(self, camera_id):
        return self._cameras[camera_id]

    @property
    def camera_ids(self):
        return list(self._cameras.keys())

    def visible_camera_ids(self, point_index=0):
        """Return list of camera_ids that see the given point."""
        return [cid for cid, covs in self._cameras.items() if covs[point_index] is not None]

    def __repr__(self):
        n = len(self.fused)
        n_visible = sum(1 for c in self.fused if c is not None)
        return f"NetworkCovariance(points={n}, visible={n_visible}, cameras={self.camera_ids})"


class CameraNetwork:
    """A network of cameras with spatial covariance models.

    Provides multi-camera uncertainty analysis: covariance fusion,
    triangulation, and spatial uncertainty mapping.

    Args:
        cameras: list of CameraProjection instances
    """

    def __init__(self, cameras):
        if not cameras:
            raise ValueError("CameraNetwork requires at least one camera")

        self.cameras = {}
        self.covariances = {}
        self._efov_gnd_xy = {}    # ground EFOV polygon vertices (M, 2)
        self._lifted_ctd_cache = {}  # (cid, h) -> lifted CTD polygon (M, 2)

        for cam in cameras:
            cid = cam.camera_id
            self.cameras[cid] = cam
            self.covariances[cid] = CameraSpatialCovariance(cam)

            efov = cam.ctd_geometry.get("efov_polygon")
            if efov and efov.get("coordinates"):
                pts_ctd = np.array(efov["coordinates"][0])
                pts_gnd = cam.ctd_to_gnd(pts_ctd, h=0)
                self._efov_gnd_xy[cid] = pts_gnd[:, :2]
            else:
                self._efov_gnd_xy[cid] = None

    @property
    def camera_ids(self):
        return list(self.cameras.keys())

    def __len__(self):
        return len(self.cameras)

    def __repr__(self):
        return f"CameraNetwork(cameras={self.camera_ids})"

    def _is_in_front(self, cid, point):
        """Check if point is in front of the camera (not behind)."""
        key_point = self.covariances[cid].key_point
        cam = self.cameras[cid]
        center_ray = cam.ctd_to_ray(np.array([[cam.im_width / 2, cam.im_height / 2]]))[0]
        return float(np.dot(point - key_point, center_ray)) > 0

    def _visible_camera_ids(self, point, use_efov):
        """Return list of camera_ids that see the given 3D point.

        When use_efov=True: lifts EFOV ground polygon to the point's height,
        projects to CTD, and checks if the point's CTD projection falls inside.
        Also checks that the point is in front of the camera.

        When use_efov=False: checks that the point projects onto valid image area.
        """
        visible = []
        for cid, cam in self.cameras.items():
            if use_efov:
                gnd_xy = self._efov_gnd_xy[cid]
                if gnd_xy is None:
                    continue
                if not self._is_in_front(cid, point):
                    continue
                # Lift EFOV vertices to point's height and project to CTD (cached)
                h = float(point[2])
                cache_key = (cid, h)
                if cache_key not in self._lifted_ctd_cache:
                    lifted_gnd = np.column_stack([gnd_xy, np.full(len(gnd_xy), h)])
                    self._lifted_ctd_cache[cache_key] = cam.gnd_to_ctd(lifted_gnd)
                lifted_ctd = self._lifted_ctd_cache[cache_key]
                # Check if point's CTD projection falls inside lifted polygon
                point_ctd = cam.gnd_to_ctd(point.reshape(1, 3))[0]
                if _points_in_polygon(point_ctd.reshape(1, 2), lifted_ctd)[0]:
                    visible.append(cid)
            else:
                ctd = cam.gnd_to_ctd(point.reshape(1, 3))[0]
                if 0 <= ctd[0] < cam.im_width and 0 <= ctd[1] < cam.im_height:
                    visible.append(cid)
        return visible

    def get_covariance(self, points, detection_sigma=0.0, sigma_binding=0.0, use_efov=True):
        """Compute per-camera and fused covariance for 3D points.

        Visibility is always checked:
        - use_efov=True (default): point's xy projection must fall inside the
          camera's ground EFOV polygon
        - use_efov=False: point must project onto valid image area

        Args:
            points: (N, 3) array of 3D points [x, y, z]
            detection_sigma: detection uncertainty (fraction of image width)
            sigma_binding: spatial binding uncertainty in meters
            use_efov: if True, use EFOV ground polygons for visibility;
                      if False, use image projection bounds

        Returns:
            NetworkCovariance with:
              result[camera_id] — list of N per-camera covariances (or None)
              result.fused       — list of N fused covariances (or None)
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {points.shape}")

        N = len(points)
        per_camera = {cid: [None] * N for cid in self.cameras}
        fused = [None] * N

        for i in range(N):
            p = points[i]
            visible_ids = self._visible_camera_ids(p, use_efov)

            covs_for_fusion = []
            for cid in visible_ids:
                cov = self.covariances[cid].get_covariance(p, detection_sigma, sigma_binding)
                if cov is not None:
                    per_camera[cid][i] = cov
                    covs_for_fusion.append(cov)

            if covs_for_fusion:
                fused[i] = triangulation.fuse_covariances(covs_for_fusion)

        return NetworkCovariance(per_camera, fused)

    def triangulate(self, observations, detection_sigma=0.0, sigma_binding=0.0, n_sigma=3.0):
        """Triangulate a 3D point from observations in multiple cameras.

        Stage 1: Approximate intersection via least squares (SVD, unweighted).
        Stage 2: Project approximate point onto each ray, compute covariances,
                 fuse via information fusion.
        Stage 3: Consistency check — if the worst ray point exceeds n_sigma
                 from the fused point, scale the covariance to fit.

        Args:
            observations: dict {camera_id: src_point} where src_point is (2,)
                          source (distorted) image coordinates
            detection_sigma: detection uncertainty (fraction of image width)
            sigma_binding: spatial binding uncertainty in meters
            n_sigma: consistency threshold in sigmas (default 3.0)

        Returns:
            (p_fused, sigma_prior, sigma_posterior) —
                p_fused: (3,) fused 3D point
                sigma_prior: (3, 3) a priori covariance from error models
                sigma_posterior: (3, 3) a posteriori covariance, inflated if
                    network is inconsistent (sigma_prior == sigma_posterior
                    when consistent)
            or None if triangulation fails (e.g. degenerate rays)
        """
        if len(observations) < 2:
            raise ValueError("Need at least 2 observations for triangulation")

        # Convert src -> ctd and build rays
        rays = []
        cam_ids = []
        for cid, src_pt in observations.items():
            cam = self.cameras[cid]
            ctd_pt = cam.src_to_ctd(np.array([src_pt]))[0]
            origin = self.covariances[cid].key_point
            direction = cam.ctd_to_ray(np.array([[float(ctd_pt[0]), float(ctd_pt[1])]]))[0]
            rays.append((origin, direction))
            cam_ids.append(cid)

        # Stage 1: unweighted least squares intersection
        p_approx = triangulation.least_squares_intersection(rays)
        if p_approx is None:
            return None

        # Stage 2: project onto rays, get covariances, fuse
        ray_points = []
        ray_covs = []
        for (origin, direction), cid in zip(rays, cam_ids):
            p_on_ray = triangulation.closest_point_on_ray(origin, direction, p_approx)
            cov = self.covariances[cid].get_covariance(
                p_on_ray, detection_sigma, sigma_binding
            )
            if cov is None:
                return None
            ray_points.append(p_on_ray)
            ray_covs.append(cov)

        p_fused, sigma_prior = triangulation.information_fusion(ray_points, ray_covs)

        # Stage 3: consistency check
        inv_sigma = np.linalg.inv(sigma_prior)
        d2_max = 0.0
        for p_on_ray in ray_points:
            diff = p_on_ray - p_fused
            d2 = float(diff @ inv_sigma @ diff)
            if d2 > d2_max:
                d2_max = d2

        threshold = n_sigma ** 2
        if d2_max > threshold:
            k = d2_max / threshold
            sigma_posterior = k * sigma_prior
        else:
            sigma_posterior = sigma_prior

        return p_fused, sigma_prior, sigma_posterior
