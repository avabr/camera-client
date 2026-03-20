import numpy as np
import sympy as sp
from camera_client.loading import read_npz_file


class CameraProjection:
    """
    Vectorized camera coordinate transformer for batch processing.

    This class is optimized for processing multiple points at once.
    All methods expect array inputs with shape (N, 2) or (N, 3).

    Handles transformations between:
    - src: Source (distorted) image coordinates
    - ctd: Corrected (undistorted) image coordinates
    - gnd: Ground (3D world) coordinates
    """

    def __init__(self, cam_archive_data):
        """
        Initialize the camera transformer with calibration data.

        Args:
            cam_archive_data: Dictionary containing:
                - src2ctd: Source to corrected distortion map (H x W x 2)
                - ctd2src: Corrected to source distortion map (H x W x 2)
                - x_gnd, y_gnd, z_gnd: String expressions for ctd -> ground
                - x_im, y_im: String expressions for ground -> ctd
        """
        data = cam_archive_data

        self.camera_id = data["camera_id"]
        self.ctd_geometry = data["ctd_geometry"]

        self.plan_scale = float(data["plan_scale"])
        self.plan_url = str(data["plan_url"])
        self.plan_width = int(data["plan_width"])
        self.plan_height = int(data["plan_height"])

        self.im_src_url = str(data["im_src_url"])
        self.im_ctd_url = str(data["im_ctd_url"])
        self.im_width = int(data["im_width"])
        self.im_height = int(data["im_height"])
        self.im_wh_size = (self.im_width, self.im_height)

        # Store lookup tables
        self.src2ctd_points_map = data["src2ctd"]
        self.ctd2src_points_map = data["ctd2src"]

        self.im_size = self.src2ctd_points_map.shape[:2]

        # Compile transformation expressions for ctd -> gnd
        x_im, y_im, proj_height = sp.symbols("x_im y_im proj_height")
        self._lambda_im2gnd = sp.lambdify(
            (x_im, y_im, proj_height), sp.sympify(data["exp_im2gnd"]), "numpy"
        )

        # Compile transformation expressions for gnd -> ctd
        x_gnd, y_gnd, z_gnd = sp.symbols("x_gnd y_gnd z_gnd")
        self._lambda_gnd2im = sp.lambdify(
            (x_gnd, y_gnd, z_gnd), sp.sympify(data["exp_gnd2im"]), "numpy"
        )

        # Compile keypoint expressions (camera position in world space)
        self._lambda_key_point = sp.lambdify(
            (), sp.sympify(data["exp_key_point"]), "numpy"
        )

        # Compile ray direction expressions for ctd -> ray
        self._lambda_im2ray = sp.lambdify(
            (x_im, y_im), sp.sympify(data["exp_im2ray"]), "numpy"
        )

    def src_to_ctd(self, points):
        """
        Transform from source (distorted) to corrected coordinates.

        Args:
            points: (N, 2) array of source points [[x1, y1], [x2, y2], ...]

        Returns:
            (N, 2) array of corrected points, with [nan, nan] for out-of-bounds
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = len(points)
        result = np.full((N, 2), np.nan, dtype=float)

        # Check for NaN input points
        valid_input = ~np.isnan(points).any(axis=1)

        if not valid_input.any():
            return result

        # Round to integer coordinates
        p_int = np.round(points[valid_input]).astype(int)

        # Vectorized bounds checking
        in_bounds = (
            (p_int[:, 0] >= 0)
            & (p_int[:, 0] < self.src2ctd_points_map.shape[1])
            & (p_int[:, 1] >= 0)
            & (p_int[:, 1] < self.src2ctd_points_map.shape[0])
        )

        # Create mask for points that are both valid input and in bounds
        valid_indices = np.where(valid_input)[0]
        final_valid_indices = valid_indices[in_bounds]
        valid_p_int = p_int[in_bounds]

        # Vectorized lookup using advanced indexing
        # Note: lookup map is [y, x] indexed
        result[final_valid_indices] = self.src2ctd_points_map[
            valid_p_int[:, 1], valid_p_int[:, 0]
        ]

        return result

    def ctd_to_src(self, points):
        """
        Transform from corrected to source (distorted) coordinates.

        Args:
            points: (N, 2) array of corrected points [[x1, y1], [x2, y2], ...]

        Returns:
            (N, 2) array of source points, with [nan, nan] for out-of-bounds
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = len(points)
        result = np.full((N, 2), np.nan, dtype=float)

        # Check for NaN input points
        valid_input = ~np.isnan(points).any(axis=1)

        if not valid_input.any():
            return result

        # Round to integer coordinates
        p_int = np.round(points[valid_input]).astype(int)

        # Vectorized bounds checking
        in_bounds = (
            (p_int[:, 0] >= 0)
            & (p_int[:, 0] < self.ctd2src_points_map.shape[1])
            & (p_int[:, 1] >= 0)
            & (p_int[:, 1] < self.ctd2src_points_map.shape[0])
        )

        # Create mask for points that are both valid input and in bounds
        valid_indices = np.where(valid_input)[0]
        final_valid_indices = valid_indices[in_bounds]
        valid_p_int = p_int[in_bounds]

        # Vectorized lookup using advanced indexing
        result[final_valid_indices] = self.ctd2src_points_map[
            valid_p_int[:, 1], valid_p_int[:, 0]
        ]

        return result

    def ctd_to_gnd(self, points, h):
        """
        Transform from corrected image coordinates to ground (3D world) coordinates.

        Args:
            points: (N, 2) array of corrected points [[x1, y1], [x2, y2], ...]
            h: Scalar height or (N,) array of heights for each point

        Returns:
            (N, 3) array of ground points [[x1, y1, z1], ...]
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = points.shape[0]
        x_im, y_im = points.T
        hs = np.asarray(h, dtype=float)

        # Call lambdified function: returns shape (3, 1, N)
        # Reshape directly to (N, 3)
        ps_gnd = np.asarray(self._lambda_im2gnd(x_im, y_im, hs)).reshape(3, N).T

        return ps_gnd

    def gnd_to_ctd(self, points):
        """
        Transform from ground (3D world) to corrected image coordinates.

        Args:
            points: (N, 3) array of ground points [[x1, y1, z1], [x2, y2, z2], ...]

        Returns:
            (N, 2) array of image points [[x1, y1], ...]
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {points.shape}")

        N = points.shape[0]
        x_gnd, y_gnd, z_gnd = points.T

        # Call lambdified function: returns shape (2, 1, N)
        # Reshape directly to (N, 2)
        ps_im = np.asarray(self._lambda_gnd2im(x_gnd, y_gnd, z_gnd)).reshape(2, N).T

        return ps_im

    def src_to_gnd(self, points, h):
        """
        Transform from source to ground coordinates.

        Args:
            points: (N, 2) array of source points
            h: Scalar height or (N,) array of heights

        Returns:
            (N, 3) array of ground points
        """
        ctd_points = self.src_to_ctd(points)
        return self.ctd_to_gnd(ctd_points, h)

    def gnd_to_src(self, points):
        """
        Transform from ground to source coordinates.

        Args:
            points: (N, 3) array of ground points

        Returns:
            (N, 2) array of source points
        """
        ctd_points = self.gnd_to_ctd(points)
        return self.ctd_to_src(ctd_points)

    def get_key_point(self):
        """
        Get the key-point of the camera in world space.

        Returns:
            (3,) 3D coordinates of the key-point of the camera in world space
        """
        key_point = self._lambda_key_point()
        return np.array(key_point, dtype=np.float64).flatten()

    def ctd_to_ray(self, points):
        """
        Transform from corrected image coordinates to 3D rays in camera space.

        Args:
            points: (N, 2) array of corrected image points [[x1, y1], [x2, y2], ...]

        Returns:
            (N, 3) array of ray directions in camera space, with [nan, nan, nan] for invalid
        Ray directions are normalized to unit length.
        The ray for a point is defined as the vector from key-point of the camera.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = points.shape[0]
        x_im, y_im = points.T

        # Call lambdified function: returns shape (3, 1, N)
        # Reshape directly to (N, 3)
        rays = np.asarray(self._lambda_im2ray(x_im, y_im)).reshape(3, N).T

        # Normalize ray directions to unit length
        ray_lengths = np.linalg.norm(rays, axis=1, keepdims=True)
        ray_lengths = np.where(ray_lengths > 0, ray_lengths, 1.0)
        rays_normalized = rays / ray_lengths

        return rays_normalized

    def src_to_ray(self, points):
        """
        Transform from source image coordinates to 3D rays in camera space.

        Args:
            points: (N, 2) array of source points [[x1, y1], [x2, y2], ...]

        Returns:
            (N, 3) array of ray directions in camera space, with [nan, nan, nan] for invalid
        Ray directions are normalized to unit length.
        The ray for a point is defined as the vector from key-point of the camera.
        """
        ctd_points = self.src_to_ctd(points)
        return self.ctd_to_ray(ctd_points)

    @classmethod
    def load(cls, archive_path):
        """
        Load camera transformer from NPZ archive file.

        Args:
            archive_path: Path to .npz calibration file

        Returns:
            CameraProjectionVectorized instance
        """

        # Convert to regular dict for easier access
        cam_data = read_npz_file(archive_path)

        return cls(cam_data)
