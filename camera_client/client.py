import numpy as np
from camera_client.utils import compile_safe_expression
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

    # Allowed variable names for different transformation types
    VARS_CTD_TO_GND = {"x_im", "y_im", "proj_height", "np", "float64", "asarray", "dtype"}
    VARS_GND_TO_CTD = {"x_gnd", "y_gnd", "z_gnd", "np", "float64", "asarray", "dtype"}
    VARS_CTD_TO_RAY = {"x_im", "y_im", "np", "float64", "asarray", "dtype"}
    VARS_KEYPOINT = {"np", "float64", "asarray", "dtype"}

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

        self.plan_scale = data["plan_scale"]
        self.im_width = data["im_width"]
        self.im_height = data["im_height"]
        self.im_wh_size = (self.im_width, self.im_height)

        # Store lookup tables
        self.src2ctd_points_map = data["src2ctd"]
        self.ctd2src_points_map = data["ctd2src"]

        self.im_size = self.src2ctd_points_map.shape[:2]

        # Compile transformation expressions for ctd -> gnd
        self._x_gnd_func = compile_safe_expression(
            data["x_gnd"],
            param_names=["x_im", "y_im", "proj_height"],
            allowed_vars=self.VARS_CTD_TO_GND,
        )
        self._y_gnd_func = compile_safe_expression(
            data["y_gnd"],
            param_names=["x_im", "y_im", "proj_height"],
            allowed_vars=self.VARS_CTD_TO_GND,
        )
        self._z_gnd_func = compile_safe_expression(
            data["z_gnd"],
            param_names=["x_im", "y_im", "proj_height"],
            allowed_vars=self.VARS_CTD_TO_GND,
        )

        # Compile transformation expressions for gnd -> ctd
        self._x_im_func = compile_safe_expression(
            data["x_im"],
            param_names=["x_gnd", "y_gnd", "z_gnd"],
            allowed_vars=self.VARS_GND_TO_CTD,
        )
        self._y_im_func = compile_safe_expression(
            data["y_im"],
            param_names=["x_gnd", "y_gnd", "z_gnd"],
            allowed_vars=self.VARS_GND_TO_CTD,
        )

        # Compile keypoint expressions (camera position in world space)
        self._x_key_func = compile_safe_expression(
            data["x_key"],
            param_names=[],
            allowed_vars=self.VARS_KEYPOINT,
        )
        self._y_key_func = compile_safe_expression(
            data["y_key"],
            param_names=[],
            allowed_vars=self.VARS_KEYPOINT,
        )
        self._z_key_func = compile_safe_expression(
            data["z_key"],
            param_names=[],
            allowed_vars=self.VARS_KEYPOINT,
        )

        # Compile ray direction expressions for ctd -> ray
        self._x_ray_func = compile_safe_expression(
            data["x_ray"],
            param_names=["x_im", "y_im"],
            allowed_vars=self.VARS_CTD_TO_RAY,
        )
        self._y_ray_func = compile_safe_expression(
            data["y_ray"],
            param_names=["x_im", "y_im"],
            allowed_vars=self.VARS_CTD_TO_RAY,
        )
        self._z_ray_func = compile_safe_expression(
            data["z_ray"],
            param_names=["x_im", "y_im"],
            allowed_vars=self.VARS_CTD_TO_RAY,
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
            (N, 3) array of ground points [[x1, y1, z1], ...], with [nan, nan, nan] for invalid
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = len(points)
        result = np.full((N, 3), np.nan, dtype=float)

        # Handle scalar or array height
        if np.isscalar(h):
            h_array = np.full(N, h, dtype=float)
        else:
            h_array = np.asarray(h, dtype=float)
            if h_array.shape != (N,):
                raise ValueError(
                    f"Height array must have shape ({N},), got {h_array.shape}"
                )

        # Check for NaN input points
        valid = ~np.isnan(points).any(axis=1)

        if not valid.any():
            return result

        # Extract valid points
        valid_points = points[valid]
        valid_heights = h_array[valid]

        # Vectorized expression evaluation
        # NumPy operations in expressions will work element-wise on arrays
        x_im = valid_points[:, 0]
        y_im = valid_points[:, 1]
        proj_height = valid_heights

        # Evaluate expressions (they should handle arrays automatically via NumPy)
        gnd_x = self._x_gnd_func(x_im, y_im, proj_height)
        gnd_y = self._y_gnd_func(x_im, y_im, proj_height)
        gnd_z = self._z_gnd_func(x_im, y_im, proj_height)

        # Stack results and assign to valid indices
        result[valid] = np.column_stack([gnd_x, gnd_y, gnd_z])

        return result

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

    def gnd_to_ctd(self, points):
        """
        Transform from ground (3D world) to corrected image coordinates.

        Args:
            points: (N, 3) array of ground points [[x1, y1, z1], [x2, y2, z2], ...]

        Returns:
            (N, 2) array of image points, with [nan, nan] for invalid
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {points.shape}")

        N = len(points)
        result = np.full((N, 2), np.nan, dtype=float)

        # Check for NaN input points
        valid = ~np.isnan(points).any(axis=1)

        if not valid.any():
            return result

        # Extract valid points
        valid_points = points[valid]

        # Vectorized expression evaluation
        x_gnd = valid_points[:, 0]
        y_gnd = valid_points[:, 1]
        z_gnd = valid_points[:, 2]

        # Evaluate expressions
        x_im = self._x_im_func(x_gnd, y_gnd, z_gnd)
        y_im = self._y_im_func(x_gnd, y_gnd, z_gnd)

        # Stack results and assign to valid indices
        result[valid] = np.column_stack([x_im, y_im])

        return result

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

    def ctd_to_ray(self, points):
        """
        Transform from corrected image coordinates to 3D rays in camera space.

        Args:
            points: (N, 2) array of corrected image points [[x1, y1], [x2, y2], ...]

        Returns two argiments:
            - (3,) 3D coordinates of the key-point of the camera in world space
            - (N, 3) array of ray directions in camera space, with [nan, nan, nan] for invalid
        Ray directions are normalized to unit length.
        The ray for a point is defined as the vector from key-point of the camera.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) array, got shape {points.shape}")

        N = len(points)
        ray_directions = np.full((N, 3), np.nan, dtype=float)

        # Calculate keypoint (camera position in world space)
        # These are typically constants, so we call with no arguments
        keypoint = np.array([
            self._x_key_func(),
            self._y_key_func(),
            self._z_key_func()
        ], dtype=float)

        # Check for valid points (not NaN)
        valid = ~np.isnan(points).any(axis=1)

        if not valid.any():
            return keypoint, ray_directions

        # Extract valid corrected points
        valid_points = points[valid]

        # Vectorized ray direction evaluation
        x_im = valid_points[:, 0]
        y_im = valid_points[:, 1]

        # Evaluate ray direction expressions
        ray_x = self._x_ray_func(x_im, y_im)
        ray_y = self._y_ray_func(x_im, y_im)
        ray_z = self._z_ray_func(x_im, y_im)

        # Stack ray components and ensure float64 dtype
        # (expressions with large integers can create object dtype arrays)
        rays = np.column_stack([ray_x, ray_y, ray_z]).astype(np.float64)

        # Normalize ray directions to unit length
        ray_lengths = np.linalg.norm(rays, axis=1, keepdims=True)
        # Avoid division by zero
        ray_lengths = np.where(ray_lengths > 0, ray_lengths, 1.0)
        rays_normalized = rays / ray_lengths

        # Assign normalized rays to valid indices
        ray_directions[valid] = rays_normalized

        return keypoint, ray_directions

    def src_to_ray(self, points):
        """
        Transform from source image coordinates to 3D rays in camera space.

        Args:
            points: (N, 2) array of source points [[x1, y1], [x2, y2], ...]

        Returns two argiments:
            - (3,) 3D coordinates of the key-point of the camera in world space
            - (N, 3) array of ray directions in camera space, with [nan, nan, nan] for invalid
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
