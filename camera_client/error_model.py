import numpy as np


CALIBRATION_ERROR_LAYER_KEY = "camera_calibration_error_layer"



def _get_error_layer(camera):
    """Extract calibration error layer from camera, raising on missing/null."""
    layer = camera.camera_layers.get(CALIBRATION_ERROR_LAYER_KEY)
    if layer is None:
        raise ValueError(
            f"Camera {camera.camera_id}: "
            f"'{CALIBRATION_ERROR_LAYER_KEY}' layer is missing or null. "
            f"Calibration error parameters must be configured in the camera archive."
        )
    return layer


class DistortionError:
    """Distortion correction error model.

    sigma_dist(r) = sigma_max * (r / r_max) ^ beta

    where r is the radial distance from image center in normalized coordinates
    (fraction of image width), sigma is returned in the same units.
    """

    def __init__(self, camera, sigma_max, beta):
        self.camera = camera
        self.sigma_max = sigma_max
        self.beta = beta

        self.w = camera.im_width
        self.h = camera.im_height
        self.cx = 0.5
        self.cy = (self.h / self.w) / 2.0
        self.r_max = np.sqrt(self.cx ** 2 + self.cy ** 2)

    @classmethod
    def from_camera(cls, camera):
        """Create from camera's calibration error layer."""
        layer = _get_error_layer(camera)
        return cls(camera, sigma_max=layer["dist_sigma_max"], beta=layer["dist_beta"])

    def get_ctd_sigma(self, x, y):
        """Sigma at corrected image point (x, y), in fraction of image width."""
        xn = x / self.w
        yn = y / self.w
        r = np.sqrt((xn - self.cx) ** 2 + (yn - self.cy) ** 2)
        return self.sigma_max * (r / self.r_max) ** self.beta

    def get_ctd_sigma_px(self, x, y):
        """Sigma at corrected image point (x, y), in pixels."""
        return self.get_ctd_sigma(x, y) * self.w

    def get_src_sigma(self, x, y):
        """Sigma at source (distorted) image point (x, y), in fraction of image width."""
        ctd_xy = self.camera.src_to_ctd(np.array([[x, y]]))[0]
        return self.get_ctd_sigma(float(ctd_xy[0]), float(ctd_xy[1]))


class SceneCalibError:
    """Geometric calibration error model.

    sigma_geom(r) = sigma_center + (sigma_edge - sigma_center) * (r / r_max) ^ gamma

    where r is the radial distance from image center in normalized coordinates.
    """

    def __init__(self, camera, sigma_center, sigma_edge, gamma):
        self.camera = camera
        self.sigma_center = sigma_center
        self.sigma_edge = sigma_edge
        self.gamma = gamma

        self.w = camera.im_width
        self.h = camera.im_height
        self.cx = 0.5
        self.cy = (self.h / self.w) / 2.0
        self.r_max = np.sqrt(self.cx ** 2 + self.cy ** 2)

    @classmethod
    def from_camera(cls, camera):
        """Create from camera's calibration error layer."""
        layer = _get_error_layer(camera)
        return cls(
            camera,
            sigma_center=layer["geom_sigma_center"],
            sigma_edge=layer["geom_sigma_edge"],
            gamma=layer["geom_gamma"],
        )

    def get_ctd_sigma(self, x, y):
        """Sigma at corrected image point (x, y), in fraction of image width."""
        xn = x / self.w
        yn = y / self.w
        r = np.sqrt((xn - self.cx) ** 2 + (yn - self.cy) ** 2)
        return self.sigma_center + (self.sigma_edge - self.sigma_center) * (r / self.r_max) ** self.gamma

    def get_ctd_sigma_px(self, x, y):
        """Sigma at corrected image point (x, y), in pixels."""
        return self.get_ctd_sigma(x, y) * self.w

    def get_src_sigma(self, x, y):
        """Sigma at source (distorted) image point (x, y), in fraction of image width."""
        ctd_xy = self.camera.src_to_ctd(np.array([[x, y]]))[0]
        return self.get_ctd_sigma(float(ctd_xy[0]), float(ctd_xy[1]))


class TotalImageCov:
    """Combined pixel-level uncertainty.

    sigma_total^2 = sigma_dist^2 + sigma_calib^2 + sigma_detection^2
    """

    def __init__(self, distortion_error, scene_calib_error):
        self.dist = distortion_error
        self.calib = scene_calib_error
        self.w = self.dist.w
        self.h = self.dist.h

    @classmethod
    def from_camera(cls, camera):
        """Create from camera's calibration error layer."""
        return cls(
            DistortionError.from_camera(camera),
            SceneCalibError.from_camera(camera),
        )

    def get_ctd_sigma(self, x, y, detection_sigma=0.0):
        """Combined sigma at corrected image point (x, y), in fraction of image width."""
        s_dist = self.dist.get_ctd_sigma(x, y)
        s_calib = self.calib.get_ctd_sigma(x, y)
        return np.sqrt(s_dist ** 2 + s_calib ** 2 + detection_sigma ** 2)

    def get_ctd_sigma_px(self, x, y, detection_sigma=0.0):
        """Combined sigma at corrected image point (x, y), in pixels."""
        return self.get_ctd_sigma(x, y, detection_sigma) * self.w


def make_grid(sigma_fn, w, h, nx=19, ny=12):
    """Build a grid of sigma values over image of size w x h.

    Args:
        sigma_fn: callable(x, y) returning sigma (in fraction of image width)
        w: image width in pixels
        h: image height in pixels
        nx: number of grid steps along x
        ny: number of grid steps along y

    Returns:
        list of dicts with keys: x, y, sigma, sigma_px
    """
    step_x = w / nx
    step_y = h / ny
    grid = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = ix * step_x
            y = iy * step_y
            sigma = sigma_fn(x, y)
            grid.append({
                "x": round(x, 1),
                "y": round(y, 1),
                "sigma": round(sigma, 6),
                "sigma_px": round(sigma * w, 2),
            })
    return grid


class CameraSpatialCovariance:
    """Spatial covariance for a single camera.

    For a 3D point P, computes the 3x3 covariance matrix:

        Sigma_cam(P) = Sigma_perp(P) + Sigma_par(P) + Sigma_bind

    where:
        Sigma_perp = d^2 * J * Sigma_pix * J^T   (transverse, from pixel uncertainty)
        Sigma_par  = d^2 * e_ray * e_ray^T        (longitudinal, bearing-only)
        Sigma_bind = sigma_binding^2 * I_3         (spatial binding uncertainty)

        d = |P - key_point|
        J = d(e_ray) / d(x_im, y_im)              (ray direction Jacobian)
    """

    def __init__(self, camera):
        self.camera = camera
        self.total_image_cov = TotalImageCov.from_camera(camera)
        self.key_point = np.array(camera.get_key_point(), dtype=np.float64).flatten()

    def get_covariance(self, p, detection_sigma=0.0, sigma_binding=0.0):
        """Full 3x3 covariance matrix Sigma_cam(P) for a 3D point p=[x,y,z].

        Args:
            p: 3D point [x, y, z]
            detection_sigma: detection uncertainty (fraction of image width)
            sigma_binding: spatial binding uncertainty in meters
        """
        p = np.asarray(p, dtype=np.float64)
        cam = self.camera

        ctd = cam.gnd_to_ctd(p.reshape(1, 3))[0]
        x_ctd, y_ctd = float(ctd[0]), float(ctd[1])

        d = np.linalg.norm(p - self.key_point)
        e_ray = cam.ctd_to_ray(np.array([[x_ctd, y_ctd]]))[0]
        J = cam.ctd_to_ray_jacobian(x_ctd, y_ctd)

        # Transverse: d^2 * J * Sigma_pix * J^T (Sigma_pix = sigma_px^2 * I_2)
        sigma_px = self.total_image_cov.get_ctd_sigma_px(x_ctd, y_ctd, detection_sigma)
        Sigma_perp = (d * sigma_px) ** 2 * (J @ J.T)

        # Longitudinal: d^2 * e_ray * e_ray^T
        Sigma_par = d ** 2 * np.outer(e_ray, e_ray)

        # Binding: sigma_binding^2 * I_3
        Sigma_bind = sigma_binding ** 2 * np.eye(3)

        return Sigma_perp + Sigma_par + Sigma_bind
