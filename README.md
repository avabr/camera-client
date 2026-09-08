# camera-client

Python SDK for camera calibration, projection transformations, and multi-camera spatial uncertainty analysis. Transform coordinates between distorted image space, corrected image space, and real-world 3D coordinates using pre-computed calibration data. Quantify measurement uncertainty and fuse observations from multiple cameras.

## Features

- **Vectorized operations** - Process multiple points simultaneously for high performance
- **Multiple coordinate systems** - Transform between source (distorted), corrected, and ground (3D world) coordinates
- **Lens distortion handling** - Correct for camera lens distortion using calibration lookup tables
- **Ground plane projection** - Project image coordinates to 3D world coordinates and vice versa
- **Ray casting** - Generate 3D rays from image coordinates for ray tracing and 3D reconstruction
- **Calibration error models** - Quantify pixel-level uncertainty from distortion correction and geometric calibration
- **3D spatial covariance** - Propagate pixel uncertainty to full 3x3 covariance matrices in world space
- **Multi-camera fusion** - Combine measurements from multiple cameras via information fusion
- **Triangulation** - Recover 3D positions from multi-camera observations with consistency checking
- **Sympy-based transformations** - Fast compiled symbolic expressions for mathematical transformations
- **NumPy-based** - Fast array operations with minimal dependencies

## Installation

Install from PyPI:

```bash
pip install camera-client
```

## CLI Usage

Download camera calibration archives from URL:

```bash
# Download single archive
python -m camera_client get_camera_archive https://example.com/camera.npz

# Download from file with URLs (one per line, non-URL lines ignored)
python -m camera_client get_camera_archive -f urls.txt -o ./archives

# Download from JSON config (list of objects with "archive_url" or "camera_uuid" key)
python -m camera_client get_camera_archive -f config.json -o ./archives

# Download only specific camera from JSON config
python -m camera_client get_camera_archive -f config.json --camera_id=66 -o ./archives
```

JSON config entries may use `"archive_url"` (direct link) or `"camera_uuid"` (requires
`CAMERA_SERVICE_ENTRYPOINT` env variable — the URL is constructed as
`<CAMERA_SERVICE_ENTRYPOINT>/processing_api/projection_npz_archive/<camera_uuid>`).

## Quick Start (Projection)

```python
import numpy as np
from camera_client import CameraProjection

# Load camera calibration data from NPZ archive
camera = CameraProjection.load("camera_calibration_archive.npz")

# All methods work with (N, 2) or (N, 3) shaped arrays
source_points = np.array([
    [100, 200],
    [300, 400],
    [500, 600]
])  # Shape: (3, 2)

# ── Forward: image → world ──

# Project to ground plane: src → gnd (at height = 0)
ground_points = camera.src_to_gnd(source_points, h=0)
print(ground_points)  # (N, 3) array with [x, y, z] coordinates

# Remove lens distortion only: src → ctd
ctd_points = camera.src_to_ctd(source_points)

# ── Reverse: world → image ──

# Project 3D points back to distorted image coordinates
world_points = np.array([[10.0, 5.0, 0.0], [15.0, 8.0, 1.5]])
src_points = camera.gnd_to_src(world_points)   # gnd → src
ctd_points = camera.gnd_to_ctd(world_points)   # gnd → ctd

# Corrected back to distorted
src_from_ctd = camera.ctd_to_src(ctd_points)   # ctd → src
```

## Quick Start (Measurements)

Multi-camera uncertainty analysis: error models, covariance fusion, and triangulation.

```python
import numpy as np
from camera_client import CameraProjection, CameraNetwork, triangulation

# Load cameras and create a network
cameras = [
    CameraProjection.load("camera_1.npz"),
    CameraProjection.load("camera_2.npz"),
    CameraProjection.load("camera_3.npz"),
]
net = CameraNetwork(cameras)
# net.cameras      — dict {camera_id: CameraProjection}
# net.covariances  — dict {camera_id: CameraSpatialCovariance}
```

### Spatial covariance

`get_covariance` computes 3x3 spatial covariance matrices for 3D points.
The matrix encodes how pixel-level uncertainty (distortion + geometric calibration + detection)
propagates into world-space uncertainty through the camera's ray geometry.

```python
points = np.array([
    [15.0, 5.0, 0.0],
    [18.0, 6.0, 1.5],
])

# Per-camera covariance (no fusion, no visibility check)
covs_cam = net.get_covariance(points, camera_id=1177, detection_sigma=0.01)
# covs_cam[i] is a (3, 3) covariance matrix from camera 1177

# Fused covariance from all visible cameras (information fusion)
covs_fused = net.get_covariance(points, detection_sigma=0.01)
# covs_fused[i] is (3, 3) fused covariance, or None if not visible to any camera

for i, cov in enumerate(covs_fused):
    if cov is not None:
        stds = np.sqrt(np.linalg.eigvalsh(cov))
        print(f"Point {i}: σ = {stds[0]:.3f}m, {stds[1]:.3f}m, {stds[2]:.3f}m")
```

### Triangulation

Recover a 3D point from pixel observations in multiple cameras.
Returns the fused position, a priori covariance (from error models),
and a posteriori covariance (inflated if cameras are inconsistent).

```python
# Observations: {camera_id: src_point} — source (distorted) image coordinates
observations = {
    1177: np.array([946.9, 853.1]),
    1178: np.array([956.0, 765.0]),
}

result = net.triangulate(observations, detection_sigma=0.01)
p_fused, sigma_prior, sigma_posterior = result

print(f"Position: {p_fused}")
print(f"Consistent: {sigma_prior is sigma_posterior}")  # True if within n_sigma

# Adjust consistency threshold (default n_sigma=3.0)
result = net.triangulate(observations, n_sigma=2.0)
```

### Mahalanobis distance

Check statistical consistency between two point estimates with their covariances.

```python
d2 = triangulation.mahalanobis_distance(p1, cov1, p2, cov2)
# d2 is squared Mahalanobis distance; compare to chi-squared thresholds
# e.g. chi2(3 dof, 99%) ≈ 11.34
```

## Coordinate Systems

This library handles transformations between three coordinate systems:

- **src** (Source): Distorted image coordinates from the camera
- **ctd** (Corrected): Undistorted image coordinates after lens correction
- **gnd** (Ground): Real-world 3D coordinates (x, y, z)

```
     Distorted                Undistorted              World 3D
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│   Source (src)  │ <──> │ Corrected (ctd) │ <──> │  Ground (gnd)   │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
   Lens distortion        Lens correction         3D projection
```

## API Reference

### `CameraProjection`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `load(archive_path)` | str | `CameraProjection` | Load calibration from .npz file |
| `src_to_ctd(points)` | (N, 2) | (N, 2) | Source → corrected (undistort) |
| `ctd_to_src(points)` | (N, 2) | (N, 2) | Corrected → source (redistort) |
| `src_to_gnd(points, h)` | (N, 2), scalar/array | (N, 3) | Source → 3D ground at height h |
| `gnd_to_src(points)` | (N, 3) | (N, 2) | 3D ground → source |
| `ctd_to_gnd(points, h)` | (N, 2), scalar/array | (N, 3) | Corrected → 3D ground at height h |
| `gnd_to_ctd(points)` | (N, 3) | (N, 2) | 3D ground → corrected |
| `src_to_ray(points)` | (N, 2) | (N, 3) | Source → normalized ray directions |
| `ctd_to_ray(points)` | (N, 2) | (N, 3) | Corrected → normalized ray directions |
| `ctd_to_ray_jacobian(x, y)` | scalar, scalar | (3, 2) | Ray direction Jacobian at CTD point |
| `get_key_point()` | — | (3,) | Camera position in world space |
| `get_ctd_points_context(points)` | (N, 2) | dict | Scale context (wscale, hscale, vangle) at CTD points |
| `get_src_points_context(points)` | (N, 2) | dict | Scale context at source points (converts to CTD internally) |

All point methods expect 2D arrays. For a single point: `np.array([[x, y]])`.
Out-of-bounds points return NaN. Height `h` can be a scalar or per-point (N,) array.

### `CameraNetwork`

| Method | Description |
|--------|-------------|
| `CameraNetwork(cameras)` | Create network from list of `CameraProjection` instances |
| `get_covariance(points, ...)` | Fused 3x3 covariance for (N, 3) points from all visible cameras |
| `get_covariance(points, camera_id=id, ...)` | Per-camera 3x3 covariance (no fusion, no visibility check) |
| `triangulate(observations, ...)` | 3D point + covariance from `{camera_id: src_point}` observations |

Common parameters: `detection_sigma` (float), `sigma_binding` (float), `use_efov` (bool), `n_sigma` (float).

### `triangulation` module

| Function | Description |
|----------|-------------|
| `fuse_covariances(covs)` | (Σ₁⁻¹ + ... + Σₙ⁻¹)⁻¹ |
| `information_fusion(points, covs)` | Fused point + covariance |
| `mahalanobis_distance(p1, cov1, p2, cov2)` | Squared Mahalanobis distance |
| `least_squares_intersection(rays)` | Closest point to N rays (SVD) |
| `closest_point_on_ray(origin, dir, point)` | Project point onto ray |

## Calibration File Format

The calibration file is a NumPy `.npz` archive containing:

### Lookup Tables
- `src2ctd`: Source to corrected coordinate map (H x W x 2)
- `ctd2src`: Corrected to source coordinate map (H x W x 2)
- `map_scale_h`: Height scale values (H x W)
- `map_scale_w`: Width scale values (H x W)
- `map_scale_vang`: Vertical angle values (H x W)

### Symbolic Expressions (stored as strings, parsed with SymPy)
- `exp_im2gnd`: Image to ground coordinate transformation
- `exp_gnd2im`: Ground to image coordinate transformation
- `exp_key_point`: Camera key-point (position) in world space
- `exp_im2ray`: Image to ray direction transformation

### Metadata
- `format_version`: Version string of the data format
- `camera_id`: Integer identifier for the camera
- `plan_url`: URL or path to the ground plan image
- `plan_scale`: Scale factor for ground plane coordinates (pixels per meter)
- `plan_width`: Width of the ground plan in pixels
- `plan_height`: Height of the ground plan in pixels
- `im_src_url`: URL or path to the source (distorted) camera image
- `im_ctd_url`: URL or path to the corrected (undistorted) camera image
- `im_width`: Width of the camera image in pixels
- `im_height`: Height of the camera image in pixels
- `ctd_geometry`: JSON object with geometry data in CTD coordinates (efov_polygon, counting_lines)

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SymPy >= 1.10.0

## Links

- **Repository**: [https://github.com/avabr/camera-client](https://github.com/avabr/camera-client)
- **Issues**: [https://github.com/avabr/camera-client/issues](https://github.com/avabr/camera-client/issues)
- **PyPI**: [https://pypi.org/project/camera-client/](https://pypi.org/project/camera-client/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Alexander V. Abramov ([avabr.me@gmail.com](mailto:avabr.me@gmail.com))

## Upload PyPi

    rm dist/* && python -m build && python -m twine upload dist/*

