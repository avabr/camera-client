# camera-client

Python SDK for camera calibration and projection transformations. Transform coordinates between distorted image space, corrected image space, and real-world 3D coordinates using pre-computed calibration data.

## Features

- **Vectorized operations** - Process multiple points simultaneously for high performance
- **Multiple coordinate systems** - Transform between source (distorted), corrected, and ground (3D world) coordinates
- **Lens distortion handling** - Correct for camera lens distortion using calibration lookup tables
- **Ground plane projection** - Project image coordinates to 3D world coordinates and vice versa
- **Ray casting** - Generate 3D rays from image coordinates for ray tracing and 3D reconstruction
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

# Download from JSON config (list of objects with "archive_url" key)
python -m camera_client get_camera_archive -f config.json -o ./archives

# Download only specific camera from JSON config
python -m camera_client get_camera_archive -f config.json --camera_id=66 -o ./archives
```

## Quick Start

```python
import numpy as np
from camera_client import CameraProjection

# Load camera calibration data from NPZ archive
camera = CameraProjection.load("camera_calibration.npz")

# Transform multiple points (vectorized operations)
# Note: All methods require (N, 2) or (N, 3) shaped arrays
source_points = np.array([
    [100, 200],
    [300, 400],
    [500, 600]
])  # Shape: (3, 2)

# Remove lens distortion
corrected_points = camera.src_to_ctd(source_points)

# Project to ground plane (height = 0)
ground_points = camera.src_to_gnd(source_points, h=0)
print(ground_points)  # Returns (N, 3) array with [x, y, z] coordinates
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

## Usage Examples

### Basic Coordinate Transformations

```python
from camera_client import CameraProjection
import numpy as np

# Load calibration data
camera = CameraProjection.load("camera_calibration.npz")

# Source (distorted) to Corrected (undistorted)
src_points = np.array([[640, 480], [1280, 720]])
ctd_points = camera.src_to_ctd(src_points)

# Corrected back to Source
src_points_back = camera.ctd_to_src(ctd_points)

# Check round-trip accuracy
error = np.linalg.norm(src_points - src_points_back, axis=1)
print(f"Round-trip error: {error}")
```

### 3D Ground Projection

```python
# Project image points to ground plane
src_points = np.array([[640, 480], [800, 600]])

# Project to ground at height = 0 (ground level)
ground_points = camera.src_to_gnd(src_points, h=0)
print(ground_points)  # Shape: (2, 3) with [x, y, z] coordinates

# Project to elevated plane (e.g., 1.5 meters above ground)
elevated_points = camera.src_to_gnd(src_points, h=1.5)

# Different height for each point
heights = np.array([0, 1.5])
mixed_points = camera.src_to_gnd(src_points, h=heights)
```

### Reverse Projection (3D to Image)

```python
# Project 3D world coordinates back to image
world_points = np.array([
    [10.0, 5.0, 0.0],    # x, y, z in meters
    [15.0, 8.0, 1.5]
])

# Get corrected image coordinates
ctd_points = camera.gnd_to_ctd(world_points)

# Get source (distorted) image coordinates
src_points = camera.gnd_to_src(world_points)
```

### Batch Processing

```python
# Process large batches of points efficiently
num_points = 10000
random_points = np.random.rand(num_points, 2) * [1920, 1080]

# Vectorized transformation (fast!)
corrected = camera.src_to_ctd(random_points)
ground = camera.src_to_gnd(random_points, h=0)
```

### Ray Casting (3D Reconstruction)

```python
# Get 3D rays from image points (useful for ray tracing, 3D reconstruction)
src_points = np.array([[640, 480], [800, 600]])

# Get ray directions from source (distorted) coordinates
rays = camera.src_to_ray(src_points)
print(rays.shape)  # (2, 3) - normalized direction vectors

# Or from corrected coordinates
ctd_points = camera.src_to_ctd(src_points)
rays = camera.ctd_to_ray(ctd_points)

# Get camera position in world space (ray origin)
key_point = camera.get_key_point()
print(key_point.shape)  # (3,) - [x, y, z] camera position

# Ray equation: point_on_ray = key_point + t * ray_direction
# All rays are normalized to unit length
```

### Accessing Camera Properties

```python
# Get camera dimensions
print(f"Image size: {camera.im_width} x {camera.im_height}")
print(f"Image WH: {camera.im_wh_size}")
print(f"Plan scale: {camera.plan_scale}")
```

## API Reference

### `CameraProjection.load(archive_path)`

Load camera calibration from NPZ file.

**Parameters:**
- `archive_path` (str): Path to .npz calibration archive

**Returns:**
- `CameraProjection` instance

---

### `src_to_ctd(points)`

Transform from source (distorted) to corrected coordinates.

**Parameters:**
- `points` (np.ndarray): Shape **(N, 2)** array of [x, y] coordinates

**Returns:**
- `np.ndarray`: Shape **(N, 2)** corrected coordinates

**Note:** Input must be 2D array. For single point use `np.array([[x, y]])`

---

### `ctd_to_src(points)`

Transform from corrected to source (distorted) coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 2) source coordinates

---

### `src_to_gnd(points, h)`

Transform from source coordinates to 3D ground coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates
- `h` (float or np.ndarray): Height(s) above ground. Scalar or shape (N,) array

**Returns:**
- `np.ndarray`: Shape (N, 3) ground coordinates [x, y, z]

---

### `gnd_to_src(points)`

Transform from 3D ground coordinates to source coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 3) array of [x, y, z] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 2) source coordinates

---

### `ctd_to_gnd(points, h)`

Transform from corrected coordinates to 3D ground coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates
- `h` (float or np.ndarray): Height(s) above ground

**Returns:**
- `np.ndarray`: Shape (N, 3) ground coordinates

---

### `gnd_to_ctd(points)`

Transform from 3D ground coordinates to corrected coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 3) array of [x, y, z] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 2) corrected coordinates

---

### `src_to_ray(points)`

Generate 3D ray directions from source (distorted) image coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 3) normalized ray direction vectors

**Note:** All rays originate from the camera key-point (use `get_key_point()`)

---

### `ctd_to_ray(points)`

Generate 3D ray directions from corrected (undistorted) image coordinates.

**Parameters:**
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 3) normalized ray direction vectors

---

### `get_key_point()`

Get the camera position (key-point) in world space.

**Returns:**
- `np.ndarray`: Shape (3,) array with [x, y, z] camera position

---

### `get_ctd_points_context(ctd_points)`

Get scale context values for corrected (CTD) image points.

**Parameters:**
- `ctd_points` (np.ndarray): Shape (N, 2) array of [x, y] corrected coordinates

**Returns:**
- `dict`: Dictionary with keys:
  - `wscale` (np.ndarray): Shape (N,) width scale values
  - `hscale` (np.ndarray): Shape (N,) height scale values
  - `vangle` (np.ndarray): Shape (N,) vertical angle values (radians)

**Note:** Out-of-bounds points will have NaN values

**Example:**
```python
ctd_points = np.array([[640, 480], [800, 600]])
context = camera.get_ctd_points_context(ctd_points)
print(context['wscale'])  # Width scale at each point
print(context['hscale'])  # Height scale at each point
print(context['vangle'])  # Vertical angle at each point
```

---

### `get_src_points_context(src_points)`

Get scale context values for source (distorted) image points.

**Parameters:**
- `src_points` (np.ndarray): Shape (N, 2) array of [x, y] source coordinates

**Returns:**
- `dict`: Dictionary with keys:
  - `wscale` (np.ndarray): Shape (N,) width scale values
  - `hscale` (np.ndarray): Shape (N,) height scale values
  - `vangle` (np.ndarray): Shape (N,) vertical angle values (radians)

**Note:** Internally converts source points to CTD coordinates first, then retrieves context

**Example:**
```python
src_points = np.array([[640, 480], [800, 600]])
context = camera.get_src_points_context(src_points)
print(context['wscale'])  # Width scale at each point
```

---

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

Alexander Abramov ([extremal.ru@gmail.com](mailto:extremal.ru@gmail.com))

## Upload PyPi

    rm dist/* && python -m build && python -m twine upload dist/*
    
