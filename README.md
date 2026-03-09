# camera-client

Python SDK for camera calibration and projection transformations. Transform coordinates between distorted image space, corrected image space, and real-world 3D coordinates using pre-computed calibration data.

## Features

- **Vectorized operations** - Process multiple points simultaneously for high performance
- **Multiple coordinate systems** - Transform between source (distorted), corrected, and ground (3D world) coordinates
- **Lens distortion handling** - Correct for camera lens distortion using calibration lookup tables
- **Ground plane projection** - Project image coordinates to 3D world coordinates and vice versa
- **NumPy-based** - Fast array operations with minimal dependencies

## Installation

Install from PyPI:

```bash
pip install camera-client
```

## Quick Start

```python
import numpy as np
from camera_client import CameraProjection

# Load camera calibration data from NPZ archive
camera = CameraProjection.load("camera_calibration.npz")

# Transform single or multiple points (vectorized operations)
source_points = np.array([
    [100, 200],
    [300, 400],
    [500, 600]
])

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
- `points` (np.ndarray): Shape (N, 2) array of [x, y] coordinates

**Returns:**
- `np.ndarray`: Shape (N, 2) corrected coordinates

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

## Calibration File Format

The calibration file is a NumPy `.npz` archive containing:

- `src2ctd`: Lookup table for source to corrected transformation (H x W x 2)
- `ctd2src`: Lookup table for corrected to source transformation (H x W x 2)
- `x_gnd`, `y_gnd`, `z_gnd`: Expression strings for corrected to ground transformation
- `x_im`, `y_im`: Expression strings for ground to corrected transformation
- `im_width`, `im_height`: Image dimensions
- `plan_scale`: Scale factor for ground plane

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0

## Links

- **Repository**: [https://github.com/avabr/camera-client](https://github.com/avabr/camera-client)
- **Issues**: [https://github.com/avabr/camera-client/issues](https://github.com/avabr/camera-client/issues)
- **PyPI**: [https://pypi.org/project/camera-client/](https://pypi.org/project/camera-client/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Alexander Abramov ([extremal.ru@gmail.com](mailto:extremal.ru@gmail.com))
