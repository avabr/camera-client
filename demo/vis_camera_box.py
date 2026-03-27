import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import requests
from io import BytesIO
from camera_client.loading import read_npz_file
from camera_client import CameraProjection


def interpolate_polygon(polygon, step_size=30):
    """
    Interpolate polygon edges with fixed step size.

    Args:
        polygon: List of [x, y] coordinate pairs
        step_size: Distance between interpolated points in pixels

    Returns:
        List of interpolated [x, y] points
    """
    interpolated_points = []
    polygon = np.array(polygon)

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]  # Wrap around to first point

        # Calculate edge length
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)

        # Number of steps for this edge
        num_steps = int(np.ceil(edge_length / step_size))

        # Interpolate points along this edge (excluding the last point to avoid duplicates)
        for j in range(num_steps):
            t = j / num_steps
            point = p1 + t * edge_vector
            interpolated_points.append(point.tolist())

    return interpolated_points


def line_polygon_intersections(line_start, line_end, polygon):
    """
    Find all intersections between a line segment and a polygon.

    Args:
        line_start: [x, y] starting point of the line
        line_end: [x, y] ending point of the line
        polygon: numpy array of shape (N, 2) with polygon vertices

    Returns:
        List of intersection points sorted by distance from line_start
    """
    intersections = []
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    for i in range(len(polygon)):
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]

        # Find intersection between line and polygon edge
        intersection = line_segment_intersection(
            line_start, line_end, edge_start, edge_end
        )
        if intersection is not None:
            intersections.append(intersection)

    # Sort intersections by distance from line_start
    if intersections:
        intersections = sorted(
            intersections, key=lambda p: np.linalg.norm(np.array(p) - line_start)
        )

    return intersections


def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point between two line segments p1-p2 and p3-p4.

    Returns:
        Intersection point [x, y] or None if no intersection
    """
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

    d1 = p2 - p1
    d2 = p4 - p3
    d3 = p3 - p1

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-10:
        return None

    t1 = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
    t2 = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return p1 + t1 * d1

    return None


def generate_grid_network(polygon_gnd, grid_step=1.0, interpolation_step=0.1):
    """
    Generate a grid network inside a polygon in ground coordinates.

    Args:
        polygon_gnd: numpy array of shape (N, 2) or (N, 3) with polygon vertices in ground coords
        grid_step: Grid spacing in meters (default 1.0)
        interpolation_step: Interpolation step along grid lines in meters (default 0.1)

    Returns:
        List of line segments, where each segment is a list of [x, y] points forming a continuous line
    """
    # Use only x, y coordinates (ignore z if present)
    polygon_xy = polygon_gnd[:, :2]

    # Find bounding box of polygon
    min_x, min_y = polygon_xy.min(axis=0)
    max_x, max_y = polygon_xy.max(axis=0)

    grid_lines = []

    # Generate vertical lines (constant X)
    x_values = np.arange(
        np.floor(min_x / grid_step) * grid_step,
        np.ceil(max_x / grid_step) * grid_step + grid_step,
        grid_step,
    )

    for x in x_values:
        line_start = [x, min_y - 1]
        line_end = [x, max_y + 1]
        intersections = line_polygon_intersections(line_start, line_end, polygon_xy)

        # Process pairs of intersections (entry and exit points)
        for i in range(0, len(intersections) - 1, 2):
            if i + 1 < len(intersections):
                p1 = np.array(intersections[i])
                p2 = np.array(intersections[i + 1])

                # Interpolate between intersection points
                segment_length = np.linalg.norm(p2 - p1)
                num_points = int(np.ceil(segment_length / interpolation_step))

                line_segment = []
                for j in range(num_points + 1):
                    t = j / max(num_points, 1)
                    point = p1 + t * (p2 - p1)
                    line_segment.append(point)

                grid_lines.append(line_segment)

    # Generate horizontal lines (constant Y)
    y_values = np.arange(
        np.floor(min_y / grid_step) * grid_step,
        np.ceil(max_y / grid_step) * grid_step + grid_step,
        grid_step,
    )

    for y in y_values:
        line_start = [min_x - 1, y]
        line_end = [max_x + 1, y]
        intersections = line_polygon_intersections(line_start, line_end, polygon_xy)

        # Process pairs of intersections (entry and exit points)
        for i in range(0, len(intersections) - 1, 2):
            if i + 1 < len(intersections):
                p1 = np.array(intersections[i])
                p2 = np.array(intersections[i + 1])

                # Interpolate between intersection points
                segment_length = np.linalg.norm(p2 - p1)
                num_points = int(np.ceil(segment_length / interpolation_step))

                line_segment = []
                for j in range(num_points + 1):
                    t = j / max(num_points, 1)
                    point = p1 + t * (p2 - p1)
                    line_segment.append(point)

                grid_lines.append(line_segment)

    return grid_lines


# Usage
camera_uuid = os.environ["TESTING_CAMERA_UUID"]
fname = f"camera_archives/camera_{camera_uuid}.npz"
data = read_npz_file(fname)
cp = CameraProjection.load(fname)


src_image_url = data["im_src_url"]
ctd_image_url = data["im_ctd_url"]

efov = cp.ctd_geometry["efov_polygon"]["coordinates"][0]
print(efov)

# Interpolate polygon with 30 pixel steps
efov_interpolated = interpolate_polygon(efov, step_size=30)
print(f"Original polygon points: {len(efov)}")
print(f"Interpolated points: {len(efov_interpolated)}")

# Project interpolated polygon to source image
efov_interpolated_array = np.array(efov_interpolated)
efov_src = cp.ctd_to_src(efov_interpolated_array)
# efov_src = cp.ctd_to_src(efov)
print(f"Projected to source: {efov_src.shape}")

# Generate 1-meter grid network inside the EFOV polygon
print("\nGenerating 1-meter grid network...")
efov_array = np.array(efov)
efov_gnd = cp.ctd_to_gnd(efov_array, h=0)
print(f"EFOV polygon in ground coordinates: {efov_gnd.shape}")

grid_lines_gnd = generate_grid_network(efov_gnd, grid_step=0.5, interpolation_step=0.1)
print(f"Generated {len(grid_lines_gnd)} grid lines in ground coords")

# Project grid lines to ctd and src coordinates
grid_lines_ctd = []
grid_lines_src = []

if len(grid_lines_gnd) > 0:
    for line_gnd in grid_lines_gnd:
        line_gnd_array = np.array(line_gnd)
        # Add z=0 coordinate to make it 3D for gnd_to_ctd
        line_gnd_3d = np.column_stack([line_gnd_array, np.zeros(len(line_gnd_array))])
        line_ctd = cp.gnd_to_ctd(line_gnd_3d)
        line_src = cp.gnd_to_src(line_gnd_3d)
        grid_lines_ctd.append(line_ctd)
        grid_lines_src.append(line_src)
    print(f"Projected {len(grid_lines_ctd)} grid lines to CTD and SRC coords")
else:
    print("Warning: No grid lines generated")

# Fetch images
response_ctd = requests.get(ctd_image_url)
img_ctd = Image.open(BytesIO(response_ctd.content))

response_src = requests.get(src_image_url)
img_src = Image.open(BytesIO(response_src.content))

# Create side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# CTD image with interpolated polygon
ax1.imshow(img_ctd)
polygon_ctd = Polygon(efov, fill=False, edgecolor="red", linewidth=2, label="Original")
ax1.add_patch(polygon_ctd)
ax1.scatter(
    efov_interpolated_array[:, 0],
    efov_interpolated_array[:, 1],
    c="blue",
    s=10,
    alpha=0.6,
    label="Interpolated points",
)
# Add grid network as lines
if len(grid_lines_ctd) > 0:
    for line in grid_lines_ctd:
        ax1.plot(line[:, 0], line[:, 1], c="green", linewidth=1.5, alpha=0.7)
    # Add a dummy line for legend
    ax1.plot([], [], c="green", linewidth=0.5, alpha=1.0, label="1m grid network")
ax1.set_title("CTD Image")
ax1.legend()
ax1.axis("off")

# Source image with projected polygon
ax2.imshow(img_src)
ax2.scatter(
    efov_src[:, 0],
    efov_src[:, 1],
    c="red",
    s=10,
    alpha=0.6,
    label="Projected points",
)
# Add grid network as lines
if len(grid_lines_src) > 0:
    for line in grid_lines_src:
        ax2.plot(line[:, 0], line[:, 1], c="green", linewidth=1.5, alpha=0.7)
    # Add a dummy line for legend
    ax2.plot([], [], c="green", linewidth=0.5, alpha=1.0, label="1m grid network")
ax2.set_title("Source Image")
ax2.legend()
ax2.axis("off")

plt.tight_layout()
plt.show()
