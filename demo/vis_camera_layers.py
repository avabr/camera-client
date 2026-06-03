"""Visualize camera layers from a calibration archive."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import urllib.request
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from camera_client import CameraProjection


def load_image_from_url(url):
    with urllib.request.urlopen(url) as resp:
        return np.array(Image.open(BytesIO(resp.read())))


def resolve_archive_path(arg):
    """Resolve archive path from a full path or camera_id integer."""
    if arg.isdigit():
        archives_dir = os.path.join(os.path.dirname(__file__), "..", "camera_archives")
        import glob
        matches = glob.glob(os.path.join(archives_dir, f"camera_{arg}_*.npz"))
        if not matches:
            print(f"No archive found for camera_id={arg} in {archives_dir}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple archives found for camera_id={arg}, using first: {matches[0]}")
        return matches[0]
    return arg


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <archive.npz | camera_id>")
        sys.exit(1)

    cam = CameraProjection.load(resolve_archive_path(sys.argv[1]))

    src_img = load_image_from_url(cam.im_src_url)
    ctd_img = load_image_from_url(cam.im_ctd_url)

    fig, (ax_src, ax_ctd) = plt.subplots(1, 2, figsize=(18, 6))

    # Source image with subframes
    ax_src.imshow(src_img)
    ax_src.set_title("Source (distorted)")

    subframes = cam.camera_layers.get("camera_subframes_layer", [])
    for sf in subframes:
        tl = sf["top_left"]["coordinates"]
        br = sf["bottom_right"]["coordinates"]
        x, y = tl
        w, h = br[0] - tl[0], br[1] - tl[1]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax_src.add_patch(rect)
        ax_src.text(
            x, y - 4, sf.get("marker", ""),
            color="lime", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    blind_areas = cam.camera_layers.get("camera_blind_areas_layer", [])
    for ba in blind_areas:
        coords = np.array(ba["polygon"]["coordinates"])
        poly = patches.Polygon(
            coords, closed=True,
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.25,
        )
        ax_src.add_patch(poly)
        ax_src.text(
            coords[0, 0], coords[0, 1] - 4, ba.get("marker", ""),
            color="red", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    # Corrected image with counting lines
    ax_ctd.imshow(ctd_img)
    ax_ctd.set_title("Corrected (undistorted)")

    counting_lines = cam.camera_layers.get("camera_counting_lines_layer", [])
    colors = plt.cm.tab10.colors
    for i, cl in enumerate(counting_lines):
        coords = np.array(cl["pass_line"]["coordinates"])
        color = colors[i % len(colors)]
        ax_ctd.plot(coords[:, 0], coords[:, 1], linewidth=2, color=color)
        ax_ctd.text(
            coords[0, 0], coords[0, 1] - 4, cl.get("name", ""),
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    for ax in (ax_src, ax_ctd):
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
