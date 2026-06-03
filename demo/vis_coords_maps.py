import os, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera_client.loading import read_npz_file
import matplotlib.pyplot as plt
import numpy as np


def resolve_archive_path(arg):
    """Resolve archive path from a full path or camera_id integer."""
    if arg.isdigit():
        archives_dir = os.path.join(os.path.dirname(__file__), "..", "camera_archives")
        matches = glob.glob(os.path.join(archives_dir, f"camera_{arg}_*.npz"))
        if not matches:
            print(f"No archive found for camera_id={arg} in {archives_dir}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple archives found for camera_id={arg}, using first: {matches[0]}")
        return matches[0]
    return arg


# Usage
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <archive.npz | camera_id>")
    sys.exit(1)

fname = resolve_archive_path(sys.argv[1])

data = read_npz_file(fname)

# Visualize all maps
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("Camera Projection Maps Visualization", fontsize=16)

# src2ctd - undistorted (corrected) image coordinates
im0 = axes[0, 0].imshow(data["src2ctd"][:, :, 0], cmap="viridis")
axes[0, 0].set_title("src2ctd - X Channel (Undistorted)")
axes[0, 0].axis("off")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(data["src2ctd"][:, :, 1], cmap="viridis")
axes[0, 1].set_title("src2ctd - Y Channel (Undistorted)")
axes[0, 1].axis("off")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# ctd2src - distorted (raw) image coordinates
im2 = axes[0, 2].imshow(data["ctd2src"][:, :, 0], cmap="plasma")
axes[0, 2].set_title("ctd2src - X Channel (Distorted)")
axes[0, 2].axis("off")
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

im3 = axes[0, 3].imshow(data["ctd2src"][:, :, 1], cmap="plasma")
axes[0, 3].set_title("ctd2src - Y Channel (Distorted)")
axes[0, 3].axis("off")
plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

# Scalar maps
im4 = axes[1, 0].imshow(data["map_scale_h"], cmap="coolwarm")
axes[1, 0].set_title("Height Scale Map")
axes[1, 0].axis("off")
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

im5 = axes[1, 1].imshow(data["map_scale_w"], cmap="coolwarm")
axes[1, 1].set_title("Width Scale Map")
axes[1, 1].axis("off")
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

im6 = axes[1, 2].imshow(data["map_scale_vang"], cmap="coolwarm")
axes[1, 2].set_title("Vertical Angle Map")
axes[1, 2].axis("off")
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

# Hide unused subplot
axes[1, 3].axis("off")

plt.tight_layout()
plt.show()
