import os
from camera_client.loading import read_npz_file
import matplotlib.pyplot as plt
import numpy as np

# Usage
camera_uuid = os.environ["TESTING_CAMERA_UUID"]
fname = f"camera_archives/camera_{camera_uuid}.npz"

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
