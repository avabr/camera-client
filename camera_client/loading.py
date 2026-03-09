import numpy as np


# Method 1: Read from a saved file
def read_npz_file(filename):
    """Read camera projection data from saved .npz file.

    Args:
        filename (str): Path to the .npz file containing camera projection data

    Returns:
        dict: Dictionary containing the following keys:
            - plan_scale (float): Scale factor for ground plane coordinates (pixels per meter)
            - im_width (int): Width of the image in pixels
            - im_height (int): Height of the image in pixels
            - src2ctd (np.ndarray): Map of coordinates for undistorted (corrected) image.
                                   Shape: (height, width, 2) where channel 0 is X, channel 1 is Y
            - ctd2src (np.ndarray): Map of coordinates for distorted (raw) image based on undistorted.
                                   Shape: (height, width, 2) where channel 0 is X, channel 1 is Y
            - map_scale_h (np.ndarray): Scalar map of height scale values. Shape: (height, width)
            - map_scale_w (np.ndarray): Scalar map of width scale values. Shape: (height, width)
            - map_scale_vang (np.ndarray): Scalar map of vertical angle values. Shape: (height, width)
            - x_gnd (str): Symbolic expression for gndal X coordinate perspective transformation
            - y_gnd (str): Symbolic expression for gndal Y coordinate perspective transformation
            - z_gnd (str): Symbolic expression for gndal Z coordinate perspective transformation
            - x_im (str): Symbolic expression for image X coordinate transformation
            - y_im (str): Symbolic expression for image Y coordinate transformation
            - x_key (str): Symbolic expression for keypoint X coordinate
            - y_key (str): Symbolic expression for keypoint Y coordinate
            - z_key (str): Symbolic expression for keypoint Z coordinate
            - x_ray (str): Symbolic expression for ray X direction
            - y_ray (str): Symbolic expression for ray Y direction
            - z_ray (str): Symbolic expression for ray Z direction
    """
    data = np.load(filename)

    plan_scale = data["plan_scale"]
    im_width = data["im_width"]
    im_height = data["im_height"]

    # Access the arrays
    src2ctd = data["src2ctd"]
    ctd2src = data["ctd2src"]
    map_scale_h = data["map_scale_h"]
    map_scale_w = data["map_scale_w"]
    map_scale_vang = data["map_scale_vang"]

    # String values (convert from numpy string to Python string)
    x_gnd = str(data["x_gnd_exp"])
    y_gnd = str(data["y_gnd_exp"])
    z_gnd = str(data["z_gnd_exp"])
    x_im = str(data["x_im_exp"])
    y_im = str(data["y_im_exp"])
    x_key = str(data["x_key_exp"])
    y_key = str(data["y_key_exp"])
    z_key = str(data["z_key_exp"])
    x_ray = str(data["x_ray_exp"])
    y_ray = str(data["y_ray_exp"])
    z_ray = str(data["z_ray_exp"])

    # Don't forget to close the file
    data.close()

    return {
        "plan_scale": plan_scale,
        "im_width": im_width,
        "im_height": im_height,
        "src2ctd": src2ctd,
        "ctd2src": ctd2src,
        "map_scale_h": map_scale_h,
        "map_scale_w": map_scale_w,
        "map_scale_vang": map_scale_vang,
        "x_gnd": x_gnd,
        "y_gnd": y_gnd,
        "z_gnd": z_gnd,
        "x_im": x_im,
        "y_im": y_im,
        "x_key": x_key,
        "y_key": y_key,
        "z_key": z_key,
        "x_ray": x_ray,
        "y_ray": y_ray,
        "z_ray": z_ray,
    }
