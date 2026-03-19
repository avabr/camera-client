import numpy as np
import sympy as sp


# Method 1: Read from a saved file
def read_npz_file(filename):
    """Read camera projection data from saved .npz file.

    Args:
        filename (str): Path to the .npz file containing camera projection data

    Returns:
        dict: Dictionary containing the following keys:
            - format_version (str): Version string of the data format
            - plan_url (str): URL or path to the ground plan image
            - plan_scale (float): Scale factor for ground plane coordinates (pixels per meter)
            - plan_width (int): Width of the ground plan in pixels
            - plan_height (int): Height of the ground plan in pixels
            - im_src_url (str): URL or path to the source (distorted) camera image
            - im_ctd_url (str): URL or path to the corrected (undistorted) camera image
            - im_width (int): Width of the camera image in pixels
            - im_height (int): Height of the camera image in pixels
            - src2ctd (np.ndarray): Map of coordinates for undistorted (corrected) image.
                                   Shape: (height, width, 2) where channel 0 is X, channel 1 is Y
            - ctd2src (np.ndarray): Map of coordinates for distorted (raw) image based on undistorted.
                                   Shape: (height, width, 2) where channel 0 is X, channel 1 is Y
            - map_scale_h (np.ndarray): Scalar map of height scale values. Shape: (height, width)
            - map_scale_w (np.ndarray): Scalar map of width scale values. Shape: (height, width)
            - map_scale_vang (np.ndarray): Scalar map of vertical angle values. Shape: (height, width)
            - exp_im2gnd (sp.Expr): Sympy expression for image to ground coordinate transformation
            - exp_gnd2im (sp.Expr): Sympy expression for ground to image coordinate transformation
            - exp_key_point (sp.Expr): Sympy expression for keypoint coordinates
            - exp_im2ray (sp.Expr): Sympy expression for image to ray direction transformation
    """
    data = np.load(filename)

    format_version = str(data["format_version"])

    # Plan options
    plan_url = str(data["plan_url"])
    plan_scale = data["plan_scale"]
    plan_width = data["plan_width"]
    plan_height = data["plan_height"]

    # Camera image options
    im_src_url = str(data["im_src_url"])
    im_ctd_url = str(data["im_ctd_url"])
    im_width = data["im_width"]
    im_height = data["im_height"]

    # Access the arrays
    src2ctd = data["src2ctd"]
    ctd2src = data["ctd2src"]
    map_scale_h = data["map_scale_h"]
    map_scale_w = data["map_scale_w"]
    map_scale_vang = data["map_scale_vang"]

    # String values (convert from sympy srepr to sympy expressions)

    exp_im2gnd = sp.sympify(str(data["exp_im2gnd"]))
    exp_gnd2im = sp.sympify(str(data["exp_gnd2im"]))
    exp_key_point = sp.sympify(str(data["exp_key_point"]))
    exp_im2ray = sp.sympify(str(data["exp_im2ray"]))

    # Don't forget to close the file
    data.close()

    return {
        # Plan options
        "plan_url": plan_url,
        "plan_scale": plan_scale,
        "plan_width": plan_width,
        "plan_height": plan_height,
        # Camera image option
        "im_src_url": im_src_url,
        "im_ctd_url": im_ctd_url,
        "im_width": im_width,
        "im_height": im_height,
        # Distortion coords maps
        "src2ctd": src2ctd,
        "ctd2src": ctd2src,
        # Perspective projection expressions
        "exp_im2gnd": exp_im2gnd,
        "exp_gnd2im": exp_gnd2im,
        "exp_key_point": exp_key_point,
        "exp_im2ray": exp_im2ray,
        # Scale maps
        "map_scale_h": map_scale_h,
        "map_scale_w": map_scale_w,
        "map_scale_vang": map_scale_vang,
    }
