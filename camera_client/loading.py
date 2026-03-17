import numpy as np
import sympy as sp


# Method 1: Read from a saved file
def read_npz_file(filename):
    """Read camera projection data from saved .npz file.

    Args:
        filename (str): Path to the .npz file containing camera projection data

    Returns:
        dict: Dictionary containing the following keys:
            - format_version (str): Version of the data format
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
            - exp_im2gnd (sp.Expr): Sympy expression for image to ground coordinate transformation
            - exp_gnd2im (sp.Expr): Sympy expression for ground to image coordinate transformation
            - exp_key_point (sp.Expr): Sympy expression for keypoint coordinates
            - exp_im2ray (sp.Expr): Sympy expression for image to ray direction transformation
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

    # String values (convert from sympy srepr to sympy expressions)

    exp_im2gnd = sp.sympify(str(data["exp_im2gnd"]))
    exp_gnd2im = sp.sympify(str(data["exp_gnd2im"]))
    exp_key_point = sp.sympify(str(data["exp_key_point"]))
    exp_im2ray = sp.sympify(str(data["exp_im2ray"]))

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
        "exp_im2gnd": exp_im2gnd,
        "exp_gnd2im": exp_gnd2im,
        "exp_key_point": exp_key_point,
        "exp_im2ray": exp_im2ray,
    }
