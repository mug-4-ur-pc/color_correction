import numpy as np
import torch

def XYZ_to_LAB(xyz, white_point, use_torch = False):
    am = torch if use_torch else np

    def f(t):
        nonlocal am
        delta = 6/29

        return (t > delta ** 3) * (t ** (1/3)) + (t <= delta ** 3) * (1/3 * t * delta ** (-2) + 4/29)

    xyz_flat = xyz.reshape(-1, 3)
    res = am.copy(xyz_flat)

    f_x = f(xyz_flat[:, 0] / white_point[0])
    f_y = f(xyz_flat[:, 1] / white_point[1])
    f_z = f(xyz_flat[:, 2] / white_point[2])
    
    res[:, 0] = 116 * f_y - 16
    res[:, 1] = 500 * (f_x - f_y)
    res[:, 2] = 200 * (f_y - f_z)

    return res.reshape(xyz.shape)
