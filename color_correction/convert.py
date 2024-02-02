import numpy as np
import numpy.typing as npt
import torch


ArrayOrTensor = npt.NDArray[np.float64] | torch.Tensor


def XYZ_to_LAB(xyz: ArrayOrTensor,
               white_point: ArrayOrTensor,
               use_torch: bool = False) -> ArrayOrTensor:
    am = torch if use_torch else np

    def f(t: ArrayOrTensor) -> ArrayOrTensor:
        nonlocal am
        delta = 6/29
        delta_cubed = delta * delta * delta

        return (t > delta_cubed) * am.sign(t) * am.abs(t) ** (1/3) + (t <= delta_cubed) * (1/3 * t * (delta ** (-2)) + 4/29)

    xyz_flat = xyz.reshape(-1, 3)
    res = am.zeros(xyz_flat.shape, dtype=am.float64)

    f_x = f(xyz_flat[:, 0] / white_point[0])
    f_y = f(xyz_flat[:, 1] / white_point[1])
    f_z = f(xyz_flat[:, 2] / white_point[2])
    
    res[:, 0] = 116 * f_y - 16
    res[:, 1] = 500 * (f_x - f_y)
    res[:, 2] = 200 * (f_y - f_z)

    return res.reshape(xyz.shape)
