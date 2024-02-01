from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from color_correction.errors import CIELABDE


class _Polynom:
    def __init__(self: Self,
                 degree: int,
                 norm_degrees: bool) -> None:
        if degree == 1:
            self.fit = lambda x: x
            return

        self._create_poly(degree)
        if norm_degrees:
            self._normalize()

    def fit(self: Self,
            x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x_flat = x.reshape(-1, 3)
        res = np.zeros((x_flat.shape[0], len(self._poly)))
        for i, k in enumerate(self._poly):
            res[:, i] = np.prod(x ** k, axis=1)

        return res.reshape(x.shape[:-1] + tuple([-1]))

    def _create_poly(self: Self, d: int) -> None:
        if d < 1:
            raise ValueError("Degree of the polynom must be a positive number")

        self._poly: list[tuple] = []
        for a in range(d + 1):
            for b in range(d - a + 1):
                for c in range(d - a - b + 1):
                    if a == b == c == 0:
                        continue
                    
                    self._poly.append((a, b, c))

    def _normalize(self: Self) -> None:
        for i, (a, b, c) in enumerate(self._poly):
            s = a + b + c
            self._poly[i] = (a / s, b / s, c / s)

        self._poly = list(set(self._poly))


class RegressionCC:
    def __init__(self: Self,
                 degree: int = 1,
                 norm_degrees: bool = False,
                 loss: str = "mse") -> None:
        self._poly = _Polynom(degree=degree, norm_degrees=norm_degrees)

        if loss == "mse":
            self.fit = self._fit_mse
        elif loss == "cielabde":
            self.fit = self._fit_cielab
        else:
            raise ValueError(f"Unknown loss function {loss}")

    def _get_global_min_mse(self: Self, rgb_poly: npt.NDArray[np.float64],
                        xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.linalg.inv(rgb_poly.T @ rgb_poly) @ rgb_poly.T @ xyz

    def _fit_mse(self: Self, rgb: npt.NDArray[np.float64],
                    xyz: npt.NDArray[np.float64],
                    _: npt.NDArray[np.float64]) -> None:
        rgb_poly = self._poly.fit(rgb)
        self._M = self._get_global_min_mse(rgb_poly, xyz)
        
    def _fit_cielab(self: Self, rgb: npt.NDArray[np.float64],
                    xyz: npt.NDArray[np.float64],
                    white_point: npt.NDArray[np.float64]) -> None:

        def loss(m: npt.NDArray[np.float64],
                 rgb_poly: npt.NDArray[np.float64],
                 xyz: npt.NDArray[np.float64],
                 white_point: npt.NDArray[np.float64]) -> float:
            m = m.reshape(-1, 3)
            return CIELABDE(rgb_poly @ m, xyz, white_point).sum()

        rgb_poly = self._poly.fit(rgb)
        m0 = self._get_global_min_mse(rgb_poly, xyz)
        self._M = minimize(loss, m0.reshape(-1),
                           args=(rgb_poly, xyz, white_point),
                           method="Nelder-Mead").x.reshape(-1, 3)

    def predict(self: Self, rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not "_M" in dir(self):
            raise RuntimeError("Model was not trained")
            
        return self._poly.fit(rgb) @ self._M


class LCC(RegressionCC):
    def __init__(self: Self) -> None:
        super().__init__()


class PCC(RegressionCC):
    def __init__(self: Self,
                 degree: int = 2,
                 loss: str = "mse") -> None:
        super().__init__(degree=degree, norm_degrees=False, loss=loss)


class RPCC(RegressionCC):
    def __init__(self: Self,
                 degree: int = 2,
                 loss: str = "mse") -> None:
        super().__init__(degree=degree, norm_degrees=True, loss=loss)
