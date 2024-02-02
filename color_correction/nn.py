from typing import Self
import numpy as np
import numpy.typing as npt
import torch

from color_correction.errors import CIEDE2000


class MLP(torch.nn.Module):
    def __init__(self: Self,
                 epochs: int = 2500,
                 max_inc_seq_len: int = 100,
                 lr: float = 0.001,
                 test_size: float = 0.2,
                 batch_size: int = 256) -> None:
        super().__init__()

        self._epochs = epochs
        self._max_inc_seq_len = max_inc_seq_len
        self._lr = lr
        self._test_size = test_size
        self._batch_size = batch_size

        self._net = torch.nn.Sequential(
                torch.nn.Linear(3, 79, dtype=torch.float64),
                torch.nn.ELU(),
                torch.nn.Linear(79, 36, dtype=torch.float64),
                torch.nn.ELU(),
                torch.nn.Linear(36, 3, dtype=torch.float64),
        )
        self._device = "cuda" if torch.cuda.is_available() else 'cpu'

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)

    def fit(self: Self,
            rgb: npt.NDArray[np.float64],
            xyz: npt.NDArray[np.float64],
            white_point: npt.NDArray[np.float64]) -> None:
        test_size = round(self._test_size * rgb.shape[0])
        rgb_test = torch.tensor(rgb[:test_size]).to(self._device)
        xyz_test = torch.tensor(xyz[:test_size]).to(self._device)
        rgb_train = torch.tensor(rgb[test_size:], requires_grad=True).to(self._device)
        xyz_train = torch.tensor(xyz[test_size:], requires_grad=True).to(self._device)
        wp = torch.tensor(white_point, requires_grad=True).to(self._device)

        self.to(self._device)

        batch_size = rgb_train.shape[0] if self._batch_size == 0 else self._batch_size
        min_loss = float('inf')
        inc_loss_counter = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        for _ in range(self._epochs):
            permutation = torch.randperm(rgb_train.shape[0])
            
            for i in range(0, rgb_train.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                rgb_batch = rgb_train[indices]
                xyz_batch = xyz_train[indices]

                xyz_pred = self(rgb_batch)
                loss = CIEDE2000(xyz_pred, xyz_batch, wp, use_torch=True).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                xyz_pred = self(rgb_test)
                loss = CIEDE2000(xyz_pred, xyz_test, wp, use_torch=True).sum()
                if min_loss > loss:
                    min_loss = loss
                    inc_loss_counter = 0
                elif loss > min_loss:
                    inc_loss_counter += 1
                    if inc_loss_counter == self._max_inc_seq_len:
                        self.to('cpu')
                        return

        self.to('cpu')

    def predict(self: Self,
                rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self(torch.tensor(rgb, requires_grad=False)).detach().numpy()

   
class MLPExtendedTrain(MLP):
    def __init__(self: Self,
                 epochs: int = 500,
                 max_inc_seq_len: int = 100,
                 lr: float = 0.001,
                 test_size: float = 0.2,
                 batch_size: int = 256,
                 exposures: list[float] = [0.1, 0.2, 9.5, 1, 2, 5, 10]) -> None:
        super().__init__(epochs=epochs,
                         max_inc_seq_len=max_inc_seq_len,
                         lr=lr,
                         test_size=test_size,
                         batch_size=batch_size)
        self._exposures = exposures
        
    def fit(self: Self,
            rgb: npt.NDArray[np.float64],
            xyz: npt.NDArray[np.float64],
            white_point: npt.NDArray[np.float64]) -> None:
        rgb_aug = np.zeros([0, 3], dtype=np.float64)
        xyz_aug = np.zeros([0, 3], dtype=np.float64)
        for exposure in self._exposures:
            rgb_aug = np.vstack((rgb_aug, exposure * rgb))
            xyz_aug = np.vstack((xyz_aug, exposure * xyz))

        permutation = np.random.permutation(rgb_aug.shape[0])
        rgb_aug = rgb_aug[permutation]
        xyz_aug = xyz_aug[permutation]
            
        super().fit(rgb_aug, xyz_aug, white_point)


class MLPExposureInvariant:
    def __init__(self: Self,
                 epochs: int = 2500,
                 max_inc_seq_len: int = 2500,
                 lr: float = 0.0001,
                 test_size: float = 0.2,
                 batch_size: int = 256) -> None:
        self._net = MLP(epochs=epochs,
                        max_inc_seq_len=max_inc_seq_len,
                        lr=lr,
                        test_size=test_size,
                        batch_size=batch_size)
        self._brightness_correct = np.zeros([3, 1], dtype=np.float64)

    def fit(self: Self,
            rgb: npt.NDArray[np.float64],
            xyz: npt.NDArray[np.float64],
            white_point: npt.NDArray[np.float64]) -> None:
        rgb_norm = np.sum(rgb, axis=1).reshape(-1, 1)
        xyz_norm = np.sum(xyz, axis=1).reshape(-1, 1)
        wp_norm = np.sum(white_point)

        self._net.fit(rgb / rgb_norm, xyz / xyz_norm, white_point / wp_norm)
        self._brightness_correct = np.linalg.pinv(rgb.T @ rgb) @ rgb.T @ xyz_norm

    def predict(self: Self,
                rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        rgb_norm = np.sum(rgb, axis=1).reshape(-1, 1)
        xyz_norm = rgb @ self._brightness_correct
        xyz_normalized = self._net.predict(rgb / rgb_norm)
        return xyz_normalized * xyz_norm
