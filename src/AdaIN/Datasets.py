import numpy as np
import torch
import h5py
from pathlib import Path


def load_h5(path, key):
    with h5py.File(path, mode="r") as f:
        return f[key][:]


# torch doesn't support most unsigned types,
# so we map them to their signed equivalent
DTYPE_MAP = {
    np.dtype("uint16"): np.int16,
    np.dtype("uint32"): np.int32,
    np.dtype("uint64"): np.int64,
}


def ensure_tensor(tensor, dtype=None):
    if isinstance(tensor, np.ndarray):
        if np.dtype(tensor.dtype) in DTYPE_MAP:
            tensor = tensor.astype(DTYPE_MAP[tensor.dtype])
        tensor = torch.from_numpy(tensor)

    assert torch.is_tensor(tensor), f"Cannot convert {type(tensor)} to torch"
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


class PatchPositionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_path: Path,
        raw_key: str,
        patch_starts: np.array,
        patch_shape: tuple = (1, 80, 80),
        dtype=torch.float32,
        n_samples: int = None,
        repeat_patches: bool = False,
        random_seed: int = None,
    ):
        super(PatchPositionDataset, self).__init__()
        self.raw_path = raw_path
        self.raw_key = raw_key
        self.raw = load_h5(raw_path, raw_key)
        self.repeat_patches = repeat_patches
        self.patch_shape = patch_shape
        self.dtype = dtype
        self.random_seed = random_seed
        self.max_len = len(patch_starts)
        self._len = self.max_len if n_samples is None else n_samples
        self.patch_start_positions = self.patch_start_sample(
            patch_starts, self._len, self.random_seed, self.repeat_patches
        )
        assert len(self.patch_start_positions) == self._len

    def __len__(self):
        return self._len

    @staticmethod
    def patch_start_sample(patch_positions, num_samples, random_seed, repeat_patches):
        if random_seed is None:
            assert len(patch_positions) >= num_samples, "Not enough patches to sample"
            return patch_positions[:num_samples]
        else:
            r = np.random.RandomState(random_seed)
            return patch_positions[
                r.choice(
                    patch_positions.shape[0],
                    num_samples,
                    replace=repeat_patches,
                )
            ]

    def _sample_bounding_box(self, bb_start):
        return tuple(
            slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape)
        )

    def _get_sample(self, index):
        if self.raw is None:
            raise RuntimeError(
                "ClassificationDataset has not been properly deserialized."
            )
        bb_start = self.patch_start_positions[index]
        bb = self._sample_bounding_box(bb_start)
        raw = self.raw[bb]

        # ensure greyscale image has 1 channel
        if len(self.patch_shape) == 2:
            raw = np.expand_dims(raw, 0)

        return raw

    def __getitem__(self, index):
        raw = self._get_sample(index)

        raw = ensure_tensor(raw, dtype=self.dtype)

        return raw
