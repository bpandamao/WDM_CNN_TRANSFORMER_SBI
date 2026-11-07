"""Training data: stored clean waveforms, a gap library, and the train/validation split.

Only clean waveforms are stored. Noise is added on the GPU inside each training
step, and each training sample gets a mask drawn at random from the library,
so repeated presentations of a waveform come with different noise and gaps.
Validation uses one fixed mask per waveform.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .wdm import mask_from_indices


def load_gap_library(path):
    """List of index arrays of the missing samples, one per mask."""
    with h5py.File(path, "r") as f:
        return [indices[:] for indices in f["sparse_gap_indices"]]


class WaveformDataset(Dataset):
    """Returns (standardized parameters, clean waveform, random mask from the library)."""

    def __init__(self, waveform_path, gap_library, train_indices):
        self.path = waveform_path
        with h5py.File(waveform_path, "r") as f:
            self.parameters = f["parameters"][:]
        # The flow works on parameters standardized by the training-set mean and standard deviation
        train = self.parameters[np.sort(np.asarray(train_indices))]
        self.param_mean = np.mean(train, axis=0, dtype=np.float64)
        self.param_std = np.std(train, axis=0, dtype=np.float64)
        self.gap_library = gap_library
        self.file = None

    def __len__(self):
        return len(self.parameters)

    def waveform(self, index):
        if self.file is None:                       # opened lazily in each DataLoader worker
            self.file = h5py.File(self.path, "r")
        return np.asarray(self.file["time_signals"][index], dtype=np.float32)

    def target(self, index):
        theta = np.asarray(self.parameters[index], dtype=np.float64)
        return ((theta - self.param_mean) / self.param_std).astype(np.float32)

    def __getitem__(self, index):
        clean = self.waveform(index)
        mask = mask_from_indices(self.gap_library[np.random.randint(len(self.gap_library))], len(clean))
        return self.target(index), clean, mask


class FixedMaskDataset(Dataset):
    """Validation waveforms, each with one fixed mask from the library."""

    def __init__(self, dataset, indices, seed=config.VALIDATION_MASK_SEED):
        self.dataset, self.indices = dataset, list(indices)
        self.pattern = [int(np.random.default_rng(seed + int(i)).integers(0, len(dataset.gap_library)))
                        for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        clean = self.dataset.waveform(index)
        mask = mask_from_indices(self.dataset.gap_library[self.pattern[position]], len(clean))
        return self.dataset.target(index), clean, mask


def train_validation_split(n, fraction=config.TRAIN_FRACTION, seed=config.SPLIT_SEED):
    """The 80/20 split of the paper (torch random_split with seed 42)."""
    n_train = int(fraction * n)
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return np.asarray(permutation[:n_train]), np.asarray(permutation[n_train:])
