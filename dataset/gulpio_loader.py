from .default_loader import MolImageDataset, MolImageMismatchDataset
from gulpio import GulpDirectory

import torch

class MolImageGulpDataset(MolImageDataset):
    def __init__(self, datadir, metafile, mode="train", num_samples=None, **kwargs):
        super(MolImageGulpDataset, self).__init__(datadir=datadir, metafile=metafile, mode=mode, num_samples=num_samples)
        self.gulpdir = GulpDirectory(self.datadir)

    def load_img(self, key):
        img, _ = self.gulpdir[key]
        img = [self.transforms(img[idx]) for idx in range(5)]
        img = torch.cat(img, 0)

        return img

class MolImageMismatchGulpDataset(MolImageGulpDataset, MolImageMismatchDataset):
    def __init__(self, datadir, metafile, mode="train", mismatch_prob=0.5, num_samples=None, **kwargs):
        super(MolImageMismatchGulpDataset, self).__init__(datadir=datadir, metafile=metafile, mode=mode, num_samples=num_samples)
