import torch
from torch.utils.data.dataloader import default_collate

from .default_loader import MolImageDataset, MolImageMismatchDataset
from .gulpio_loader import MolImageGulpDataset, MolImageMismatchGulpDataset

dataset_dict = {'default': MolImageDataset, 'gulp': MolImageGulpDataset, 
                'mismatch': MolImageMismatchDataset, 'mismatch-gulp': MolImageMismatchGulpDataset}

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return default_collate(batch)