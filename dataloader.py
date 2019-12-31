import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as Transforms

import numpy as np
import pandas as pd

import os

class MolImageDataset(Dataset):
    def __init__(self, datadir, metafile, mode="train"):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)

        if mode == 'train':
            self.transforms = Transforms.Compose([Transforms.ToPILImage(),
                                                 Transforms.RandomCrop(224),
                                                 Transforms.RandomVerticalFlip(),
                                                 Transforms.RandomHorizontalFlip(),
                                                 Transforms.ToTensor(),
                                                 ])
        elif mode == 'val' or mode == 'test':
            self.transforms = Transforms.Compose([Transforms.ToPILImage(),
                                                 Transforms.CenterCrop(224),
                                                 Transforms.ToTensor()
                                                 ])
        else:
            raise KeyError("mode %s is not valid, must be 'train' or 'val' or 'test'" % mode)

        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def load_img(self, key):
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        img = [self.transforms(img[:,:,idx]) for idx in range(5)]
        img = torch.cat(img, 0)


    def __getitem__(self, idx):
        '''Returns a dict corresponding to data sample for the provided index'''

        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']

        # load 5-channel image
        try:
            img = self.load_img(key)

        except Exception as e:
            print(e)
            return None

        return {'key': key, 'cpd_name': sample['CPD_NAME'], 'image': img, 'smiles': sample['SMILES']}

class MolImageMismatchDataset(MolImageDataset):
    def __init__(self, datadir, metafile, mode="train", mismatch_prob=0.5):
        super(MolImageMismatchDataset, self).__init__(datadir=datadir, metafile=metafile, mode=mode)
        self.mismatch_prob = mismatch_prob

    def __getitem__(self, idx):
        '''Returns a dict corresponding to data sample with probability of mismatch'''

        sample_chem = self.metadata.iloc[idx]
        key_chem = sample_chem['SAMPLE_KEY']

        if self.mode == 'train':
            matched = np.random.binomial(1, 1-self.mismatch_prob)
            mismatch_idx = np.random.randint(len(self))
        else:
            matched = np.random.RandomState(seed=idx).binomial(1, 1-self.mismatch_prob)
            mismatch_idx = np.random.RandomState(seed=idx).randint(len(self))

        if matched:
            sample_img = sample_chem
            key_img = key_chem
        else:
            sample_img = self.metadata.iloc[mismatch_idx]
            key_img = sample_img['SAMPLE_KEY']
        
        try:
            img = self.load_img(key_img)

        except Exception as e:
            print(e)
            return None

        return {'key_chem': key_chem, 'key_img': key_img, 
                'cpd_name': sample_chem['CPD_NAME'], 'image': img, 'smiles': sample_chem['SMILES'],
                'target': matched}

def my_collate(batch):
    batch = filter (lambda x:x is not None, batch)
    return default_collate(batch)
