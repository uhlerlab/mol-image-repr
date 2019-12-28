import torch
from torch.utils.data import Dataset, DataLoader
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

    def __getitem__(self, idx):
        '''Returns a dict corresponding to data sample for the provided index'''

        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']

        # load 5-channel image
        try:
            img = np.load(os.path.join(self.datadir, "%s.npz" % key))
            img = img["sample"] # Shape 520 x 696 x 5

        except Exception as e:
            print(e)
            return None

        img = [self.transforms(img[:,:,idx]) for idx in range(5)]
        img = torch.cat(img, 0)

        return {'key': key, 'cpd_name': sample['CPD_NAME'], 'image': img, 'smiles': sample['SMILES']}


#def test_dataset(mode):
#    dataset = MolImageDataset(datadir='data/images/',
#                              metafile='data/metadata/datasplit1-test.csv',
#                              mode=mode)

#    sample = dataset[0]
    
#    for k in sample.keys():
#        print(k)
#        try:
#            print(sample[k].shape)
#        except:
#            print(sample[k])
#
#    return dataset

# test code
#if __name__ == '__main__':
#    test_dataset(mode='train')
#    test_dataset(mode='test')
