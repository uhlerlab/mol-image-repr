import torch
import torchvision.transforms as Transforms

import numpy as np
import pandas as pd
from PIL import Image

import os

class MolImageSpotCheck(object):
    def __init__(self, datadir, metafile, savedir="examples", mode="train"):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.savedir = savedir

        if mode == 'train':
            self.transforms = Transforms.Compose([Transforms.ToPILImage(),
                                                 Transforms.RandomCrop(224),
                                                 Transforms.RandomVerticalFlip(),
                                                 Transforms.RandomHorizontalFlip(),
                                                 ])
        elif mode == 'val' or mode == 'test':
            self.transforms = Transforms.Compose([Transforms.ToPILImage(),
                                                 Transforms.CenterCrop(224),
                                                 ])
        else:
            raise KeyError("mode %s is not valid, must be 'train' or 'val' or 'test'" % mode)

        self.mode = mode

    def save_img(self, idx):
        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']
        
        print("Saving %s" % key)
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        img = [self.transforms(img[:,:,idx]) for idx in range(5)]

        for idx, im in enumerate(img):
            print(min(im), max(im))
            im.save(os.path.join(self.savedir, "%s_%s.png" % (key, idx)))

if __name__ == "__main__":
    dataset = MolImageSpotCheck(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)
    for idx in range(10):
        dataset.save_img(idx)
