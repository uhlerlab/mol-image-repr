import torch
import numpy as np
import pandas as pd
from PIL import Image

from transforms import get_spot_check_transform
from gulpio import GulpDirectory

import os

class MolImageSpotCheck(object):
    def __init__(self, datadir, metafile, savedir="examples", gulp=False):
        self.datadir = datadir
        self.metadata = pd.read_csv(metafile)
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)

        if gulp:
            self.gulpdir = GulpDirectory(datadir)
        else:
            self.gulpdir = None

        self.transforms = get_spot_check_transform()

    def load_img(self, key):
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5

    def save_img(self, idx):
        sample = self.metadata.iloc[idx]
        key = sample['SAMPLE_KEY']
        
        if self.gulpdir is None:
            img = np.load(os.path.join(self.datadir, "%s.npz" % key))
            img = img["sample"] # Shape 520 x 696 x 5
            img = [img[:,:,j] for j in range(5)]
        
        else:
            img, _ = self.gulpdir[key]
        
        img = self.transforms(img)

        print("Saving %s" % key)
        for idx, im in enumerate(img):
            print(im.dtype)
            im = Image.fromarray(im)
            im.save(os.path.join(self.savedir, "%s_%s.png" % (key, idx)))

if __name__ == "__main__":
    dataset = MolImageSpotCheck(datadir='../data/gulpio/',
                              metafile='../data/metadata/datasplit1-train.csv',
                              savedir='examples_gulp', gulp=True)
    for idx in range(1):
        dataset.save_img(idx)
    
    dataset = MolImageSpotCheck(datadir='../data_full/images/',
                              metafile='../data/metadata/datasplit1-train.csv',
                              savedir='examples', gulp=False)
    for idx in range(1):
        dataset.save_img(idx)
