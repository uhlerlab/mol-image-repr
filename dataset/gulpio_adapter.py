from gulpio.adapters import AbstractDatasetAdapter
from gulpio.fileio import GulpIngestor

import pandas as pd
import numpy as np
import torchvision.transforms as Transforms

import os

class MolChemAdapter(AbstractDatasetAdapter):
    def __init__(self, datadir, metafile):
        self.datadir = datadir
        self.metadata = self.load_metadata(metafile)
        print(self.metadata.head())
        print(len(self))
    
    def __len__(self):
        return len(self.metadata)

    def load_metadata(self, metafile):
        if isinstance(metafile, list):
            data = [pd.read_csv(f) for f in metafile]
            return pd.concat(data, ignore_index=True)
        else:
            return pd.read_csv(metafile)

    def iter_data(self, slice_element=None):
        s = slice_element or slice(0, len(self))
        indices = range(s.start, s.stop, s.step if s.step is not None else 1)
        for idx in indices:
            sample = self.metadata.iloc[idx].to_dict()
            id = sample['SAMPLE_KEY']
            frames = self.load_img(id)
            result = {'id': id, 'frames': frames, 'meta': sample}
            yield result

    def load_img(self, key):
        img = np.load(os.path.join(self.datadir, key+".npz"))
        img = img["sample"] # Shape 520 x 696 x 5
        img = [img[:,:,idx] for idx in range(5)]
        return img

if __name__ == "__main__":
    adapter = MolChemAdapter(datadir='../data/images/', metafile=['../data/metadata/datasplit1-train.csv', 
                                                               '../data/metadata/datasplit1-val.csv', 
                                                               '../data/metadata/datasplit1-test.csv'])
    ingestor = GulpIngestor(adapter,
                            output_folder='../data/gulpio/',
                            videos_per_chunk=1000,
                            num_workers=16)
    ingestor()
