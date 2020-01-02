from gulpio.adapters import AbstractDatasetAdapter
from gulpio.fileio import GulpIngestor

class MolChemAdapter(AbstractDatasetAdapter):
    def __init__(self, datadir, metadata):
        self.datadir = datadir
        self.metadata = self.load_metadata(metafile)
        print(metadata.head())
        self.transforms = Transforms.Compose([Transforms.ToPILImage(),
                                                 ])
    def __len__(self):
        return len(self.metadata)

    def load_metadata(self, metafile):
        if isinstance(metafile, list):
            data = [pd.read_csv(f) for f in metafile]
            return pd.concat(data, ignore_index=True)
        else:
            return pd.read_csv(metafile)

    def iter_data(self, slice_element=None):
        slice_element = slice_element or slice(0, len(self))
        for idx in list(slice_element):
            sample = self.metadata.iloc[idx]
            id = sample['SAMPLE_KEY']
            frames = self.load_img(id)
            
            yield {'id': id, 'frames': frames, 'meta': sample}

    def load_img(self, key):
        img = np.load(os.path.join(self.datadir, "%s.npz" % key))
        img = img["sample"] # Shape 520 x 696 x 5
        img = [self.transforms(img[:,:,idx]) for idx in range(5)]

        return img

if __name__ == "__main__":
    adapter = MolChemAdapter(datadir='data/images/', metafile=['data/metadata/datasplit1-train.csv', 
                                                               'data/metadata/datasplit1-val.csv', 
                                                               'data/metadata/datasplit1-test.csv'])
    ingestor = GulpIngestor(adapter,
                            output_folder='data/gulpio/',
                            videos_per_chunk=1000,
                            num_workers=16)
    ingestor()
