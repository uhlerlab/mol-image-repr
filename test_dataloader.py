from dataset.default_loader import MolImageMismatchDataset
from dataset.gulpio_loader import MolImageMismatchGulpDataset

import time

def test_dataset(mode):
    dataset = MolImageMismatchDataset(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)

    start_time = time.time()
    for idx in range(100):
        sample = dataset[idx]
#        print(sample['image'].shape)
#        print(sample['smiles'])
    print(time.time()-start_time)

def test_gulp_dataset(mode):
    dataset = MolImageMismatchGulpDataset(datadir='data/gulpio/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)
    start_time = time.time()
    for idx in range(100):
        sample = dataset[idx]
#        print(sample['image'].shape)
#        print(sample['smiles'])
    print(time.time()-start_time)

# test code
if __name__ == '__main__':
    test_dataset(mode='train')
    test_dataset(mode='test')
    test_gulp_dataset(mode='train')
    test_gulp_dataset(mode='test')
