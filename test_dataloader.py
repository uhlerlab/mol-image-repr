from dataset.dataloader import MolImageMismatchDataset
from dataset.gulpio_loader import MolImageMismatchGulpDataset

def test_dataset(mode):
    dataset = MolImageMismatchDataset(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)

    sample = dataset[0]
    for idx in range(100):
        sample = dataset[idx]
        print(sample['img'].shape)
        print(sample['smile'])

def test_gulp_dataset(mode):
    dataset = MolImageMismatchGulpDataset(datadir='data/gulpio/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)

    sample = dataset[0]
    for idx in range(100):
        sample = dataset[idx]
        print(sample['img'].shape)
        print(sample['smile'])

# test code
if __name__ == '__main__':
    start = time.time()
    test_dataset(mode='train')
    print(time.time()-start)
    start = time.time()
    test_gulp_dataset(mode='train')
    print(time.time()-start)
