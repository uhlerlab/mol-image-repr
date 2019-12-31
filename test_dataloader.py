from dataloader import MolImageMismatchDataset

def test_dataset(mode):
    dataset = MolImageMismatchDataset(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode=mode)

    sample = dataset[0]
    
    for k in sample.keys():
        print(k)
        try:
            print(sample[k].shape)
        except:
            print(sample[k])

    return dataset

# test code
if __name__ == '__main__':
    test_dataset(mode='train')
    test_dataset(mode='test')
