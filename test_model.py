import torch
from torch.utils.data import DataLoader

from models.molimagenet import MolImageNet
from dataloader import MolImageDataset, my_collate


def test_model():
    dataset = MolImageDataset(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode='train')

    dataloader = DataLoader(dataset, batch_size=32)

    net = MolImageNet()
    net.cuda()

    for batch in dataloader:
        output = net(batch)
        for k in output.keys():
            print(k)
            print(output[k].shape)
        break

if __name__ == '__main__':
    test_model()
