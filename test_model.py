import torch
from torch.utils.data import DataLoader

from models.molimagenet import model_dict
from dataloader import MolImageDataset, my_collate


def test_model(name):
    dataset = MolImageDataset(datadir='data/images/',
                              metafile='data/metadata/datasplit1-test.csv',
                              mode='train')

    dataloader = DataLoader(dataset, batch_size=32)

    net = model_dict[name]()
    net.cuda()

    for batch in dataloader:
        output = net(batch)
        for k in output.keys():
            print(k)
            print(output[k].shape)
        break

if __name__ == '__main__':
    test_model('molimagenet')
    test_model('molimagenetclass')
