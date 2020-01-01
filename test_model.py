import torch
from torch.utils.data import DataLoader

from models.molimagenet import model_dict
from dataloader import MolImageMismatchDataset, my_collate


def test_model(name):
    dataset = MolImageMismatchDataset(datadir='data/images/',
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
        loss = net.compute_loss(output, batch['target'])
        print("Loss: %s" % loss)
        acc = net.compute_acc(output, batch['target'])
        print("Acc: %s" % (acc / len(batch['target'])))
        break

if __name__ == '__main__':
    test_model('molimagenet')
    test_model('molimagenetclass')
