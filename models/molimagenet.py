import torch
import torch.nn as nn
from torchvision.models import resnet18

from chemprop.features import get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.models import MPN

from types import SimpleNamespace

class MolImageNet(nn.Module):
    def __init__(self):
        super(MolImageNet, self).__init__()

        self.imagenet = self.get_image_net()
        self.chemnet = self.get_chem_net()

    def get_image_net(self):
        feature_net = resnet18(pretrained=False)
        embed_net = nn.Sequential(nn.Linear(512, 512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 512),
            )

        return nn.ModuleDict({'feature_net': feature_net, 'embed_net': embed_net})

    def get_chem_net(self):

        args = SimpleNamespace()
        args.hidden_size=512
        args.bias=False
        args.depth=3
        args.dropout=0.0
        args.activation='ReLU'
        args.undirected=False
        args.ffn_hidden_size=None
        args.atom_messages=False
        args.use_input_features=False
        args.features_only=False

        feature_net = MPN(args)
        embed_net = nn.Sequential(nn.Linear(512, 512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 512),
            )

        return nn.ModuleDict({'feature_net': feature_net, 'embed_net': embed_net})

    def forward(self, x):
        image_batch = x['image']

        if next(self.parameters()).is_cuda:
            image_batch = image_batch.cuda()

        mol_batch = x['smiles']

        image_batch = self.imagenet['feature_net'](image_batch)
        image_batch = image_batch.view(image_batch.size(0), 512)
        image_batch = self.imagenet['embed_net'](image_batch)

        mol_batch = self.chemnet['feature_net'](mol_batch)
        mol_batch = self.chemnet['embed_net'](mol_batch)

        return {'image_embedding': image_batch, 'chem_embedding': mol_batch}