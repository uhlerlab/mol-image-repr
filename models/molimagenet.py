import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet18Features

from chemprop.features import get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.models.mpn import MPN

from types import SimpleNamespace

def get_default_args():
    '''Returns Namespace of arguments for MPN and mol2graph compat'''

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
    args.no_cache=False
    args.cuda = True

    return args

class MolImageNet(nn.Module):
    def __init__(self):
        super(MolImageNet, self).__init__()

        self.imagenet = self.get_image_net()
        self.chemnet = self.get_chem_net()

    def get_image_net(self):
        feature_net = Resnet18Features(in_channels=5)
        embed_net = nn.Sequential(nn.Linear(512, 512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 512),
            )

        return nn.ModuleDict({'feature_net': feature_net, 'embed_net': embed_net})

    def get_chem_net(self):

        args = get_default_args()

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

    def compute_loss(self, outputs, targets):

        if next(self.parameters()).is_cuda:
            targets = targets.cuda()

        L1_dist = torch.sum(torch.abs(outputs['image_embedding'] - outputs['chem_embedding']), dim=1)
        targets = (targets*2)-1
        return F.hinge_embedding_loss(L1_dist, targets, margin=1, reduction='mean')

    def compute_acc(self, outputs, targets):
        
        if next(self.parameters()).is_cuda:
            targets = targets.cuda()
        
        L1_dist = torch.sum(torch.abs(outputs['image_embedding'] - outputs['chem_embedding']), dim=1)
        pred = L1_dist > 1
        correct = pred.long().eq(targets.view(-1)).float().sum().item()
        return correct


class MolImageNetClass(nn.Module):
    def __init__(self):
        super(MolImageNetClass, self).__init__()

        self.imagenet = self.get_image_net()
        self.chemnet = self.get_chem_net()
        self.classifier = self.get_classifier()

    def get_image_net(self):
        feature_net = Resnet18Features(in_channels=5)

        return nn.ModuleDict({'feature_net': feature_net})

    def get_chem_net(self):

        args = get_default_args()

        feature_net = MPN(args)

        return nn.ModuleDict({'feature_net': feature_net})

    def get_classifier(self):
        classifier = nn.Sequential(nn.Linear(1024, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1),
            )
        return classifier

    def forward(self, x):
        image_batch = x['image']

        if next(self.parameters()).is_cuda:
            image_batch = image_batch.cuda()

        mol_batch = x['smiles']

        image_batch = self.imagenet['feature_net'](image_batch)
        image_batch = image_batch.view(image_batch.size(0), 512)

        mol_batch = self.chemnet['feature_net'](mol_batch)

        combined_feats = torch.cat((image_batch, mol_batch), dim=1)
        logit = self.classifier(combined_feats)

        return {'image_embedding': image_batch, 'chem_embedding': mol_batch, 'logit': logit}

    def compute_loss(self, outputs, targets):
        '''Binary cross entropy loss function'''

        if next(self.parameters()).is_cuda:
            targets = targets.cuda()
        
        outputs = outputs['logit']
        return F.binary_cross_entropy_with_logits(outputs, targets.view(-1,1).float(), reduction='mean')

    def compute_acc(self, outputs, targets):
        '''Accuracy function'''
        
        if next(self.parameters()).is_cuda:
            targets = targets.cuda()
        
        outputs = outputs['logit']
        pred = outputs>0
        correct = pred.long().eq(targets.view(-1,1)).float().sum().item()
        return correct

model_dict = {'molimagenet': MolImageNet, 'molimagenetclass': MolImageNetClass}
