import torch
from torch import optim

import argparse
import logging
import os

def setup_args():

    options = argparse.ArgumentParser()
    # save and directory options
    options.add_argument('--datadir', action="store", default="data/images/")
    #options.add_argument('--datadir', action="store", default="data/gulpio/")
    options.add_argument('--train-metafile', action="store", default="data/metadata/datasplit1-train.csv")
    options.add_argument('--val-metafile', action="store", default="data/metadata/datasplit1-val.csv")
    options.add_argument('--save-dir', action="store", default='results/test/')
    options.add_argument('--save-freq', action="store", default=1, type=int)

    # model parameters
    options.add_argument('--model', action="store", dest="model", default='molimagenetclass')
    options.add_argument('--dataset', action="store", dest="dataset", default='mismatch')
    options.add_argument('--optimizer', action="store", dest="optimizer", default='sgd')
    options.add_argument('--scheduler', action="store", dest="scheduler", default='step')

    # training parameters
    options.add_argument('--batch-size', action="store", dest="batch_size", default=64, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lr', '--learning-rate', action="store", dest="learning_rate", default=1e-2, type=float)
    options.add_argument('--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('--weight-decay', action="store", dest="weight_decay", default=1e-5, type=float)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    return options.parse_args()

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, 'info.log'))
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger

def save_checkpoint(current_state, filename):
    torch.save(current_state, filename)

def setup_optimizer(name, param_list):
    if name == 'sgd':
        return optim.SGD(param_list, momentum=0.9)
    elif name == 'adam':
        return optim.Adam(param_list)
    else:
        raise KeyError("%s is not a valid optimizer (must be one of ['sgd', adam']" % name)

def setup_lr_scheduler(name, optimizer):
    if name == 'none':
        return None
    elif name == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    
