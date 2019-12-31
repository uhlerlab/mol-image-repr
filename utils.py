import argparse
import logging

def setup_args():

    options = argparse.ArgumentParser()
    # save and directory options
    options.add_argument('--data-dir', action="store", default="data/images/")
    options.add_argument('--train-metafile', action="store", default="data/metadata/datasplit1-train.csv")
    options.add_argument('--val-metafile', action="store", default="data/metadata/datasplit1-val.csv")
    options.add_argument('--save-dir', action="store", default='results/test/')
    options.add_argument('--save-freq', action="store", default=1, type=int)

    # model parameters
    options.add_argument('--model', action="store", dest="model", default='molimageclassnet')

    # training parameters
    options.add_argument('--batch-size', action="store", dest="batch_size", default=32, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lr', '--learning-rate', action="store", dest="learning_rate", default=1e-4, type=float)
    options.add_argument('--max-epochs', action="store", dest="max_epochs", default=1000, type=int)
    options.add_argument('--weight-decay', action="store", dest="weight_decay", default=1e-5, type=float)

    # gpu options
    options.add_argument('--use-gpu', action="store_false", default=True)

    return options.parse_args()

def create_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, 'info.log'))
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger