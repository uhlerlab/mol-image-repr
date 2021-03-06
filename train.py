import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset.dataloader import dataset_dict, my_collate
from models.molimagenet import model_dict
from training.utils import train_model, evaluate_model
from utils import setup_args, setup_logger, save_checkpoint, setup_optimizer, setup_lr_scheduler

import os
import random

def run_training(args, logger):

    # seed run
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # setup model
    net = model_dict[args.model]()

    logger.info(net)
    
    if args.use_gpu:
        net.cuda()

    # load data
    
    trainset = dataset_dict[args.dataset](datadir=args.datadir, metafile=args.train_metafile, mode="train")
    testset = dataset_dict[args.dataset](datadir=args.datadir, metafile=args.val_metafile, mode="val")

    trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)
    testloader = DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers, collate_fn=my_collate)

    # setup optimizer and scheduler
    optimizer = setup_optimizer(name=args.optimizer, param_list=[{'params': net.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}])
    scheduler = setup_lr_scheduler(name=args.scheduler, optimizer=optimizer)

    # main training loop
    best_accuracy = 0

    for epoch in range(args.max_epochs):
        logger.info("Epoch %s:" % epoch)

        train_summary = train_model(trainloader=trainloader, model=net, optimizer=optimizer, loss_fn=net.compute_loss, acc_fn=net.compute_acc,
                                    single_batch=args.debug_mode)
        logger.info("Training summary: %s" % train_summary)

        test_summary = evaluate_model(testloader=testloader, model=net, loss_fn=net.compute_loss, acc_fn=net.compute_acc,
                                      single_batch=args.debug_mode)
        logger.info("Evaluation summary: %s" % test_summary)

        if scheduler is not None:
            scheduler.step()
        
        current_state = {'epoch': epoch, 'state_dict': net.cpu().state_dict(), 'best_acc': best_accuracy, 
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None}

        if test_summary['test_acc'] > best_accuracy:
            best_accuracy = test_summary['test_acc']
            current_state['best_acc'] = best_accuracy
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir,"models/best.pth"))

        logger.info("Best accuracy: %s" % best_accuracy)
        
        if epoch % args.save_freq == 0:
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/epoch_%s.pth" % epoch))

        save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/last.pth"))

        if args.use_gpu:
            net.cuda()

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)
    logger = setup_logger(name="training_log", save_dir=args.save_dir)
    logger.info(args)
    run_training(args, logger)
