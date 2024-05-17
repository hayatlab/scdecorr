import sys, os
from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from torch import nn
import torch
import numpy as np
import wandb

from optimizer import LARS, adjust_learning_rate
from contrastive_models.btwins import BarlowTwins
from augment import Transform
from dataset import TwoBatchDataset
from loss import MMD_loss

from utils import count_parameters

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='1024-1024-1024', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=20, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=0, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--load_checkpoint', default='', type=Path,
                    metavar='DIR', help='path to checkpoint file to load model from')
parser.add_argument('--model_name', default='barlow_twins', type=str,
                    metavar='mid', help='name/ id of the model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet21', choices=['densenet11','densenet63','densenet21','densenet29'],
                    help='model architecture (default: densenet21)')
parser.add_argument('--in_features', default=2000, type=int, help="number of features")
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--batch_obs1', type=str,
                    help='batch1 name')
parser.add_argument('--batch_obs2', type=str,
                    help='batch2 name')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

def main():

    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58469'
        args.world_size = args.ngpus_per_node

    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    wandb.init(project="contrasive-scRNA", entity="rtb7syl",config=args)

    args.rank += gpu

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.double()
    wandb.watch(model, log_freq=100)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    lars_optim = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    '''
    sgd_optim = torch.optim.SGD(parameters, lr=0,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay) 
    '''
    #max mean discrepancy loss
    mmd_loss_fn = MMD_loss()

    # automatically resume from checkpoint if it exists
    if (args.load_checkpoint).is_file():
        ckpt = torch.load(args.load_checkpoint,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        print('start_epoch ',start_epoch)
        model.load_state_dict(ckpt['model'],strict=False)

        ####remove comment when training from scratch/resuming training
        ####comment while fine-tuning
        #lars_optim.load_state_dict(ckpt['lars_optim'])

    else:
        start_epoch = 0

    print(model)
    count_parameters(model)

    dataset = TwoBatchDataset(args.data, args.batch_obs1, args.batch_obs2, transform=Transform())

    sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):

        epoch_id='epoch_'+str(epoch)
        epoch_dir=args.checkpoint_dir / epoch_id
        epoch_dir.mkdir(exist_ok=True)
        sampler.set_epoch(epoch)

        #s=source, t=target
        for step, (s, s1, s2, t, t1, t2) in enumerate(loader, start=epoch * len(loader)):
        #for step, (s1, s2, t1, t2) in enumerate(loader, start=epoch * len(loader)):
            #print(s, s1, s2, t, t1, t2)
            s = s.cuda(gpu, non_blocking=True).double()
            s1 = s1.cuda(gpu, non_blocking=True).double()
            s2 = s2.cuda(gpu, non_blocking=True).double()
            t = t.cuda(gpu, non_blocking=True).double()
            t1 = t1.cuda(gpu, non_blocking=True).double()
            t2 = t2.cuda(gpu, non_blocking=True).double()

            adjust_learning_rate(args, lars_optim, loader, step)
            lars_optim.zero_grad()

            with torch.cuda.amp.autocast():
                #sloss = model.forward(s1, s2)
                #tloss = model.forward(t1, t2)
                sloss, h_s = model.forward(s, s1, s2)
                tloss, h_t = model.forward(t, t1, t2)

            contra_loss = (sloss + tloss)*0.5
            mmd_loss = mmd_loss_fn.forward(h_s, h_t)
            #loss = (sloss + tloss)*0.5

            #alpha hyperparam
            contra_loss_factor = 1

            #beta hyperparam
            mmd_loss_factor = 20
            loss = contra_loss + mmd_loss
            #loss = (contra_loss_factor*contra_loss) + (mmd_loss_factor*mmd_loss)

            #print('contra_loss, mmd_loss, loss',contra_loss_factor*contra_loss, mmd_loss, loss)

            scaler.scale(loss).backward()
            scaler.step(lars_optim)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                    lr_weights=lars_optim.param_groups[0]['lr'],
                                    lr_biases=lars_optim.param_groups[1]['lr'],
                                    contra_loss = contra_loss.item(),
                                    mmd_loss = mmd_loss.item(),
                                    loss=loss.item(),
                                    time=int(time.time() - start_time))
                    wandb.log(stats)
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            stats = dict(epoch=epoch, step=step,
                            lr_weights=lars_optim.param_groups[0]['lr'],
                            lr_biases=lars_optim.param_groups[1]['lr'],
                            contra_loss = contra_loss.item(),
                            mmd_loss = mmd_loss.item(),
                            loss=loss.item(),
                            time=int(time.time() - start_time))
            wandb.log(stats)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         lars_optim=lars_optim.state_dict())
            torch.save(state, args.checkpoint_dir / epoch_id / 'checkpoint.pth')

    if (args.rank == 0):
        # save final model
        torch.save(model.module.backbone.state_dict(),
                args.checkpoint_dir / epoch_id / 'model.pth')




def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

if __name__ == '__main__':
    main()