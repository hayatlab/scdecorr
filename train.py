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
from torch.utils.tensorboard import SummaryWriter

from optimizer import LARS, adjust_learning_rate
from models.btwins import BarlowTwins, DSBNBarlowTwins, MMDDSBNBarlowTwins, MMDBarlowTwins
from models.simclr_refactor import SimCLR, DSBNSimCLR 

from augment import Transform
from dataset import TwoBatchDataset, TwoBatchMMDDataset
from loss import InfoNceDist, MMD_loss

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
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='64-64', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
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
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--batch_obs1', type=str,
                    help='batch1 name')
parser.add_argument('--batch_obs2', type=str,
                    help='batch2 name')
parser.add_argument('--batch_obs_name', type=str,
                    help='batch obs field name')
parser.add_argument('--data_obsm_name', type=str,
                    help='data obsm field name')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('-cl', '--contrastive_model', metavar='CL', default='btwins', choices=['simclr','btwins'],
                    help='contrastive learning model (default: simclr)')
parser.add_argument('-optim', '--optimizer', metavar='OPTIM', default='lars', choices=['adam','lars'],
                    help='optimizer (default: lars)')
parser.add_argument('--use_mmd', action='store_true',
                    help='Use MMD loss')
parser.add_argument('--use_dsbn', action='store_true',
                    help='Use Domain Specific BatchNorm')

def main():


    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."    

    if args.use_mmd:
        print('[INFO]....Using MMD....')
        #assert args.contrastive_model == 'simclr' #only supports mmd with simclr
    if args.use_dsbn:
        print('[INFO]....Using DSBN....')
        #assert args.contrastive_model == 'btwins' #only supports dsbn with btwins

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
        args.dist_url = 'tcp://localhost:58499'
        args.world_size = args.ngpus_per_node

    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):

    wandb.init(project="contrasive-scRNA", entity="rtb7syl",config=args)
    writer = SummaryWriter()

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

    # data
    if args.use_mmd:
        dataset = TwoBatchMMDDataset(args.data, args.batch_obs1, args.batch_obs2, args.batch_obs_name,args.data_obsm_name,transform=Transform())
    else:
        dataset = TwoBatchDataset(args.data, args.batch_obs1, args.batch_obs2, args.batch_obs_name,args.data_obsm_name,transform=Transform())

    sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    # model
    if args.contrastive_model == 'btwins':
        print('Running Barlow Twins Model')
        if args.use_dsbn and args.use_mmd:
            model = MMDDSBNBarlowTwins(args).cuda(gpu)
        elif args.use_dsbn and not args.use_mmd:
            model = DSBNBarlowTwins(args).cuda(gpu)
        elif args.use_mmd and not args.use_dsbn:
            model = MMDBarlowTwins(args).cuda(gpu)
        else:
            model = BarlowTwins(args).cuda(gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)


    elif args.contrastive_model == 'simclr':
        print('Running SimCLR Model')
        model = DSBNSimCLR(args).cuda(gpu) if args.use_dsbn else SimCLR(args).cuda(gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        crit = InfoNceDist(temper=0.1, margin=0.).cuda(gpu) # crit not use distributed mode

    model = model.double()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    wandb.watch(model, log_freq=100)

    # optimizer
    if args.optimizer == 'lars':
        
        print('Optimzer: lars')
        param_weights = []
        param_biases = []
        for param in model.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]

        optim = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=True,
                        lars_adaptation_filter=True)

    elif args.optimizer == 'adam':
        print('Optimzer: adam')

        optim = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(loader), eta_min=0,
                                                           last_epoch=-1)

    #max mean discrepancy loss
    if args.use_mmd:
        mmd_loss_fn = MMD_loss()



    # automatically resume from checkpoint if it exists
    if (args.load_checkpoint).is_file():
        ckpt = torch.load(args.load_checkpoint,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        print('start_epoch ',start_epoch)
        model.load_state_dict(ckpt['model'],strict=False)


    else:
        start_epoch = 0

    print(model)
    count_parameters(model)


    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):

        epoch_id='epoch_'+str(epoch)
        epoch_dir=args.checkpoint_dir / epoch_id
        epoch_dir.mkdir(exist_ok=True)
        sampler.set_epoch(epoch)

        #s=source, t=target
        for step, (source_data, target_data) in enumerate(loader, start=epoch * len(loader)):
        #for step, (s1, s2, t1, t2) in enumerate(loader, start=epoch * len(loader)):
            if not args.use_mmd:
                s1, s2 = source_data[0],source_data[1]
                t1, t2 = target_data[0],target_data[1]

            #when using mmd
            else:
                s, s1, s2 = source_data[0],source_data[1],source_data[2] 
                t, t1, t2 = target_data[0],target_data[1],target_data[2]

                s = s.cuda(gpu, non_blocking=True).double()
                t = t.cuda(gpu, non_blocking=True).double()

            s1 = s1.cuda(gpu, non_blocking=True).double()
            s2 = s2.cuda(gpu, non_blocking=True).double()
            t1 = t1.cuda(gpu, non_blocking=True).double()
            t2 = t2.cuda(gpu, non_blocking=True).double()

            if args.use_dsbn:
                sbatch_label = torch.tensor([0]).cuda(gpu)
                tbatch_label = torch.tensor([1]).cuda(gpu)

            if args.optimizer == 'lars':
                adjust_learning_rate(args, optim, loader, step)

            optim.zero_grad()

            with torch.cuda.amp.autocast():
                if args.contrastive_model == "btwins":
                    if args.use_dsbn and args.use_mmd:
                        sloss, emb_s = model.forward(s, s1, s2, sbatch_label)
                        tloss, emb_t = model.forward(t, t1, t2, tbatch_label)

                    elif args.use_dsbn and not args.use_mmd:
                        sloss = model.forward(s1, s2, sbatch_label)
                        tloss = model.forward(t1, t2, tbatch_label)

                    elif args.use_mmd and not args.use_dsbn:
                        sloss, emb_s = model.forward(s, s1, s2)
                        tloss, emb_t = model.forward(t, t1, t2)

                    else:
                        sloss = model.forward(s1, s2)
                        tloss = model.forward(t1, t2)


                elif args.contrastive_model == "simclr":
                    if args.use_dsbn:
                        emb_s1 = model.forward(s1,sbatch_label)
                        emb_s2 = model.forward(s2,sbatch_label)
                        emb_t1 = model.forward(t1,tbatch_label)
                        emb_t2 = model.forward(t2,tbatch_label)
                    else:
                        emb_s1 = model.forward(s1)
                        emb_s2 = model.forward(s2)
                        emb_t1 = model.forward(t1)
                        emb_t2 = model.forward(t2)

                    sloss, stop1, stop5 = crit(emb_s1,emb_s2)
                    tloss, ttop1, ttop5 = crit(emb_t1,emb_t2)

                    if args.use_mmd:
                        if args.use_dsbn:
                            emb_s = model.forward(s,sbatch_label)
                            emb_t = model.forward(t,tbatch_label)
                        else:
                            emb_s = model.forward(s)
                            emb_t = model.forward(t)
                        

                contra_loss = (sloss + tloss)*0.5

                if args.use_mmd:
                    mmd_loss = mmd_loss_fn.forward(emb_s, emb_t)
                    loss = contra_loss + mmd_loss
                else:
                    loss = contra_loss


            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                    source_loss=sloss.item(),
                                    target_loss=tloss.item(),
                                    loss=loss.item(),
                                    time=int(time.time() - start_time))

                    writer.add_scalar('source_loss', sloss, global_step=step)
                    writer.add_scalar('target_loss', tloss, global_step=step)
                    writer.add_scalar('loss', loss, global_step=step)

                    if args.use_mmd:
                        stats.update(dict(contra_loss = contra_loss.item(),mmd_loss = mmd_loss.item()))
                        writer.add_scalar('contra_loss', contra_loss, global_step=step)
                        writer.add_scalar('mmd_loss', mmd_loss, global_step=step)

                    if args.optimizer == 'lars':
                        stats.update(dict(lr_weights = optim.param_groups[0]['lr'],lr_biases = optim.param_groups[1]['lr']))

                        writer.add_scalar('lr_weights', optim.param_groups[0]['lr'], global_step=step)
                        writer.add_scalar('lr_biases', optim.param_groups[1]['lr'], global_step=step)

                    elif args.optimizer == 'adam':
                        stats['lr'] = scheduler.get_lr()[0]

                        writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step=step)

                    if args.contrastive_model == 'simclr':
                        writer.add_scalar('source_acc/top1', stop1[0], global_step=step)
                        writer.add_scalar('source_acc/top5', stop5[0], global_step=step)
                        writer.add_scalar('target_acc/top1', ttop1[0], global_step=step)
                        writer.add_scalar('target_acc/top5', ttop5[0], global_step=step)

                    wandb.log(stats)
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        
        # warmup for the first 10 epochs for cos lr schedule
        if epoch >= start_epoch + 10 and args.optimizer == 'adam':
            scheduler.step()

        if args.rank == 0:
            stats = dict(epoch=epoch, step=step,
                            #contra_loss = contra_loss.item(),
                            #mmd_loss = mmd_loss.item(),
                            source_loss=sloss.item(),
                            target_loss=tloss.item(),
                            loss=loss.item(),
                            time=int(time.time() - start_time))
            if args.use_mmd:
                stats.update(dict(contra_loss = contra_loss.item(),mmd_loss = mmd_loss.item()))

            if args.optimizer == 'lars':
                stats.update(dict(lr_weights = optim.param_groups[0]['lr'],lr_biases = optim.param_groups[1]['lr']))

            elif args.optimizer == 'adam':
                stats['lr'] = scheduler.get_lr()[0]

            wandb.log(stats)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optim=optim.state_dict())
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