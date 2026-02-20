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
from functools import reduce
from torch import nn
import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

from optimizer import LARS, adjust_learning_rate
from models.btwins import BarlowTwins, DSBNBarlowTwins
from models.simclr_refactor import SimCLR, DSBNSimCLR 
from seeding import seed_everything
from augment import Transform
from dataset import MultiBatchDataset
from loss import InfoNceDist
import yaml

from utils import count_parameters

# put this at the very top of your training script (before other imports)
import warnings

# silence the specific pydantic messages
#warnings.filterwarnings("ignore")

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
parser.add_argument('--save-freq', default=20, type=int, metavar='N',
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
parser.add_argument('--n_domains', default=2, type=int,
                    help='Number of experimental batches (default: 2)')
parser.add_argument('--batch_obs_name', type=str,
                    help='batch obs field name')
parser.add_argument('--data_obsm_name', type=str,
                    help='data obsm field name')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('-cl', '--contrastive_model', metavar='CL', default='btwins', choices=['simclr','btwins'],
                    help='contrastive learning model (default: simclr)')
parser.add_argument('-optim', '--optimizer', metavar='OPTIM', default='lars', choices=['adam', 'lars', 'adamw'],
                    help='optimizer (default: lars)')
parser.add_argument('--use_dsbn', action='store_true',
                    help='Use Domain Specific BatchNorm')
parser.add_argument('--cfg_path', default='model_config.yaml', type=Path, help='path to output model config file')
parser.add_argument('--seed', default=42, type=Path, help='seed')



def main():

    args = parser.parse_args()

    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."    

    print('[INFO]....number of domains = {n}'.format(n=args.n_domains))
    
    assert args.contrastive_model == 'btwins'

    if args.use_dsbn:
        print('[INFO]....Using DSBN....')
        #assert args.contrastive_model == 'btwins' #only supports dsbn with btwins
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_worker(device, args)
    return 0


def main_worker(device, args):
    wandb.init(project="contrasive-scRNA", entity="rtb7syl",config=args)
    writer = SummaryWriter()

    seed_everything(seed=args.seed, cuda=True, workers=True)


    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)


    dataset = MultiBatchDataset(args.data, args.n_domains, args.batch_obs_name,args.data_obsm_name,transform=Transform())
    print(dataset)

    print(vars(args))
    print('args.batch_size', args.batch_size)

    loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, 
            num_workers=args.workers, shuffle=True, pin_memory=True
    )

    print(len(loader))
    # model
    if args.contrastive_model == 'btwins':
        if args.use_dsbn:
            model = DSBNBarlowTwins(args)
        else:
            model = BarlowTwins(args)

    elif args.contrastive_model == 'simclr':
        print('Running SimCLR Model')
        model = DSBNSimCLR(args).to(device) if args.use_dsbn else SimCLR(args)
        crit = InfoNceDist(temper=0.1, margin=0.) # crit not use distributed mode

    model = model.to(device=device, dtype=torch.float32)

    wandb.watch(model, log_freq=100)

    print(f'Learning Rate LR={args.lr}')

    # optimizer
    if args.optimizer == 'lars':
        
        print('Optimzer: LARS')
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
        print('Optimzer: Adam')
        optim = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7, last_epoch=-1)

    elif args.optimizer == 'adamw':
        print('Optimzer: AdamW')
        optim = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7, last_epoch=-1)
    
    else:
        raise ValueError(f'Optimizer must be in [lars, adam, adamw], got optimizer={args.optimizer}')

    # automatically resume from checkpoint if it exists
    if (args.load_checkpoint).is_file():
        ckpt = torch.load(args.load_checkpoint,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        print('start_epoch ',start_epoch)
        model.load_state_dict(ckpt['model'],strict=False)
        optim.load_state_dict(ckpt['optim'])

    else:
        start_epoch = 0

    print(model)
    args.total_params = count_parameters(model)

    # 3. Convert args (Namespace) to a dict
    args_dict = vars(args)
    # 4. Save arguments to YAML file
    with open(args.cfg_path, "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)

    #torch.autograd.set_detect_anomaly(True)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_id='epoch_'+str(epoch)
        epoch_dir=args.checkpoint_dir / epoch_id

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            epoch_dir.mkdir(exist_ok=True)

        epoch_loss = 0.0
        #s=source, t=target
        for step, domain_datas in enumerate(loader, start=epoch * len(loader)):
            #print('domain_datas shapes',list(map(lambda domain_data: (domain_data[0].size(),domain_data[1].size()),domain_datas)))
            #domain_datas_aug_views = [(domain1_data_aug1,domain1_data_aug2),.....]
            domain_datas_aug_views = list(map(lambda domain_data:(domain_data[0].to(device=device, dtype=torch.float32, non_blocking=True), domain_data[1].to(device=device, dtype=torch.float32, non_blocking=True)), domain_datas))

            if args.use_dsbn:
                domain_labels = list(map(lambda i:torch.tensor([i]).to(device=device), range(args.n_domains)))
                #print('domain_labels',domain_labels)

            if args.optimizer == 'lars':
                adjust_learning_rate(args, optim, loader, step)

            optim.zero_grad()

            if args.contrastive_model == "btwins":

                if args.use_dsbn:
                    domain_losses = list(map(lambda i:model.forward(domain_datas_aug_views[i][0], domain_datas_aug_views[i][1], domain_labels[i]), range(args.n_domains)))

                else:
                    domain_losses = list(map(lambda i:model.forward(domain_datas_aug_views[i][0], domain_datas_aug_views[i][1]), range(args.n_domains)))

            elif args.contrastive_model == "simclr":

                #domain_datas_aug_views_embeddings = [(emb_domain1_aug1,emb_domain1_aug2),....]
                if args.use_dsbn:

                    domain_datas_aug_views_embeddings = list(map(lambda i:(model.forward(domain_datas_aug_views[i][0], domain_labels[i]),model.forward(domain_datas_aug_views[i][1], domain_labels[i])), range(args.n_domains)))

                else:
                    domain_datas_aug_views_embeddings = list(map(lambda i:(model.forward(domain_datas_aug_views[i][0]),model.forward(domain_datas_aug_views[i][1])), range(args.n_domains)))

                domain_losses = list(map(lambda i:(crit(domain_datas_aug_views_embeddings[i][0], domain_datas_aug_views_embeddings[i][1])[0]), range(args.n_domains)))

            #print('domain_losses, ' ,domain_losses)

            loss = (1/args.n_domains)*reduce(lambda x, y: x + y, domain_losses)

            #print('loss',loss)

            loss.backward()
            optim.step()
            
            epoch_loss += loss.item()

            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                                loss=loss.item(),
                                domain_losses = list(map(lambda x:x.item(),domain_losses)),
                                time=int(time.time() - start_time))

                writer.add_scalar('loss', loss.item(), global_step=step)

                for i in range(args.n_domains):
                    writer.add_scalar('domain_{i}_loss'.format(i=i), domain_losses[i].item(), global_step=step)

                if args.optimizer == 'lars':
                    stats.update(dict(lr_weights = optim.param_groups[0]['lr'],lr_biases = optim.param_groups[1]['lr']))

                    writer.add_scalar('lr_weights', optim.param_groups[0]['lr'], global_step=step)
                    writer.add_scalar('lr_biases', optim.param_groups[1]['lr'], global_step=step)

                elif args.optimizer.startswith('adam'):
                    stats['lr'] = scheduler.get_last_lr()[0]
                    writer.add_scalar('learning_rate', stats['lr'], global_step=step)

                wandb.log(stats)
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        
        # warmup for the first 10 epochs for cos lr schedule
        if (epoch >= start_epoch + 10) and (args.optimizer.startswith('adam')):
            #scheduler.step()
            pass

        stats = dict(epoch=epoch, step=step,
                        epoch_loss=epoch_loss/len(loader),
                        time=int(time.time() - start_time))

        if args.optimizer == 'lars':
            stats.update(dict(lr_weights = optim.param_groups[0]['lr'],lr_biases = optim.param_groups[1]['lr']))

        elif args.optimizer.startswith('adam'):
            stats['lr'] = scheduler.get_last_lr()[0]

        wandb.log(stats)

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optim=optim.state_dict())
            torch.save(state, args.checkpoint_dir / epoch_id / 'checkpoint.pth')

    # save final model
    torch.save(model.state_dict(), args.checkpoint_dir / epoch_id / 'model.pth')





def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

if __name__ == '__main__':
    main()