import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.amp as amp

from utils import accuracy

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class InfoNceDist(nn.Module):

    def __init__(self, temper=0.1, margin=0.):
        super(InfoNceDist, self).__init__()
        self.crit = nn.CrossEntropyLoss()
        # we use margin, but not use s, because temperature works in same way as s
        self.margin = margin
        self.temp_factor = 1. / temper

    def forward(self, embs1, embs2):
        '''
        embs1, embs2: n x c, one by one pairs
            1 positive, 2n - 2 negative
            distributed mode, no need to wrap with nn.DistributedParallel
        '''
        embs1 = F.normalize(embs1, dim=1)
        embs2 = F.normalize(embs2, dim=1)
        logits, labels = InfoNceFunction.apply(embs1, embs2, self.temp_factor, self.margin)

        loss = self.crit(logits, labels.detach())
        top1, top5 = accuracy(logits, labels.detach(), topk=(1, 5))
        return (loss, top1, top5)


class InfoNceFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(device_type='cuda')
    def forward(ctx, embs1, embs2, temper_factor, margin):
        assert embs1.size() == embs2.size()
        N, C = embs1.size()
        dtype = embs1.dtype
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = embs1.device

        # gather for negative
        all_embs1 = torch.zeros(
                size=[N * world_size, C], dtype=dtype).cuda(device)
        dist.all_gather(list(all_embs1.chunk(world_size, dim=0)), embs1)
        all_embs2 = torch.zeros(
                size=[N * world_size, C], dtype=dtype).cuda(device)
        dist.all_gather(list(all_embs2.chunk(world_size, dim=0)), embs2)
        all_embs = torch.cat([all_embs1, all_embs2], dim=0)
        embs12 = torch.cat([embs1, embs2], dim=0)

        logits = torch.einsum('ac,bc->ab', embs12, all_embs)
        # mask off one sample to itself
        inds1 = torch.arange(N * 2).cuda(device)
        inds2 = torch.cat([
            torch.arange(N) + rank * N,
            torch.arange(N) + (rank + world_size) * N
            ], dim=0).cuda(device)
        logits[inds1, inds2] = -10000. # such that exp should be 0

        # label: 0~N should be N * [rank, rank + 1], N~(2N-1) should be N * [world_size * rank, world_size * (rank + 1)]
        labels = inds2.view(2, -1).flip(dims=(0,)).reshape(-1)

        # subtract margin, apply temperature
        logits[inds1, labels] -= margin
        logits *= temper_factor

        ctx.vars = inds1, inds2, embs12, all_embs, temper_factor
        return logits, labels

    @staticmethod
    @amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_logits, grad_label):
        inds1, inds2, embs12, all_embs, temper_factor = ctx.vars

        grad_logits = grad_logits * temper_factor

        grad_logits[inds1, inds2] = 0
        grad_embs12 = torch.einsum('ab,bc->ac', grad_logits, all_embs)
        grad_all_embs = torch.einsum('ab,ac->bc', grad_logits, embs12)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        N = int(all_embs.size(0) / (world_size * 2))
        grad_embs1 = grad_embs12[:N] + grad_all_embs[rank * N : (rank + 1) * N]
        grad_embs2 = grad_embs12[N:] + grad_all_embs[(rank + world_size) * N : (rank + world_size + 1) * N]

        return grad_embs1, grad_embs2, None, None