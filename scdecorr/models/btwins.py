import torch
from torch import nn

from utils import off_diagonal
from models.densenet import densenet11,densenet21,densenet63,densenet29
from models.model_etc import TwoInputSequential, Linear, ReLU
from models.dsbn import DomainSpecificBatchNorm1d





class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


        arch=self.args.arch
        base_encoder = None
        if arch == 'densenet11':base_encoder = densenet11
        elif arch == 'densenet63':base_encoder = densenet63
        elif arch == 'densenet21':base_encoder = densenet21
        elif arch == 'densenet29':base_encoder = densenet29
        else:raise ValueError('Unknown arch {}'.format(arch))

        self.backbone = base_encoder(in_features=self.args.in_features, num_classes=self.args.dim)

        self.backbone.fc = nn.Identity()

        # projector
        sizes = [64] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):

        h1 = self.backbone(y1)
        h2 = self.backbone(y2)
        z1 = self.projector(h1)
        z1 = self.bn(z1)
        z2 = self.projector(h2)
        z2 = self.bn(z2)

        # empirical cross-correlation matrix2
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class MMDBarlowTwins(BarlowTwins):
    def __init__(self, args):
        super(MMDBarlowTwins, self).__init__(args)

    def forward(self, y, y1, y2):

        h = self.backbone(y)
        h1 = self.backbone(y1)
        h2 = self.backbone(y2)
        z1 = self.projector(h1)
        z1 = self.bn(z1)
        z2 = self.projector(h2)
        z2 = self.bn(z2)

        # empirical cross-correlation matrix2
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return (loss,h)


class DSBNBarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        arch=self.args.arch
        base_encoder = None
        if arch == 'densenet11':base_encoder = densenet11
        elif arch == 'densenet63':base_encoder = densenet63
        elif arch == 'densenet21':base_encoder = densenet21
        elif arch == 'densenet29':base_encoder = densenet29
        else:raise ValueError('Unknown arch {}'.format(arch))

        self.backbone = base_encoder(in_features=self.args.in_features, num_classes=self.args.dim)

        self.backbone.fc = nn.Identity()

        #self.bridge_bn = DomainSpecificBatchNorm1d(64,n_domains)

        # projector
        sizes = [64] + list(map(int, args.projector.split('-')))
        layers = []

        for i in range(len(sizes) - 2):
            layers.append(Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(DomainSpecificBatchNorm1d(sizes[i + 1],args.n_domains))
            layers.append(ReLU(inplace=True))

        layers.append(Linear(sizes[-2], sizes[-1], bias=False))

        self.projector = TwoInputSequential(*layers)

        # normalization layer for the representations z1 and z2
        #self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.bn = DomainSpecificBatchNorm1d(sizes[i + 1],args.n_domains,affine=False)


    def forward(self, y1, y2, domain_label):

        h1 = self.backbone(y1)
        h2 = self.backbone(y2)

        #h1, _ = self.bridge_bn(h1,domain_label)
        #h2, _ = self.bridge_bn(h2,domain_label)

        z1, _ = self.projector(h1,domain_label)
        z2, _ = self.projector(h2,domain_label)

        z1, _ = self.bn(z1,domain_label)
        z2, _ = self.bn(z2,domain_label)

        # empirical cross-correlation matrix2
        c = z1.T @ z2
        c.div_(self.args.batch_size)
        # sum the cross-correlation matrix between all gpus
        #torch.distributed.nn.functional.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class MMDDSBNBarlowTwins(DSBNBarlowTwins):
    def __init__(self, args):
        super(MMDDSBNBarlowTwins, self).__init__(args)


    def forward(self, y, y1, y2, domain_label):

        h = self.backbone(y)
        h1 = self.backbone(y1)
        h2 = self.backbone(y2)
        #h1, _ = self.bridge_bn(h1,domain_label)
        #h2, _ = self.bridge_bn(h2,domain_label)

        z1, _ = self.projector(h1,domain_label)
        z2, _ = self.projector(h2,domain_label)

        z1, _ = self.bn(z1,domain_label)
        z2, _ = self.bn(z2,domain_label)

        # empirical cross-correlation matrix2
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return (loss, h)
        #return loss