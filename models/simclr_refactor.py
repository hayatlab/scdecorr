import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal,accuracy
from models.densenet import densenet11,densenet21,densenet63,densenet29
from models.dsbn import DomainSpecificBatchNorm1d


class SimCLR(nn.Module):
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

        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        # projector
        sizes = [64] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)


    def forward(self, y):
        
        h = self.backbone(y)
        z = self.projector(h)

        return z

class DSBNSimCLR(nn.Module):
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

        n_domains=2

        self.bn = DomainSpecificBatchNorm1d(64,n_domains)

        # projector
        sizes = [64] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.projector = nn.Sequential(*layers)



    def forward(self, y, domain_label):
        
        h = self.backbone(y)
        h, _ = self.bn(h,domain_label)

        z = self.projector(h)

        return z