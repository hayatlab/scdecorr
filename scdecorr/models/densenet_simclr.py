from torch import nn

from models.densenet import densenet11,densenet21,densenet63,densenet29


class DenseNetSimCLR(nn.Module):
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
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(self.backbone.fc,nn.Linear(dim_mlp, dim_mlp,bias=False),nn.ReLU(), nn.Linear(dim_mlp, dim_mlp,bias=False))


    def forward(self, x):
        return self.backbone(x)