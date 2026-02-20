from prettytable import PrettyTable
import torch
import os
import shutil
import yaml

import numpy as np
import torch.utils.data
from scdecorr.models.densenet import *
from scdecorr.dataset import SingleBatchDataset
import scanpy as sc


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def get_model(arch, in_features, num_classes=64, return_feature=False):
    model = None

    kwargs = {'num_classes': num_classes, 'return_feature':return_feature}
    if arch == 'densenet11':
        model = densenet11(in_features, **kwargs)
    elif arch == 'densenet63':
        model = densenet63(in_features, **kwargs)
    elif arch == 'densenet21':
        model = densenet21(in_features, **kwargs)
    elif arch == 'densenet29':
        model = densenet29(in_features, **kwargs)
    else:
        raise ValueError("Unknown arch {}".format(arch))

    return model

def load_pretrained_model_from_checkpoint(arch, in_features, checkpoint, num_classes=64, return_feature=True):
    model = get_model(arch, in_features, num_classes=num_classes, return_feature=return_feature)
    #print('model ',model)
    #model = DenseNetSimCLR(args={"arch":arch,"in_features":in_features,"dim":num_classes})

    for _, param in model.named_parameters():
        #if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

    state_dict = torch.load(checkpoint, map_location="cpu")['model']
    state_dict = {k.lstrip('module.backbone.'): v for k, v in state_dict.items() if k.startswith('module.backbone')}
    #model_dict.update(state_dict) 
    #print('state_dict.model',list(state_dict.keys()))
    msg = model.load_state_dict(state_dict, strict=False)
    #torch.save(model.state_dict(),save_path)
    #print('msg',msg)
    #assert msg.missing_keys == ["fc.weight", "fc.bias"] and msg.unexpected_keys == []

    return model

def load_pretrained_model(arch, in_features, checkpoint, num_classes=64, return_feature=True):
    model = get_model(arch, in_features, num_classes=num_classes, return_feature=return_feature)
    print(model)

    for name, param in model.named_parameters():
        #if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

    state_dict = torch.load(checkpoint, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print('msg',msg)
    #assert msg.missing_keys == ["fc.weight", "fc.bias"] and msg.unexpected_keys == []
    return model


def extract_features(model, X):
    # set to eval mode
    model.eval()
    
    val_dataset = SingleBatchDataset(X)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=2)

    features = []
    
    with torch.no_grad():
        for x, _ in val_loader:
            output = model(x)
            features.extend(output.detach().cpu().numpy())
            
    features = np.asarray(features)
    
    return features


def extract_features_from_adata(adata,checkpoint_root,data_obs_name=None,in_features=2000,arch='densenet21'):

    if data_obs_name is not None:
        X = adata.obsm[data_obs_name]
    else:
        try:
            X=adata.X.toarray()
        except AttributeError:
            X=adata.X

    #print(X.mean(axis=0))
    print('arch, in_features, X.shape',arch,in_features,X.shape[1])
    assert in_features == X.shape[1]

    checkpoint=os.path.join(checkpoint_root,'checkpoint.pth')

    model = load_pretrained_model_from_checkpoint(arch, in_features, checkpoint, return_feature=True)
    features = extract_features(model, X)
    #print('features',features,flush=True)
    print('Features shape', features.shape,flush=True)

    return features


def extract_features_and_add_to_adata(adata,in_features,arch=None,checkpoint=None,X_obsm_name=None,X_feature_out_obsm_name=None,strategy='pca'):

    if strategy == 'cl':
        features = extract_features_from_adata(adata,in_features,arch,checkpoint,X_obsm_name)
        adata.obsm[X_feature_out_obsm_name] = features
    
    elif strategy == 'pca':
        sc.tl.pca(adata)
    
    return adata

