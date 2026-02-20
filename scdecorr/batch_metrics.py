#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Thu 10 Jan 2019 07:38:10 PM CST

# File Name: metrics.py
# Description:

Adapted from https://github.com/jsxlei/SCALEX
"""

import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
import pandas as pd


def batch_entropy_mixing_score_(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score
    
    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
    
    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
        
    Returns
    -------
    Batch entropy mixing score
    """
#     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i]/P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i]/P[i])/a
            entropy = entropy - adapt_p[i]*np.log(adapt_p[i]+10**-8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
            P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
                                                 [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))


def batch_entropy_mixing_score(adata, batch_obs_name='batch', cell_obs_name='cell_type', feature_obsm_name='X_emb'):
    df = pd.DataFrame(adata.obs.groupby(cell_obs_name)[batch_obs_name].nunique())
    cell_types = list(df[df[batch_obs_name] >1].index)
    cell_type_scores = []
    for cell_type in cell_types:
        adata_ = adata[adata.obs[cell_obs_name] == cell_type]
        batch_id = adata_.obs[batch_obs_name].values.astype('object').astype('str')        
        n_neighbors = 30 if adata_.shape[0] > 30 else adata_.shape[0]
        n_samples_per_pool = 30 if adata_.shape[0] > 30 else adata_.shape[0]
        if feature_obsm_name in list(adata_.obsm.keys()):
            feature = adata_.obsm[feature_obsm_name]
            if not isinstance(feature, np.ndarray):
                feature = feature.toarray()
            feature[np.isnan(feature)] = 0
            cell_type_score = adata_.shape[0]*(batch_entropy_mixing_score_(feature, batch_id, n_neighbors=n_neighbors, n_pools=10, n_samples_per_pool=n_samples_per_pool))
        else:
            cell_type_score = 0
            pass
        cell_type_scores.append(cell_type_score)
    total = adata[adata.obs[cell_obs_name].isin(cell_types)].shape[0]
    return np.sum(cell_type_scores)/total



def overcorrection_score_(emb, celltype, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    n_neighbors = min(n_neighbors, len(emb) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) - scipy.sparse.identity(emb.shape[0])
    score = 0
    celltype_ = np.unique(celltype)
    celltype_dict = celltype.value_counts().to_dict()
    N_celltype = len(celltype_)
    for t in range(n_pools):
        indices = np.random.choice(np.arange(emb.shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(celltype[kmatrix[i].nonzero()[1]][:min(celltype_dict[celltype[i]], n_neighbors)] == celltype[i]) for i in indices])
    return 1-score / float(n_pools)


def overcorrection_score(adata, cell_obs_name='cell_type', feature_obsm_name='X_emb'):
    if feature_obsm_name in list(adata.obsm.keys()):
        emb = adata.obsm[feature_obsm_name]
        if not isinstance(emb, np.ndarray):
            emb = emb.toarray()
        score = overcorrection_score_(emb, adata.obs[cell_obs_name])
    else:
        score = np.nan
    return score