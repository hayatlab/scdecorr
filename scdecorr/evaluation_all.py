import numpy as np

from scib.metrics import silhouette_batch, kBET, graph_connectivity, ari, nmi, silhouette, isolated_labels_f1, isolated_labels_asw
import scanpy as sc
import pandas as pd

import os
from pathlib import Path
from scdecorr.evaluation_scratch import Eval
from scdecorr.batch_metrics import batch_entropy_mixing_score, overcorrection_score

class EvalMetrics():
    def __init__(self, adata, feature_obsm_name='X_emb', batch_obs_name='batch', cell_type_obs_name='cell_type', sample=1, knn=15, n_jobs=-1):

        ##self.eval_scratch = Eval(adata, feature_obsm_name, batch_obs_name, cell_type_obs_name,sample_size=sample,knn=knn,n_jobs=n_jobs)

        self.adata = adata
        self.feature_obsm_name = feature_obsm_name
        self.batch_obs_name = batch_obs_name
        self.cell_type_obs_name = cell_type_obs_name
        #self.batch_metrics = ['batch_asw','kbet','graph_connect']
        self.batch_metrics = ['batch_asw', 'graph_connect', 'batch_entropy_mix', 'over_correction']
        self.excl_from_mean = {'batch_metrics': ['over_correction']}
        #self.bioconsv_metrics = ['ari','nmi','silhouette','isolated_ASW','isolated_labels_F1','cisi']
        self.bioconsv_metrics = ['ari','nmi','silhouette']
        #self.bioconsv_metrics = ['ari','nmi','silhouette','isolated_asw','isolated_f1']
        #isolated_f1 takes a lot of compute time for large datasets


    ######################################
    #######Batch Correction Metrics#######
    ######################################

    def entropy(self):
        print('[INFO]...Computing Shannon Entropy...',flush=True)
        entropy_score = self.eval_scratch.shannon_entropy_score()
        print('entropy_score = ',entropy_score,flush=True)
        return entropy_score


    def batch_asw(self):
        print('[INFO]...Computing Batch Average Silhouette Width (ASW)...',flush=True)
        batch_asw_score = silhouette_batch(self.adata,self.batch_obs_name,self.cell_type_obs_name,self.feature_obsm_name)
        print('batch_asw_score = ',batch_asw_score,flush=True)
        return batch_asw_score


    def wisi(self):
        print('[INFO]...Computing Weighted Inverse Simpson Index...',flush=True)
        wisi_score = self.eval_scratch.wisi()
        print('wisi_score = ',wisi_score,flush=True)
        return wisi_score


    def kbet(self):
        print('[INFO]...Computing k-nearest neighbour batch effect test (kBET) score...',flush=True)
        kbet_score = kBET(self.adata,self.batch_obs_name,self.cell_type_obs_name,type_="embed",embed=self.feature_obsm_name)
        print('kbet_score = ',kbet_score,flush=True)
        return kbet_score


    def graph_connect(self):
        adata = self.adata.copy()  # avoid view issues
        # 1. Remove old graph completely
        for key in ["connectivities", "distances"]:
            if key in adata.obsp:
                del adata.obsp[key]
        if "neighbors" in adata.uns:
            del adata.uns["neighbors"]

        print('[INFO]...Computing Graph Connectivity score...',flush=True)
        #neighbors_key = '{use_rep}_neighbor'.format(use_rep=self.feature_obsm_name)
        sc.pp.neighbors(self.adata, use_rep=self.feature_obsm_name)
        gc = graph_connectivity(self.adata, self.cell_type_obs_name)
        print('graph_connect = ',gc,flush=True)
        return gc


    def batch_entropy_mix(self):
        print('[INFO]...Computing Batch Entropy Mixing score from SCALEX...',flush=True)
        batch_entropy_mix_score = batch_entropy_mixing_score(
            self.adata, batch_obs_name=self.batch_obs_name, 
            cell_obs_name=self.cell_type_obs_name, feature_obsm_name=self.feature_obsm_name
        )
        print('batch_entropy_mix = ', batch_entropy_mix_score,flush=True)
        return batch_entropy_mix_score

    def over_correction(self):
        print('[INFO]...Computing OverCorrection score from SCALEX...',flush=True)
        oc_score = overcorrection_score(self.adata, cell_obs_name=self.cell_type_obs_name, feature_obsm_name=self.feature_obsm_name)
        print('over_correction = ', oc_score,flush=True)
        return oc_score


    def compute_batch_metrics(self,metrics=None):
        batch_metrics = metrics if metrics is not None else self.batch_metrics
        print('batch_metrics from eval_all', metrics)
        metrics_dict = {}
        scores = []
        for batch_metric in batch_metrics:
            if batch_metric == 'entropy':
                score = self.entropy()
            elif batch_metric == 'batch_asw':
                score = self.batch_asw()
            elif batch_metric == 'wisi':
                score = self.wisi()
            elif batch_metric == 'kbet':
                score = self.kbet()
            elif batch_metric == 'graph_connect':
                score = self.graph_connect()
            elif batch_metric == 'batch_entropy_mix':
                score = self.batch_entropy_mix()
            elif batch_metric == 'over_correction':
                score = self.over_correction()
            else:
                raise Exception('WrongBatchMetricName: batch_metric={batch_metric} does not exist!'.format(batch_metric=batch_metric))
            metrics_dict[batch_metric] = score
            if batch_metric not in self.excl_from_mean['batch_metrics']:
                scores.append(score)
        print('Batch metrics_dict',metrics_dict)
        metrics_dict['score'] = np.mean(scores)
        return pd.DataFrame(metrics_dict,index=[0])


    #############################################
    #######Biological Conservation Metrics#######
    #############################################


    def ARI(self):
        print('[INFO]...Computing Adjusted Rand Index (ARI)...',flush=True)
        leiden_ari = ari(self.adata,'leiden_{use_rep}'.format(use_rep=self.feature_obsm_name),self.cell_type_obs_name)
        louvain_ari = ari(self.adata,'louvain_{use_rep}'.format(use_rep=self.feature_obsm_name),self.cell_type_obs_name)
        mean_ari = (leiden_ari+louvain_ari)/2
        max_ari = max(leiden_ari,louvain_ari)
        print('Leiden ARI = ',leiden_ari,flush=True)
        print('Louvain ARI = ',louvain_ari,flush=True)
        return {'leiden':leiden_ari,'louvain':louvain_ari,'mean':mean_ari,'max':max_ari}


    def NMI(self):
        print('[INFO]...Computing Normalized Mutual Information (NMI)...',flush=True)
        leiden_nmi = nmi(self.adata,'leiden_{use_rep}'.format(use_rep=self.feature_obsm_name),self.cell_type_obs_name)
        louvain_nmi = nmi(self.adata,'louvain_{use_rep}'.format(use_rep=self.feature_obsm_name),self.cell_type_obs_name)
        mean_nmi = (leiden_nmi+louvain_nmi)/2
        max_nmi = max(leiden_nmi,louvain_nmi)
        print('Leiden NMI = ',leiden_nmi,flush=True)
        print('Louvain NMI = ',louvain_nmi,flush=True)
        return {'leiden':leiden_nmi,'louvain':louvain_nmi,'mean':mean_nmi,'max':max_nmi}



    def ARI_new(self, res=0.2):
        print(f'[INFO]...Computing Adjusted Rand Index (ARI) on res={res}...',flush=True)
        leiden_ari = ari(self.adata, f'leiden_res={res:.1f}_{self.feature_obsm_name}', self.cell_type_obs_name)
        print('Leiden ARI = ',leiden_ari,flush=True)
        return {'leiden':leiden_ari}


    def NMI_new(self, res=0.2):
        print(f'[INFO]...Computing Normalized Mutual Information (NMI) on res={res}...',flush=True)
        leiden_ari = nmi(self.adata, f'leiden_res={res:.1f}_{self.feature_obsm_name}', self.cell_type_obs_name)
        print('Leiden NMI = ',leiden_ari,flush=True)
        return {'leiden':leiden_ari}



    def cell_type_silhouette_score(self):

        print('[INFO]...Computing Cell Type Silhouette Score (csil)...',flush=True)

        cell_type_sil = silhouette(self.adata, self.cell_type_obs_name, self.feature_obsm_name, scale=True)

        print('Cell type sil csil = ',cell_type_sil,flush=True)

        return cell_type_sil


    def isolated_ASW(self):

        print('[INFO]...Computing Isolated label score ASW...',flush=True)

        il_asw = isolated_labels_asw(self.adata,self.cell_type_obs_name,self.batch_obs_name,self.feature_obsm_name)

        print('Isolated label score ASW = ',il_asw,flush=True)

        return il_asw

    def isolated_F1(self):
        print('[INFO]...Computing Isolated label score F1...',flush=True)

        il_f1 = isolated_labels_f1(self.adata,self.cell_type_obs_name,self.batch_obs_name,self.feature_obsm_name)

        print('Isolated label score F1 = ',il_f1,flush=True)

        return il_f1


    def cisi(self):

        print('[INFO]...Computing Cell Type Inverse Simpson Index...',flush=True)
        cisi_score = self.eval_scratch.cisi()
        print('cisi_score_score = ',cisi_score,flush=True)
        return cisi_score


    def compute_cell_type_metrics(self, metrics=None):

        cell_type_metrics = metrics if metrics is not None else self.bioconsv_metrics
        metrics_dict = {}

        scores_mean_avg = 0
        scores_mean_max = 0

        for cell_type_metric in cell_type_metrics:
            if cell_type_metric == 'ari':
                score = self.ARI()
                metrics_dict['leiden.ARI'] = score['leiden']
                metrics_dict['louvain.ARI'] = score['louvain']
                scores_mean_avg += score['mean']
                scores_mean_max += score['max']

            elif cell_type_metric == 'nmi':
                score = self.NMI()
                metrics_dict['leiden.NMI'] = score['leiden']
                metrics_dict['louvain.NMI'] = score['louvain']
                scores_mean_avg += score['mean']
                scores_mean_max += score['max']

            elif cell_type_metric == 'silhouette':
                score = self.cell_type_silhouette_score()
                metrics_dict['silhouette'] = score
                scores_mean_avg += score
                scores_mean_max += score

            elif cell_type_metric == 'isolated_asw':
                score = self.isolated_ASW()
                metrics_dict['isolated_asw'] = score
                scores_mean_avg += score
                scores_mean_max += score

            elif cell_type_metric == 'isolated_f1':
                score = self.isolated_F1()
                metrics_dict['isolated_f1'] = score
                scores_mean_avg += score
                scores_mean_max += score


            elif cell_type_metric == 'cisi':
                score = self.cisi()
                metrics_dict['cisi'] = score

                scores_mean_avg += score
                scores_mean_max += score
            else:
                raise Exception('WrongCellTypeMetricName: cell_type_metric={cell_type_metric} does not exist!'.format(cell_type_metric=cell_type_metric))

        scores_mean_avg = scores_mean_avg/len(cell_type_metrics)
        scores_mean_max = scores_mean_max/len(cell_type_metrics)

        print('Cell type metrics_dict',metrics_dict)

        metrics_dict['score_avg'] = scores_mean_avg
        metrics_dict['score_max'] = scores_mean_max

        return pd.DataFrame(metrics_dict,index=[0])


    def compute_cell_type_metrics_new(self, metrics=None, res=0.2):

        cell_type_metrics = metrics if metrics is not None else self.bioconsv_metrics
        metrics_dict = {}

        scores_mean = 0

        for cell_type_metric in cell_type_metrics:
            if cell_type_metric == 'ari':
                score = self.ARI_new(res=res)
                metrics_dict['leiden.ARI'] = score['leiden']
                scores_mean += score['leiden']

            elif cell_type_metric == 'nmi':
                score = self.NMI_new(res=res)
                metrics_dict['leiden.NMI'] = score['leiden']
                scores_mean += score['leiden']

            elif cell_type_metric == 'silhouette':
                score = self.cell_type_silhouette_score()
                metrics_dict['silhouette'] = score
                scores_mean += score

            elif cell_type_metric == 'isolated_asw':
                score = self.isolated_ASW()
                metrics_dict['isolated_asw'] = score
                scores_mean += score

            elif cell_type_metric == 'isolated_f1':
                score = self.isolated_F1()
                metrics_dict['isolated_f1'] = score
                scores_mean += score

            elif cell_type_metric == 'cisi':
                score = self.cisi()
                metrics_dict['cisi'] = score
                scores_mean += score
            else:
                raise Exception('WrongCellTypeMetricName: cell_type_metric={cell_type_metric} does not exist!'.format(cell_type_metric=cell_type_metric))

        scores_mean = scores_mean/len(cell_type_metrics)

        print('Cell type metrics_dict',metrics_dict)

        metrics_dict['score_avg'] = scores_mean

        return pd.DataFrame(metrics_dict,index=[0])



    def compute_all_metrics(self):
        cell_type_metrics = self.compute_cell_type_metrics()
        batch_metrics = self.compute_batch_metrics()

        return {'batch_metrics':batch_metrics, 'cell_type_metrics':cell_type_metrics}


