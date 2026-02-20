import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import silhouette_score,adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn import preprocessing

import os
from pathlib import Path

class Eval():
    def __init__(self, adata, feature_obsm_name='X_latent_features', batch_obs_name='batch', cell_type_obs_name='cell_type',sample_size=1,knn=30,n_jobs=-1):
        self.adata = adata
        self.feature_obsm_name = feature_obsm_name
        self.batch_obs_name = batch_obs_name
        self.cell_type_obs_name = cell_type_obs_name
        self.X = adata.obsm[feature_obsm_name]
        self.knn = knn
        self.n_jobs = n_jobs

        print('feature shape,', self.X.shape,flush=True)
        np.random.seed(0)
        self.X_sampled = self.X[np.random.choice(self.X.shape[0], np.int(self.X.shape[0]*sample_size), replace=False), :]
        print('X_sampled.shape',self.X_sampled.shape,flush=True)

        self.batch_labels = self.adata.obs[self.batch_obs_name].to_numpy()
        self.batch_labels_unique = np.unique(self.batch_labels)
        self.n_batches = self.batch_labels_unique.shape[0]
        print('Number of batches ',self.n_batches)


        self.cell_type_labels = self.adata.obs[self.cell_type_obs_name].to_numpy()
        self.cell_types = np.unique(self.cell_type_labels).tolist()
        print('cell_types',self.cell_types,flush=True)

        self.distances = None
        self.indices = None
        self.cell_type_labels_int = None



    #find knn of every query point in X_sampled after fitting a KDTree on X
    def compute_nearest_neighbors(self,knn=30,n_jobs=10):

        print('[INFO]...Computing Nearest Neighbors with k ={knn}...'.format(knn=knn),flush=True)
        nbrs = NearestNeighbors(n_neighbors=knn, algorithm='kd_tree',n_jobs=self.n_jobs).fit(self.X)
        distances, indices = nbrs.kneighbors(self.X_sampled)

        print('[INFO]...Nearest Neighbors Search Complete...',flush=True)

        return distances,indices


    def shannon_entropy_score(self):

        print('[INFO]...Computing Shannon Entropy...',flush=True)

        if self.indices is None:

            self.distances,self.indices = self.compute_nearest_neighbors(self.knn,self.n_jobs)

        total_entropy = 0

        for idx in range(len(self.indices)):
            knn_batch_labels = self.batch_labels[self.indices[idx]]
            _, knn_label_counts = np.unique(knn_batch_labels, return_counts=True)
            total_entropy = total_entropy + entropy(knn_label_counts)

        mean_entropy = total_entropy/self.X.shape[0]

        #scaling to 0-1
        mean_entropy_scaled = mean_entropy/np.log2(self.n_batches)
        print('Shannon Entropy = ',mean_entropy_scaled,flush=True)

        return mean_entropy_scaled

    #inverse simon index
    def isi_(self,labels):

        if self.indices is None:
            self.distances,self.indices = self.compute_nearest_neighbors(self.knn,self.n_jobs)

        total_isi = 0

        for idx in range(len(self.indices)):
            knn_batch_labels = labels[self.indices[idx]]
            _, knn_label_counts = np.unique(knn_batch_labels, return_counts=True)
            
            isi = 1 - np.sum((knn_label_counts/np.sum(knn_label_counts))**2)

            total_isi = total_isi + isi

        return total_isi/self.X.shape[0]

    def isi(self):

        print('[INFO]...Computing Inverse Simon Index (ISI)...',flush=True)
        isi_score = self.isi_(self.batch_labels)
        print('ISI = ',isi_score)

        return isi_score

    def cisi(self):
        print('[INFO]...Computing Cell Type Inverse Simon Index (CISI)...',flush=True)
        cisi_score = self.isi_(self.cell_type_labels)
        print('CISI = ',cisi_score)
        return cisi_score

    #w=weighted batch isi
    def wisi(self):

        print('[INFO]...Computing Weighted Inverse Simon Index (ISI)...',flush=True)

        if self.indices is None:
            self.distances,self.indices = self.compute_nearest_neighbors(self.knn,self.n_jobs)

        total_wisi = 0

        for idx in range(len(self.indices)):

            ## don't count yourself in your neighborhood
            knn_indices = self.indices[idx][1:]
            knn_batch_labels = self.batch_labels[knn_indices]
            knn_dists = self.distances[idx][1:]

            knn_batch_labels_unique, _ = np.unique(knn_batch_labels, return_counts=True)
            knn_weighted_dist_label_counts = []

            for knn_batch_label in knn_batch_labels_unique:
                knn_batch_label_dists = knn_dists[knn_batch_labels==knn_batch_label]
                batch_label_count = np.sum((1./(knn_batch_label_dists))**2)
                knn_weighted_dist_label_counts.append(batch_label_count)

            knn_weighted_dist_label_counts = np.array(knn_weighted_dist_label_counts)


            wisi = 1-np.sum((knn_weighted_dist_label_counts/np.sum(knn_weighted_dist_label_counts))**2)

            total_wisi = total_wisi + wisi

        print('WISI = ',total_wisi/self.X.shape[0],flush=True)
        return total_wisi/self.X.shape[0]
    

    #avergae silhouette width; doesnt work on more than 2 batches
    def asw(self,sil_sample_size=0.7):

        print('[INFO]...Computing Avergae Silhouette Width (ASW)...',flush=True)
        '''
        #cell_types = list(Counter(self.adata.obs[self.cell_type_obs_name]).keys())
        batch1_adata = self.adata[self.adata.obs[self.batch_obs_name]==self.batch_label1]
        batch2_adata = self.adata[self.adata.obs[self.batch_obs_name]==self.batch_label2]
        common_cell_types = set(batch1_adata.obs[self.cell_type_obs_name].tolist()).intersection(set(batch2_adata.obs[self.cell_type_obs_name].tolist()))
        '''
        
        total_asw = 0
        for cell_type in self.cell_types:
            cell_type_adata = self.adata[self.adata.obs[self.cell_type_obs_name]==cell_type]
            cell_type_asw = 1-np.abs(silhouette_score(cell_type_adata.obsm[self.feature_obsm_name],cell_type_adata.obs[self.batch_obs_name],sample_size=int(cell_type_adata.n_obs*sil_sample_size)))
            total_asw = total_asw + cell_type_asw

        print('ASW = ',total_asw/len(self.cell_types),flush=True)
        return total_asw/len(self.cell_types)


    def encode_labels_(self,labels):
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        return le.transform(labels)


    def ARI(self):
        
        print('[INFO]...Computing Adjusted Rand Index (ARI)...',flush=True)
        if self.cell_type_labels_int is None:
            self.cell_type_labels_int = self.encode_labels_(self.cell_type_labels)

        leiden_ari = adjusted_rand_score(self.cell_type_labels_int,self.adata.obs['leiden'])
        louvain_ari = adjusted_rand_score(self.cell_type_labels_int,self.adata.obs['louvain'])

        print('Leiden ARI = ',leiden_ari,flush=True)
        print('Louvain ARI = ',louvain_ari,flush=True)
        return {'leiden':float(leiden_ari),'louvain':float(louvain_ari)}


    def AMI(self):

        print('[INFO]...Computing Adjusted Mutual Information (AMI)...',flush=True)
        if self.cell_type_labels_int is None:
            self.cell_type_labels_int = self.encode_labels_(self.cell_type_labels)

        leiden_ami = adjusted_mutual_info_score(self.cell_type_labels_int,self.adata.obs['leiden'])
        louvain_ami = adjusted_mutual_info_score(self.cell_type_labels_int,self.adata.obs['louvain'])
        
        print('Leiden AMI = ',leiden_ami,flush=True)
        print('Louvain AMI = ',louvain_ami,flush=True)
        return {'leiden':float(leiden_ami),'louvain':float(louvain_ami)}


    def NMI(self):

        print('[INFO]...Computing Normalized Mutual Information (NMI)...',flush=True)
        if self.cell_type_labels_int is None:
            self.cell_type_labels_int = self.encode_labels_(self.cell_type_labels)

        leiden_nmi = normalized_mutual_info_score(self.cell_type_labels_int,self.adata.obs['leiden'])
        louvain_nmi = normalized_mutual_info_score(self.cell_type_labels_int,self.adata.obs['louvain'])

        print('Leiden NMI = ',leiden_nmi,flush=True)
        print('Louvain NMI = ',louvain_nmi,flush=True)

        return {'leiden':float(leiden_nmi),'louvain':float(louvain_nmi)}


    def cell_type_silhouette_score(self,sample_size=0.5):

        print('[INFO]...Computing Cell Type Silhouette Score (csil)...',flush=True)

        leiden_sil = silhouette_score(self.X,self.adata.obs['leiden'])
        louvain_sil = silhouette_score(self.X,self.adata.obs['louvain'])
        cell_type_sil = silhouette_score(self.X,self.adata.obs[self.cell_type_obs_name])

        print('Leiden csil = ',leiden_sil,flush=True)
        print('Louvain csil = ',louvain_sil,flush=True)
        print('Cell type sil csil = ',cell_type_sil,flush=True)


        return {'leiden':float(leiden_sil),'louvain':float(louvain_sil),'cell_type_sil':float(cell_type_sil)}


    def compute_batch_metrics(self):
        #return {'entropy':float(self.shannon_entropy_score()),'isi':float(self.isi()),'wisi':float(self.wisi()),'asw':float(self.asw())}
        return {'entropy':float(self.shannon_entropy_score()),'isi':float(self.isi()),'b_asw':float(self.asw(sil_sample_size=0.7))}
    
    def compute_cell_type_metrics(self):
        return {'ARI':self.ARI(),'AMI':self.AMI(),'NMI':self.NMI(),'csil':self.cell_type_silhouette_score(),'cisi':float(self.cisi())}

    def compute_all_metrics(self):
        cell_type_metrics = self.compute_cell_type_metrics()
        batch_metrics = self.compute_batch_metrics()

        return {'batch_metrics':batch_metrics, 'cell_type_metrics':cell_type_metrics}


