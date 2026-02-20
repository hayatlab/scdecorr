import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing










def create_adata(X,n_obs,n_vars,obs_names,obs_data_list):
    counts = csr_matrix(X,dtype=np.float32)
    adata=ad.AnnData(counts)
    adata.obs_names = [f"Cell_{i:d}" for i in range(n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(n_vars)]

    for i in range(len(obs_names)):
        obs_name=obs_names[i]
        obs_data=obs_data_list[i]
        print('obs_name ',obs_name)
        adata.obs[obs_name]=pd.Categorical(obs_data)
    
    return adata


##DO NOT USE merge_adatas. use concatby anndata
##TO be DELETED
def merge_adatas(adatas, cell_type_obs_names):

    adatas_dict = {}

    for i in range(len(adatas)):
        adata = adatas[i]
        cell_type_obs = cell_type_obs_names[i]
        X = adata.X.toarray()
        n_obs,n_vars = adata.n_obs,adata.n_vars 
        cell_types = adata.obs[cell_type_obs].tolist()
        adata = create_adata(X,n_obs,n_vars,['cell_type'],[cell_types])
        adatas_dict[str(i)] = adata

    return ad.concat(adatas_dict,index_unique="_",label='dataset')
    #adatas = {'a':adata_a,'b':adata_b}


def map_cell_types_to_reference(adata_qry,cell_types_qry_ref_map,cell_type_obs_name_qry,ref_id):
    # map cell type names of all cells of query adata according to  
    # a dict map of query to reference cell types
    #cell_types_qry_ref_map = {'04. Ventricular Cardiomyocyte I':'Ventricular_Cardiomyocyte','03. Atrial Cardiomyocyte':'Ventricular_Cardiomyocyte','06. Ventricular Cardiomyocyte II':'Ventricular_Cardiomyocyte','11. Adipocyte':'Adipocytes','12. Cytoplasmic Cardiomyocyte II':'Ventricular_Cardiomyocyte','08. Macrophage':'Macrophage','01. Fibroblast I':'Fibroblast','13. Vascular Smooth Muscle':'Smooth_muscle_cells','07. Pericyte':'Pericytes','02. Fibroblast II':'Fibroblast','09. Endothelium I':'Endothelial','14. Fibroblast III':'Fibroblast','10. Endothelium II':'Endothelial','16. Neuronal':'Neuronal','15. Ventricular Cardiomyocyte III':'Ventricular_Cardiomyocyte','17. Lymphocyte':'Lymphoid','05. Cytoplasmic Cardiomyocyte I':'Ventricular_Cardiomyocyte'}
    # adds a new obs column in adata. returns adata

    cell_types_qry = adata_qry.obs[cell_type_obs_name_qry].tolist()
    cell_types_qry_mapped = [cell_types_qry_ref_map.get(cell_type,cell_type) for cell_type in cell_types_qry]
    adata_qry.obs['cell_types_mapped_'+str(ref_id)] = cell_types_qry_mapped

    return adata_qry


def preprocess_adata(adata,min_cells=3,layer=None,normalize=False,log1p=False,n_hvg_genes=2000,flavor='seurat_v3',batch_key='batch'):

    sc.pp.filter_genes(adata,min_cells=min_cells)
    print('adata.n_vars',adata.n_vars)

    if layer is not None:
        print('type layers',type(adata.layers[layer]))
        adata.X = adata.layers[layer].copy()


    tmp_bdata = adata.copy()

    if normalize:
        sc.pp.normalize_total(tmp_bdata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(tmp_bdata)


    if batch_key is not None:
        sc.pp.highly_variable_genes(tmp_bdata, n_top_genes = n_hvg_genes, flavor = flavor, batch_key= batch_key)
    else:
        sc.pp.highly_variable_genes(tmp_bdata, n_top_genes = n_hvg_genes, flavor = flavor)

    adata = tmp_bdata[:, tmp_bdata.var.highly_variable]
    print('adata.n_vars',adata.n_vars)
    return adata


def batch_wise_scale(adata,batch_obs):
    
    batches = set(adata.obs[batch_obs].tolist())
    print(batches)

    for batch in batches:

        X_batch = adata[adata.obs[batch_obs]==batch].X.copy()

        X_batch=(X_batch-X_batch.mean(axis=0))/X_batch.std(axis=0)
        X_batch=np.nan_to_num(X_batch,False)
        adata[adata.obs[batch_obs]==batch].X = X_batch

    return adata





