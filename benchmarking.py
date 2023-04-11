import numpy as np
import scanpy as sc
import os
from downstream import DownstreamTaskRun
import scib.integration as integration
import yaml
import tempfile
import warnings
from downstream_utils import gen_obsmname_task_tuples_

class Benchmark():
    def __init__(self,dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name="cell_type",batch_obs_name='batch'):

        self.dataset_root_dir=dataset_root_dir

        self.adata_fname = adata_fname
        self.adata_path=os.path.join(self.dataset_root_dir,self.adata_fname)
        self.adata = sc.read_h5ad(self.adata_path)
        try:
            self.adata.obsm['X_pca']
        except KeyError:
            sc.tl.pca(self.adata)

        print(self.adata,flush=True)

        self.batch_obs_name = batch_obs_name
        self.cell_class_obs_name = cell_class_obs_name
        self.dataset_name = dataset_name
        self.task_list = ['cluster','batch_correct','classify','all']


    # loads features from the checkpoint, 
    # and writes to a obsm "feature_obsm_name" of the adata 
    def load_features_from_checkpoint(self,checkpoint_root,data_obs_name=None,in_features=2000,arch='densenet21'):

        import utils

        features = utils.extract_features_from_adata(self.adata,checkpoint_root,data_obs_name,in_features,arch)

        print('features.shape ',features.shape)
        return features



    def compute_harmony_features_(self):

        from harmony import harmonize

        print('[INFO]...Computing harmony features...')
        print('[INFO]...Computing X_pca')
        adata = self.adata.copy()

        sc.tl.pca(adata)
        features = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=self.batch_obs_name)
        print('features shape',features.shape)
        print('[INFO]...Computing harmony features finished...')
        return features


    def compute_scVI_features_(self,max_epochs=100):
        print('[INFO]...Computing scVI features...')
        adata = self.adata.copy()
        scvi_model = integration.scvi(adata, self.batch_obs_name, hvg=None, return_model=True, max_epochs=max_epochs)
        features = scvi_model.get_latent_representation()
        print('features shape',features.shape)
        print('[INFO]...Computing scVI features finished...')

        return features



    def compute_bbknn_features_(self):

        print('[INFO]...Computing BBKNN features...')
        adata = self.adata.copy()


        adata = integration.bbknn(adata, self.batch_obs_name, hvg=None)

        print('\n....\n\n\n\n.....\n')

        features = adata.X


        #features = adata.obsm['X_emb']

        #print('features',features)
        print('features shape',features.shape)

        print('[INFO]...Computing BBKNN features finished...')
        return features


    def compute_scanorama_features_(self):

        print('[INFO]...Computing Scanorama features...')
        adata = self.adata.copy()

        try:
            del adata.obsm['X_emb']
        except KeyError:
            pass

        adata = integration.scanorama(adata, self.batch_obs_name, hvg=None)

        print('\n....\n\n\n\n.....\n')
        print('scan_adata ',adata)

        features = adata.obsm['X_emb']
        print('features shape',features.shape)
        print(features)

        print('[INFO]...Computing Scanorama features finished...')
        return features


    def compute_liger_features_(self):
        print('[INFO]...Computing Liger features...')

        import pyliger

        adata = self.adata.copy()
        print(adata)

        batch_cats = adata.obs[self.batch_obs_name].cat.categories

        bdata = adata.copy()
        # Pyliger normalizes by library size with a size factor of 1
        # So here we give it the count data
        bdata.X = bdata.layers["counts"]

        # List of adata per batch
        adata_list = [bdata[bdata.obs[self.batch_obs_name] == b].copy() for b in batch_cats]
        for i, ad in enumerate(adata_list):
            ad.uns["sample_name"] = batch_cats[i]
            # Hack to make sure each method uses the same genes
            ad.uns["var_gene_idx"] = np.arange(bdata.n_vars)


        liger_data = pyliger.create_liger(adata_list, remove_missing=False, make_sparse=False)
        # Hack to make sure each method uses the same genes
        liger_data.var_genes = bdata.var_names
        pyliger.normalize(liger_data)
        pyliger.scale_not_center(liger_data)
        pyliger.optimize_ALS(liger_data, k=30)
        pyliger.quantile_norm(liger_data)


        adata.obsm["LIGER"] = np.zeros((adata.shape[0], liger_data.adata_list[0].obsm["H_norm"].shape[1]))
        for i, b in enumerate(batch_cats):
            adata.obsm["LIGER"][adata.obs[self.batch_obs_name] == b] = liger_data.adata_list[i].obsm["H_norm"]

        print('[INFO]...Computing Liger features finished...')

        print(adata)
        features = adata.obsm['LIGER']
        print('features shape',features.shape)
        print(features)
        return features


    def compute_scanvi_features_(self):
        print('[INFO]...Computing scANVI features...')
        adata = self.adata.copy()

        try:
            del adata.obsm['X_emb']
        except KeyError:
            pass
        
        print(adata)
        adata = integration.scanvi(self.adata, self.batch_obs_name, hvg=None)

        print('\n....\n\n\n\n.....\n')
        print(adata)

        features = adata.obsm['X_emb']
        print('features shape',features.shape)
        print(features)

        print('[INFO]...Computing scANVI features finished...')
        return features


    def compute_trvae_features_(self):

        print('[INFO]...Computing trvaep features...')
        adata = self.adata.copy()

        try:
            del adata.obsm['X_emb']
        except KeyError:
            pass

        adata = integration.trvae(adata, self.batch_obs_name, hvg=None)

        print('\n....\n\n\n\n.....\n')

        features = adata.obsm['X_emb']

        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing trvaep features finished...')
        return features



    def compute_desc_features_(self,tmp_dir=None,ncores=15):

        print('[INFO]...Computing DESC features...')

        import desc


        if tmp_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            tmp_dir = temp_dir.name

        # Set number of CPUs to all available
        if ncores is None:
            ncores = os.cpu_count()

        adata_out = self.adata.copy()

        adata_out = desc.scale_bygroup(adata_out, groupby=self.batch_obs_name, max_value=6)

        res = 0.8

        adata_out = desc.train(
            adata_out,
            dims=[self.adata.shape[1], 128, 32],
            tol=0.001,
            n_neighbors=10,
            batch_size=256,
            louvain_resolution=res,
            save_encoder_weights=False,
            save_dir=tmp_dir,
            do_tsne=False,
            use_GPU=True,
            GPU_id=None,
            num_Cores=ncores,
            use_ae_weights=False,
            do_umap=False,
        )

        features = adata_out.obsm["X_Embeded_z" + str(res)]
        print('features shape',features.shape)


        print('[INFO]...Computing DESC features finished...')
        return features



    def compute_features(self,method_id,save_features=True,eval=False,**kwargs):

        method_name,version = method_id.strip(' ').split('_')

        feature_obsm_name,method_root_dir = self.compute_init_(method_id)


        if method_name == 'harmony':
            features = self.compute_harmony_features_()

        elif method_name == 'scVI':

            max_epochs = kwargs.get('scvi_max_epochs',100)
            with open(os.path.join(method_root_dir,'config.yaml'),'w') as f:
                yaml.dump({'epochs':max_epochs},f)

            features = self.compute_scVI_features_(max_epochs=max_epochs)

        elif method_name == 'bbknn':
            features = self.compute_bbknn_features_()
        
        elif method_name == 'scanorama':
            features = self.compute_scanorama_features_()

        elif method_name == 'liger':
            features = self.compute_liger_features_()

        elif method_name == 'sctwins-dsbn':
            checkpoint_root = kwargs.get('checkpoint_root')
            if not checkpoint_root:
                raise Exception('kwarg checkpoint_root not provided!')

            data_obs_name = kwargs.get('data_obs_name')
            in_features = kwargs.get('in_features',2000)
            arch = kwargs.get('arch','densenet21')

            with open(os.path.join(method_root_dir,'config.yaml'),'w') as f:
                yaml.dump({'checkpoint_root':checkpoint_root,'data_obs_name':data_obs_name,\
                           'in_features':in_features,'arch':arch},f)

            features = self.load_features_from_checkpoint(checkpoint_root,data_obs_name,in_features,arch)


        self.adata.obsm[feature_obsm_name] = features

        print(self.adata)

        if save_features:
            print('[INFO]...Writing features to obsm {obsm}...'.format(obsm=feature_obsm_name))

            self.adata.write(self.adata_path)

        if eval:

            print('[INFO]...Eval started...')
            if kwargs['task'] not in self.task_list:
                raise Exception('Illegal Argument task: task should be in ',self.task_list)
            self.eval_(method_id,feature_obsm_name,task=kwargs['task'])

            print('[INFO]...Eval finished...')

        return self.adata




    def compute_init_(self,method_id):

        feature_obsm_name = 'X_emb_{method_id}'.format(method_id=method_id)

        method_name,version = method_id.strip(' ').split('_')

        method_root_dir = os.path.join(self.dataset_root_dir,'method',method_name,version)
        if not os.path.exists(method_root_dir):
            os.makedirs(method_root_dir,exist_ok=True)

        return feature_obsm_name,method_root_dir



    def eval_(self,method_id,feature_obsm_name=None,task='all'):

        feature_obsm_name = [feature_obsm_name] if feature_obsm_name is not None else None

        taskrun = DownstreamTaskRun(dataset_root_dir=self.dataset_root_dir,adata_fname=self.adata_fname,method_ids=[method_id],task=task,dataset_name=self.dataset_name,\
                            cell_class_obs_name=self.cell_class_obs_name,batch_obs_name=self.batch_obs_name,feature_obsm_names=feature_obsm_name)
        taskrun.run(save_adata=False,classify_model="xgb",n_jobs=-1)



    def eval_benchmark_multicore(self,methods_tasks_list,n_jobs=-1):

        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(delayed(self.eval_)(method_id,feature_obsm_name,task)\
                                 for (method_id,feature_obsm_name,task) in gen_obsmname_task_tuples_(methods_tasks_list))


    def eval_benchmark(self,methods_tasks_list):

        for (method_id,feature_obsm_name,task) in gen_obsmname_task_tuples_(methods_tasks_list):
            print('[INFO]...Task {task} on {method_id} started...'.format(task=task,method_id=method_id))
            self.eval_(method_id,feature_obsm_name,task)
            print('[INFO]...Task {task} on {method_id} finished...'.format(task=task,method_id=method_id))










