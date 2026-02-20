import numpy as np
import scanpy as sc
import os
from scdecorr.downstream import DownstreamTaskRun
import scib.integration as integration
import yaml
import tempfile
import warnings
from pathlib import Path
import scdecorr.downstream_utils as dutils
import numpy as np
import scarches as sca

from scdecorr.seeding import seed_everything


class Benchmark():
    def __init__(self,dataset_root_dir,adata_fname,dataset_name,adata=None,cell_class_obs_name="cell_type",batch_obs_name='batch', seed=42):

        self.dataset_root_dir=dataset_root_dir
        self.seed = seed
        print(f'Using seed={self.seed}')
        seed_everything(seed=seed, cuda=True, workers=True)

        self.adata_fname = adata_fname
        self.adata_path=os.path.join(self.dataset_root_dir,self.adata_fname)
        self.adata = adata

        self.batch_obs_name = batch_obs_name
        self.cell_class_obs_name = cell_class_obs_name
        self.dataset_name = dataset_name
        self.task_list = ['cluster','batch_correct','classify','all']
        self.methods_list = ['scdecorr','scVI','harmony','scanorama','liger','bbknn']


    # loads features from the checkpoint, 
    # and writes to a obsm "feature_obsm_name" of the adata 
    def load_features_from_checkpoint(self,checkpoint_root,data_obs_name=None,in_features=2000,arch='densenet21'):

        import scdecorr.utils as utils

        features = utils.extract_features_from_adata(self.adata,checkpoint_root,data_obs_name,in_features,arch)

        print('features.shape ',features.shape)
        return features


    def compute_init_(self,method_id):
        
        if self.adata is None:
            self.adata = sc.read_h5ad(self.adata_path)
        print(self.adata,flush=True)

        feature_obsm_name = 'X_emb_{method_id}'.format(method_id=method_id)

        method_name,version = method_id.strip(' ').split('_')

        method_root_dir = os.path.join(self.dataset_root_dir,'method',method_name,version)
        if not os.path.exists(method_root_dir):
            os.makedirs(method_root_dir,exist_ok=True)

        return feature_obsm_name,method_root_dir


    def compute_harmony_features_(self):

        from harmony import harmonize
        print('[INFO]...Computing harmony features...')
        adata = self.adata.copy()
        adata = dutils.convert_to_sparse_(adata)
        print('[INFO]...Computing X_pca')
        sc.tl.pca(adata)
        features = harmonize(
            X=adata.obsm["X_pca"], 
            df_obs=adata.obs, 
            batch_key=self.batch_obs_name, 
            random_state=self.seed, 
            use_gpu=True
        )
        print('features shape',features.shape)
        print('[INFO]...Computing harmony features finished...')
        return features


    def compute_harmony_features_new_(self):
        import scanpy.external as sce
        if 'X_pca' in self.adata.obsm_keys():
            self.adata.obsm.pop('X_pca')
        print('[INFO]...Computing X_pca')
        sc.pp.pca(self.adata)
        sce.pp.harmony_integrate(self.adata, key=self.batch_obs_name, random_state=self.seed)
        features = self.adata.obsm['X_pca_harmony']
        print('features shape',features.shape)
        print('[INFO]...Computing harmony features finished...')
        return features


    def compute_scVI_features_(self,max_epochs=100):
        print('[INFO]...Computing scVI features...')
        adata = self.adata.copy()
        adata = dutils.convert_to_sparse_(adata)
        scvi_model = integration.scvi(adata, self.batch_obs_name, hvg=None, return_model=True, max_epochs=max_epochs)
        features = scvi_model.get_latent_representation()
        print('features shape',features.shape)
        print('[INFO]...Computing scVI features finished...')
        return features


    def compute_bbknn_features_(self):
        print('[INFO]...Computing BBKNN features...')
        adata = self.adata.copy()
        adata = dutils.convert_to_sparse_(adata)
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
        adata = dutils.convert_to_sparse_(adata)
        try:
            del adata.obsm['X_emb']
        except KeyError:
            pass
        adata = integration.scanorama(adata, self.batch_obs_name, hvg=None, seed=self.seed)
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
        print('type(adata.X) ',type(self.adata.X))
        adata = dutils.convert_to_sparse_(adata)
        print('type(adata.X) ',type(adata.X))
        print('type(adata.layers) ',type(adata.layers['counts']))
        batch_cats = adata.obs[self.batch_obs_name].cat.categories
        bdata = adata.copy()
        # Pyliger normalizes by library size with a size factor of 1
        # So here we give it the count data
        bdata.X = bdata.layers["counts"]
        print('type(bdata.X) ',type(bdata.X))
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
        adata = dutils.convert_to_sparse_(adata)
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
        adata = integration.trvaep(adata, self.batch_obs_name, hvg=None)
        print('\n....\n\n\n\n.....\n')
        features = adata.obsm['X_emb']
        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing trvaep features finished...')
        return features


    def compute_trvae_features_new_(self):
        print('[INFO]...Computing trvae features...')
        adata = self.adata.copy()
        trvae = sca.models.TRVAE(
            adata=adata,
            condition_key=self.batch_obs_name,
            hidden_layer_sizes=[128, 128],
        )
        early_stopping_kwargs = {
            "early_stopping_metric": "val_unweighted_loss",
            "threshold": 0,
            "patience": 20,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }
        trvae.train(
            n_epochs=100,
            alpha_epoch_anneal=200,
            early_stopping_kwargs=early_stopping_kwargs
        )
        print('\n....\n\n\n\n.....\n')
        features = trvae.get_latent()
        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing trvae features finished...')
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


    def compute_scalex_features_(self, outdir):
        print('[INFO]...Computing SCALEX features...')
        from scalex.function import SCALEX
        adata = self.adata.copy()
        if 'latent' in adata.obsm_keys():
            del adata.obsm['latent']
        adata=SCALEX(adata, batch_name=self.batch_obs_name, outdir=outdir, processed=True, ignore_umap=True, seed=self.seed)
        print('\n....\n\n\n\n.....\n')
        features = adata.obsm.pop('latent')
        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing SCALEX features finished...')
        return features

    def compute_sccobra_features_(self, outdir):
        print('[INFO]...Computing scCobra features...')
        import sys
        sys.path.insert(0, "/home/ca550013/Projects/scCobra/scCobra")
        #import scCobra
        from scCobra.function import scCobra
        adata = self.adata.copy()
        if 'latent' in adata.obsm_keys():
            del adata.obsm['latent']
        adata=scCobra(adata, batch_name=self.batch_obs_name, min_features=200, min_cells=3, outdir=outdir, processed=True, ignore_umap=True, seed=self.seed)
        print('\n....\n\n\n\n.....\n')
        features = adata.obsm.pop('latent')
        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing scCobra features finished...')
        return features


    def compute_pca_features_(self):
        print('[INFO]...Computing PCA features...')
        adata = self.adata.copy()
        if 'X_pca' in adata.obsm: adata.obsm.pop('X_pca')
        sc.tl.pca(adata, n_comps=30, svd_solver='arpack')
        print('\n....\n\n\n\n.....\n')
        features = adata.obsm.pop('X_pca')
        print('features shape',features.shape)
        print(features)
        print('[INFO]...Computing PCA features finished...')
        return features


    def compute_features(self,method_id,save_features=True,eval=False,**kwargs):
        method_name,version = method_id.strip(' ').split('_')
        print(f'Using method_name={method_name}, version={version}')
        
        print(f'Using seed={self.seed}')

        feature_obsm_name,method_root_dir = self.compute_init_(method_id)

        if method_name == 'harmony':
            features = self.compute_harmony_features_new_()

        elif method_name == 'trvae-sca':
            features = self.compute_trvae_features_new_()

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

        elif method_name == 'scalex':
            features = self.compute_scalex_features_(outdir=method_root_dir)

        elif method_name == 'sccobra':
            features = self.compute_sccobra_features_(outdir=method_root_dir)

        elif method_name == 'pca':
            features = self.compute_pca_features_()

        elif method_name == 'scdecorr':
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
        
        else:
            raise ValueError('Method not found')

        #self.adata.obsm[feature_obsm_name] = features
        #print(self.adata)
        print(f'Created features of shape {features.shape} using method={method_id}')

        with open(Path(method_root_dir)/'cfg.yaml', 'w') as f:
            yaml.dump({'seed': self.seed}, f, default_flow_style=False, indent=4)

        if save_features:
            out_path = Path(method_root_dir)/'latent.npy'
            print(f'[INFO]...Writing features as numpy array {out_path}...')
            with open(out_path, 'wb') as f:
                np.save(f, features)

        if eval:
            print('[INFO]...Eval started...')
            if kwargs['task'] not in self.task_list:
                raise Exception('Illegal Argument task: task should be in ',self.task_list)
            self.eval_(method_id,feature_obsm_name,task=kwargs['task'])
            print('[INFO]...Eval finished...')

        return self.adata



    def eval_(self,method_id,feature_obsm_name=None,task='all',**kwargs):

        feature_obsm_name = [feature_obsm_name] if feature_obsm_name is not None else None

        taskrun = DownstreamTaskRun(dataset_root_dir=self.dataset_root_dir,adata=self.adata,adata_fname=self.adata_fname,method_ids=[method_id],task=task,dataset_name=self.dataset_name,\
                            cell_class_obs_name=self.cell_class_obs_name,batch_obs_name=self.batch_obs_name,feature_obsm_names=feature_obsm_name)

        taskrun.run(**kwargs)




    def eval_benchmark(self,methods_tasks_list,**kwargs):

        print('...Running sequential benchmark pipeline...')

        for (method_id,feature_obsm_name,task) in dutils.gen_obsmname_task_tuples_(methods_tasks_list):
            print('[INFO]...Task {task} on {method_id} started...'.format(task=task,method_id=method_id))
            self.eval_(method_id,feature_obsm_name,task,**kwargs)
            print('[INFO]...Task {task} on {method_id} finished...'.format(task=task,method_id=method_id))


    #dont use multicore for classify tasks
    def eval_benchmark_multicore(self,methods_tasks_list,n_jobs=-1,**kwargs):

        print('...Running parallel benchmark pipeline...')
        from joblib import Parallel, delayed


        Parallel(n_jobs=n_jobs)(delayed(self.eval_)(method_id,feature_obsm_name,task,**kwargs)\
                                 for (method_id,feature_obsm_name,task) in dutils.gen_obsmname_task_tuples_(methods_tasks_list))



    def eval_benchmark_cluster_and_batch(self,method_ids=None,mode='eval',cluster_metrics=None,batch_metrics=None,n_jobs=1):

        if not method_ids:
            method_ids = list(map(lambda x:x+'_v1',self.methods_list))

        print('method_ids ',method_ids)

        methods_tasks_list = []

        for method_id in method_ids:
            methods_tasks_list.append({'method_id':method_id,'task':'cluster'})
            methods_tasks_list.append({'method_id':method_id,'task':'batch_correct'})


        print('methods_tasks_list ',methods_tasks_list)
        if n_jobs == -1 or n_jobs > 1:
            self.eval_benchmark_multicore(methods_tasks_list=methods_tasks_list,n_jobs=n_jobs,\
                                            mode=mode,cluster_metrics=cluster_metrics,batch_metrics=batch_metrics)


        elif n_jobs == 1:
            self.eval_benchmark(methods_tasks_list,mode=mode,cluster_metrics=cluster_metrics,batch_metrics=batch_metrics)

        else:
            raise Exception('n_jobs should either be eq to -1 or geq 1!')





    def eval_benchmark_batch(self,method_ids=None,mode='eval',batch_metrics=None):

        if not method_ids:
            method_ids = list(map(lambda x:x+'_v1',self.methods_list))

        print('method_ids ',method_ids)

        methods_tasks_list = []

        for method_id in method_ids:
            methods_tasks_list.append({'method_id':method_id,'task':'batch_correct'})


        print('methods_tasks_list ',methods_tasks_list)
        self.eval_benchmark(methods_tasks_list,mode=mode,batch_metrics=batch_metrics)



    def eval_benchmark_cluster(self,method_ids=None,mode='eval',cluster_metrics=None):

        if not method_ids:
            method_ids = list(map(lambda x:x+'_v1',self.methods_list))

        print('method_ids ',method_ids)

        methods_tasks_list = []


        for method_id in method_ids:
            methods_tasks_list.append({'method_id':method_id,'task':'cluster'})


        print('methods_tasks_list ',methods_tasks_list)
        self.eval_benchmark(methods_tasks_list,mode=mode,cluster_metrics=cluster_metrics)



    #use this for classify jobs
    def eval_benchmark_classify(self,methods_tasks_list,classify_model='knn',model_params=None,grid_search=None,clf_n_jobs=-1):

        self.eval_benchmark(methods_tasks_list,classify_model=classify_model,model_params=model_params,grid_search=grid_search,clf_n_jobs=clf_n_jobs)








