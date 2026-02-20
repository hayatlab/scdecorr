import scanpy as sc
import scdecorr.downstream_utils as dutils
import os
import numpy as np
import time
import warnings
from sklearn.neighbors import KNeighborsClassifier

from joblib import Parallel, delayed
from scdecorr.evaluation_all import EvalMetrics
import datetime
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import json
import yaml


#a downstream run is a an atomic process which consists of
#>evaluating perf. of a specific downstream task (cluster/batch_correct/classify) 
#on a specific version of a model trained on an anndata consisiting of multiple batches
##routine assumes our DNN models are already trained and the weight files exists 
#or the trained features are already loaded in an obsm of the anndata (for both DNN and benchmark models)
#benchmark_eval=False means it runs a task for a single feature embedding
class DownstreamTaskRun():
    def __init__(self,dataset_root_dir,adata,adata_fname,method_ids,task,benchmark_eval=False,dataset_name=None,data_obs_name=None,\
                 cell_class_obs_name="cell_type",batch_obs_name="batch",feature_obsm_names=None,is_checkpoint=False,checkpoint_root=None,in_features=None,arch=None,sc_figdir=None):

        if sc_figdir:
            sc.settings.figdir = sc_figdir

        self.dataset_root_dir=dataset_root_dir

        self.adata_fname = adata_fname
        self.adata_path=os.path.join(self.dataset_root_dir,self.adata_fname)
        self.adata = adata        
        if self.adata is None:
            self.adata = sc.read_h5ad(self.adata_path)

        #print(self.adata,flush=True)
        self.all_tasks = ['cluster','batch_correct','classify']
        task_list = self.all_tasks + ['all']
        c = 0
        for _task in task_list:
            if task.startswith(_task):
                c += 1
        if c == 0:
            raise Exception('arg "task" should be in',task_list)
        self.task = task
        self.benchmark_eval = benchmark_eval

        if not self.benchmark_eval:
            if len(method_ids) > 1:
                raise Exception('...Length of method_ids can not exceed 1...')

            self.method_ids=method_ids[0]

            method_name,version = self.method_ids.strip(' ').split('_')
            self.feature_obsm_names = 'X_emb_{method_id}'.format(method_id=self.method_ids)\
                            if feature_obsm_names is None else feature_obsm_names[0]

            self.method_root_dir = os.path.join(self.dataset_root_dir,'method',method_name,version)
            if not os.path.exists(self.method_root_dir):
                os.makedirs(self.method_root_dir,exist_ok=True)

            if self.task == "all":
                self.task_dirs = {}
                for task in self.all_tasks:
                    self.task_dirs[task] = self.make_task_dir_(task)

                print('self.task_dirs \n',self.task_dirs)

            else:
                self.task_dir = self.make_task_dir_(self.task)



        #if benchmark_eval mode (scib_metrics)
        else:

            print('[INFO]...Evaluating for multiple methods...\n',method_ids)
            self.method_ids=method_ids

            self.feature_obsm_names = list(map(lambda x:'X_emb_{x}'.format(x=x),self.method_ids))\
                                            if feature_obsm_names is None else feature_obsm_names
            
            print('self.feature_obsm_names ',self.feature_obsm_names)
            run_id = datetime.datetime.now().strftime("%d-%m-%y-%H:M%:%S:%f")
            self.results_dir = os.path.join(self.dataset_root_dir,'method','benchmarks',run_id)

            self.task_dir = os.path.join(self.results_dir,self.task)

            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir,exist_ok=True)

            if not os.path.exists(self.task_dir):
                os.makedirs(self.task_dir,exist_ok=True)

        self.dataset_name = dataset_name
        self.data_obs_name = data_obs_name
        self.cell_class_obs_name = cell_class_obs_name
        self.batch_obs_name = batch_obs_name


        self.is_checkpoint = is_checkpoint
        print(self.is_checkpoint)

        if self.is_checkpoint:
            if self.benchmark_eval:
                raise Exception('Only single method allowed if checkpoint provided. Check benchmark_eval=False')

            if checkpoint_root is None or in_features is None or arch is None:
                raise Exception('All 3 required: checkpoint_root, in_features and arch have to be provided!')
            self.checkpoint_root = checkpoint_root
            self.in_features = in_features
            self.arch = arch

            print('[INFO]...Checkpoint provided...\n...loading features...')
            self.adata = self.load_features_()
            print('[INFO]...Features loaded...')

        print(self.adata,flush=True)

    def make_task_dir_(self,task):
        task_dir = os.path.join(self.method_root_dir,task)

        if not os.path.exists(task_dir):
            os.makedirs(task_dir,exist_ok=True)
        return task_dir


    # loads features from the checkpoint, 
    # and writes to a obsm "feature_obsm_name" of the adata 
    def load_features_(self):

        import utils

        if self.feature_obsm_names in list(self.adata.obsm):
            warnings.warn("{feature_obsm_name} already exists".format(feature_obsm_name=self.feature_obsm_names))

        features = utils.extract_features_from_adata(self.adata,self.checkpoint_root,self.data_obs_name,self.in_features,self.arch)

        self.adata.obsm[self.feature_obsm_names] = features
        return self.adata


    def classify_run_(self,task_dir,classify_model="knn",model_params={'n_neighbors': 11},grid_search=None,n_jobs=-1):
        print('[INFO]...Classify run started...')

        classify = Classify(self.adata,self.method_ids,'classify',\
                            self.dataset_name,self.dataset_root_dir,self.method_root_dir,task_dir,\
                            self.feature_obsm_names,self.cell_class_obs_name,\
                                self.batch_obs_name,classify_model,model_params,grid_search)

        cum_batch_scores = classify.eval_multi(n_jobs=n_jobs)
        print('[INFO]...Classify run finished...')

        return cum_batch_scores



    def cluster_run_(self, task_dir, mode='eval', **kwargs):
        print('[INFO]...Cluster run started...')

        if mode not in ['eval', 'plot', 'eval_batch_wise']:
            raise Exception('WrongModeError: Arg mode should be in [eval,plot]')

        cluster = Cluster(self.adata,self.method_ids,'cluster',\
                            self.dataset_name,self.dataset_root_dir,self.method_root_dir,task_dir,\
                            self.feature_obsm_names,self.cell_class_obs_name,\
                                self.batch_obs_name)
        if mode == 'eval':
            res = kwargs.get('res')
            metrics = kwargs.get('cluster_metrics')
            metric_scores = cluster.eval(metrics=metrics, res=res)

        elif mode == 'eval_batch_wise':
            res = kwargs.get('res')
            metrics = kwargs.get('cluster_metrics')
            n_jobs = kwargs.get('n_batch_jobs', -1)
            metric_scores = cluster.eval_batch_wise(metrics=metrics, res=res, n_jobs=n_jobs)

        elif mode == 'plot':
            cluster.plot()

        print('[INFO]...Cluster run finished...')
        return metric_scores


    def batch_correct_run_(self,task_dir,mode='eval',**kwargs):
        print('[INFO]...Batch Correct run started...')
        print(kwargs)

        if mode not in ['eval','plot']:
            raise Exception('WrongModeError: Arg mode should be in [eval,plot]')

        batch_correct = BatchCorrect(self.adata,self.method_ids,'batch_correct',\
                            self.dataset_name,self.dataset_root_dir,self.method_root_dir,task_dir,\
                            self.feature_obsm_names,self.cell_class_obs_name,\
                                self.batch_obs_name)

        if mode == 'eval':
            metrics = kwargs.get('batch_metrics')
            print('metrics1  \n\n',metrics)
            metric_scores = batch_correct.eval(metrics=metrics)

        elif mode == 'plot':
            batch_correct.plot()

        print('[INFO]...Batch Correct run finished...')

        return metric_scores


    #classify_model="xgb",save_adata=False,n_jobs=-1
    def call_task_(self,task,task_dir,**kwargs):

        if task.startswith('classify'):
            classify_model = kwargs.get('classify_model', 'knn')
            model_params = kwargs.get('model_params', {'n_neighbors': 11})
            grid_search = kwargs.get('grid_search')
            #n_jobs for training classifier 
            n_jobs = kwargs.get('clf_n_jobs',-1)
            self.classify_run_(task_dir=task_dir, classify_model=classify_model, model_params=model_params, grid_search=grid_search, n_jobs=n_jobs)

        elif task.startswith('cluster'):
            mode = kwargs.get('mode', 'eval')
            save_adata = kwargs.get('save_adata',False)
            metrics = kwargs.get('cluster_metrics', None)
            res = kwargs.get('res', None)
            self.cluster_run_(task_dir=task_dir, mode=mode, save_adata=save_adata, cluster_metrics=metrics,res=res)

        elif task.startswith('batch_correct'):
            mode = kwargs.get('mode','eval')
            metrics = kwargs.get('batch_metrics',None)
            self.batch_correct_run_(task_dir=task_dir, mode=mode, batch_metrics=metrics)



    def run(self,**kwargs):
        print('KWARGS ', kwargs)
        if self.benchmark_eval:
            raise Exception('Arg benchmark_eval should be checked False')

        if self.task == 'all':
            print('[INFO]...All downstream tasks run started...')

            for task in self.all_tasks:
                self.call_task_(task,self.task_dirs[task],**kwargs)

        else:
            self.call_task_(self.task,self.task_dir,**kwargs)



    def out_adata(self,out_path=None):
        self.adata.write(self.adata_path) if out_path is None else self.adata.write(out_path)





#expects features to be already loaded in an .obsm 'X_emb_{method_id}'
class BaseDownstream():
    def __init__(self,adata,method_id,task,dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        self.adata = adata
        #print(self.adata,flush=True)

        self.method_id = method_id
        self.feature_obsm_name = 'X_emb_{method_id}'.format(method_id=method_id)\
                        if feature_obsm_name is None else feature_obsm_name

        if (method_id == 'pca') and (self.feature_obsm_name not in self.adata.obsm_keys()):
            # if pca does not exist create pca
            if 'X_pca' in self.adata.obsm_keys():
                self.adata.obsm.pop('X_pca')
            sc.tl.pca(self.adata, n_comps=30, svd_solver='arpack')
            self.adata.obsm[feature_obsm_name] = self.adata.obsm.pop('X_pca')

        self.dataset_name = dataset_name
        self.task = task

        self.dataset_root_dir = dataset_root_dir
        self.method_root_dir = method_root_dir if method_root_dir is not None else None 
        self.task_dir = task_dir if task_dir is not None else None

        self.cell_class_obs_name = cell_class_obs_name
        self.batch_obs_name = batch_obs_name



    def eval(self,plot=True,save=True):
        if plot or save:
            if self.dataset_root_dir is None:
                raise Exception('arg "dataset_root_dir" can not be None')

            method_name,version = self.method_id.strip(' ').split('_')

            if self.method_root_dir is None or self.task_dir is None:
                self.method_root_dir = os.path.join(self.dataset_root_dir,'method',method_name,version)
                self.task_dir = os.path.join(self.method_root_dir,self.task)


            if not os.path.exists(self.method_root_dir):
                os.makedirs(self.method_root_dir,exist_ok=True)
            if not os.path.exists(self.task_dir):
                os.makedirs(self.task_dir,exist_ok=True)


#evaluates multi feature embeddings in feature_obsm_names
class BaseDownstreamMulti():
    def __init__(self,adata,method_ids,task,dataset_name=None,dataset_root_dir=None,results_dir=None,task_dir=None,\
                 feature_obsm_names=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        self.adata = adata
        #print(self.adata,flush=True)

        self.method_ids = method_ids
        self.feature_obsm_names = list(map(lambda x:'X_emb_{x}'.format(x=x),method_ids))\
                                        if feature_obsm_names is None else feature_obsm_names
        self.dataset_name = dataset_name
        self.task = task

        self.dataset_root_dir = dataset_root_dir
        self.results_dir = results_dir if results_dir is not None else None 
        self.task_dir = task_dir if task_dir is not None else None

        self.cell_class_obs_name = cell_class_obs_name
        self.batch_obs_name = batch_obs_name



    def eval(self,plot=True,save=True):
        if plot or save:
            if self.dataset_root_dir is None:
                raise Exception('arg "dataset_root_dir" can not be None')

            if self.results_dir is None or self.task_dir is None:
                run_id = datetime.datetime.now().strftime("%d-%m-%y-%H:M%:%S:%f")
                self.results_dir = os.path.join(self.dataset_root_dir,'method','benchmarks',run_id)
                self.task_dir = os.path.join(self.results_dir,self.task)


            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir,exist_ok=True)
            if not os.path.exists(self.task_dir):
                os.makedirs(self.task_dir,exist_ok=True)



class ClusterAndBatchCorrect(BaseDownstreamMulti):
    def __init__(self,adata,method_ids,task,dataset_name=None,dataset_root_dir=None,results_dir=None,task_dir=None,\
                 feature_obsm_names=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        print('task',task)
        if task != 'cluster_n_batch':
            raise Exception('Wrong task name: use cluster_n_batch')

        super().__init__(adata,method_ids,task,dataset_name,dataset_root_dir,results_dir,task_dir,\
                 feature_obsm_names,cell_class_obs_name,batch_obs_name)


    def eval_bench(self,save=True):
        from scib_metrics.benchmark import Benchmarker

        super().eval(save=save)
        print('self.feature_obsm_names ',self.feature_obsm_names)

        bm = Benchmarker(
            self.adata,
            batch_key=self.batch_obs_name,
            label_key=self.cell_class_obs_name,
            embedding_obsm_keys=self.feature_obsm_names,
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=-1,
        )

        bm.benchmark()
        if save:
            bm.plot_results_table(save_dir=self.task_dir)
            metrics_scores = bm.get_results()
            metrics_scores = metrics_scores.transpose()
            print(metrics_scores)
            metrics_scores.to_csv(os.path.join(self.task_dir,'metrics_scores'))

        else:
            bm.plot_results_table()
            metrics_scores = bm.get_results()
            metrics_scores = metrics_scores.transpose()
            print(metrics_scores)
        
        return metrics_scores

    def compute_clusters(self,resolution=0.2,seed=123):

        self.adata = dutils.compute_clusters(self.adata, self.feature_obsm_names,\
                                                resolution, seed)
        return self.adata


    def plot_umap(self):

        print('sc_fidgdir ' ,sc.settings.figdir)
        dutils.plot_clusters_new(self.adata, self.feature_obsm_names,self.dataset_name,\
                                            self.cell_class_obs_name,leiden=True,louvain=True,other_obs_names=[self.batch_obs_name],save_dir=self.results_dir,sc_figdir=sc.settings.figdir)




class Cluster(BaseDownstream):
    def __init__(self,adata,method_id,task,dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        print('task',task)
        if not task.startswith('cluster'):
            raise Exception('Wrong task name: use cluster')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)
        
        self.batch_labels = self.adata.obs[self.batch_obs_name].unique().tolist()
        print(vars(self))


    def compute_clusters(self,resolution=0.2,seed=123):

        self.adata = dutils.compute_clusters(self.adata, self.feature_obsm_name,\
                                                resolution, seed)
        return self.adata


    def plot_umap(self, res=0.2, barcodes=None, label=None, save_dir=None):
        print('sc_figdir ' ,sc.settings.figdir)
        if save_dir is None: save_dir = self.task_dir
        res_dir = Path(save_dir)/f'res={res:.1f}'
        #res_dir.mkdir(parents=True, exist_ok=True)
        dutils.plot_clusters_new(
            self.adata, 
            feature_obsm_name=self.feature_obsm_name,
            dataset_name=self.dataset_name,
            obs_name=self.cell_class_obs_name, 
            leiden=True, 
            louvain=False,
            res=res, 
            save_dir=res_dir, 
            sc_figdir=sc.settings.figdir,
            barcodes=barcodes,
            label=label,
        )


    #method for only plotting the clusters
    def plot(self):
        super().eval()
        #print('[INFO]...Computing clusters...')
        #self.adata = self.compute_clusters()
        print('[INFO]...Plotting UMAPs...')
        self.plot_umap()
        return self.adata


    #batchwise split the adata into train-test adatas 
    #a batch: test, adata excl. that batch: train
    #generate N train-test adata pairs for N batches
    def batchwise_split_adata(self):
        print('[INFO]...Batch-wise splitting adata...')
        print('batch_labels ',self.batch_labels)

        for batch_label in self.batch_labels:
            print('[INFO]....Processing batch {batch}....'.format(batch=batch_label))
            #train_adata = self.adata[self.adata.obs[self.batch_obs_name] != batch_label]
            batch_adata = self.adata[self.adata.obs[self.batch_obs_name] == batch_label]
            batch_barcodes = batch_adata.obs.index.tolist()
            batch_save_dir = f'{self.task_dir}/Batch_{batch_label}'
            yield (batch_adata, batch_barcodes, batch_label, batch_save_dir)


    def eval(self, adata=None, metrics=None, res=None, plot=True, save=True, save_dir=None, barcodes=None, label=None):
        if res is None: res = [0.1, 0.2, 0.5, 0.8, 1.0]
        if adata is None: adata = self.adata
        if save_dir is None: save_dir = self.task_dir
        super().eval(plot=plot,save=save)
        #print('[INFO]...Computing clusters...')
        #self.adata = self.compute_clusters()
        print('[INFO]...Eval Started...')
        eval_metrics = EvalMetrics(adata, self.feature_obsm_name, self.batch_obs_name, self.cell_class_obs_name)

        for res_ in res:
            print(f'[INFO]...Computing scores on resolution={res_:.1f}...')
            metric_scores = eval_metrics.compute_cell_type_metrics_new(metrics, res=res_)
            print(f'Metric scores res={res_:.1f}',metric_scores)

            if save:
                res_dir = Path(save_dir)/f'res={res_:.1f}'
                res_dir.mkdir(parents=True, exist_ok=True)
                print(f'[INFO]...Saving clustering scores at {res_dir}...')
                metric_scores.to_csv(res_dir/'metrics_scores')

            if plot:
                print('[INFO]...Plotting UMAPs...')
                self.plot_umap(res=res_, barcodes=barcodes, label=label, save_dir=save_dir)

        return metric_scores


    def eval_batch_wise(self, metrics=None, res=None, plot=False, save=True, n_jobs=-1):
        if res is None: res = [0.1]
        # Use for Label Transfer using Clustering Tasks
        super().eval(plot=plot,save=save)
        print('[INFO]...Batch-wise Cluster started...')

        Parallel(n_jobs=n_jobs)(
            delayed(self.eval)(
                adata=batch_adata, 
                metrics=metrics,
                res=res,
                plot=plot,
                save=save,
                save_dir=save_dir,
                barcodes=batch_barcodes,
                label=batch_label,
            ) for (batch_adata, batch_barcodes, batch_label, save_dir) in self.batchwise_split_adata())

        print('[INFO]...Batch-wise Cluster eval finished...')

        #cumulative batch scores
        all_batch_scores_df = dutils.compute_mean_batch_cluster_metrics(self.batch_labels, self.task_dir, res[0])
        if save:
            print('[INFO]...Saving batch cumulative classify scores...')
            all_batch_scores_df.to_csv(os.path.join(self.task_dir,'all_batches_scores.csv'))

        return all_batch_scores_df



class BatchCorrect(BaseDownstream):
    def __init__(self,adata,method_id,task,dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        if not task.startswith('batch_correct'):
            raise Exception('Wrong task name: use batch_correct')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)

        print(vars(self))


    def plot_umap(self):
        print('sc_fidgdir ' ,sc.settings.figdir)
        dutils.plot_clusters_new(
            self.adata, 
            feature_obsm_name=self.feature_obsm_name,
            dataset_name=self.dataset_name,
            obs_name=self.batch_obs_name, 
            leiden=False, 
            louvain=False,
            res=None, 
            save_dir=self.task_dir, 
            sc_figdir=sc.settings.figdir
        )

    #method for only plotting the clusters
    def plot(self):
        super().eval()
        print('[INFO]...Plotting UMAPs...')
        self.plot_umap()
        return self.adata


    def eval(self,metrics=None,plot=True,save=True):
        super().eval(plot=plot,save=save)
        print('[INFO]...Eval Started...')
        print('   METRICS     \n\n ',metrics)
        eval_metrics = EvalMetrics(self.adata, self.feature_obsm_name, self.batch_obs_name, self.cell_class_obs_name)
        metric_scores = eval_metrics.compute_batch_metrics(metrics=metrics)

        print('Metric scores ',metric_scores)
        if save:
            print('[INFO]...Saving batch_correct scores...')
            metric_scores.to_csv(os.path.join(self.task_dir,'metrics_scores'))

        if plot:
            print('[INFO]...Plotting UMAPs...')
            self.plot_umap()

        return metric_scores




class Classify(BaseDownstream):
    def __init__(self,adata,method_id,task='classify',dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch",classify_model='knn',model_params={'n_neighbors': 11},grid_search=None):

        if not task.startswith('classify'):
            raise Exception('Wrong task name: use classify')
        if classify_model is None:
            raise Exception('Classify model not provided!')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)

        self.classify_model = classify_model
        self.model_params = model_params
        self.grid_search = grid_search
        self.batch_labels = self.adata.obs[self.batch_obs_name].unique().tolist()


    #batchwise split the adata into train-test adatas 
    #a batch: test, adata excl. that batch: train
    #generate N train-test adata pairs for N batches
    def batchwise_split_adata(self):
        print('[INFO]...Batch-wise splitting adata...')
        print('batch_labels ',self.batch_labels)

        for batch_label in self.batch_labels:
            print('[INFO]....Processing batch {batch}....'.format(batch=batch_label))

            train_adata = self.adata[self.adata.obs[self.batch_obs_name] != batch_label]
            test_adata = self.adata[self.adata.obs[self.batch_obs_name] == batch_label]

            yield (train_adata, test_adata, batch_label)

    #finds common cell types in train and test adatas
    #subset train and test adatas on those common cell types
    #return X_train, y_train, X_test, y_test, y_test_onehot
    def prepare_train_test_data(self,train_adata,test_adata,batch_label):
        print('[INFO]...Preparing train, test samples...')
        print('[INFO]...Testing on {batch}, Training on the Rest'.format(batch=batch_label))

        common_cell_types = list(set(train_adata.obs[self.cell_class_obs_name].tolist()).intersection(set(test_adata.obs[self.cell_class_obs_name].tolist())))
        print('common_cell_types: ', common_cell_types)

        train_adata_common_cell_types = train_adata[train_adata.obs[self.cell_class_obs_name].isin(common_cell_types)]
        test_adata_common_cell_types = test_adata[test_adata.obs[self.cell_class_obs_name].isin(common_cell_types)]

        X_train, X_test = train_adata_common_cell_types.obsm[self.feature_obsm_name], test_adata_common_cell_types.obsm[self.feature_obsm_name]
        y_train, y_test = train_adata_common_cell_types.obs[self.cell_class_obs_name].to_numpy(), test_adata_common_cell_types.obs[self.cell_class_obs_name].to_numpy()
        barcodes_train, barcodes_test = train_adata_common_cell_types.obs.index.to_numpy(), test_adata_common_cell_types.obs.index.to_numpy()

        #y_train, _ = dutils.compute_labels(train_adata_common_cell_types, self.cell_class_obs_name)
        y_test_onehot = dutils.to_onehot(y_test)
        print('[INFO]...Shape of X_train, y_train: {X_train_shape},{y_train_shape}...'.format(X_train_shape=X_train.shape,y_train_shape=y_train.shape))
        print('\n[INFO]...Shape of X_test, y_test: {X_test_shape},{y_test_shape},{y_test_onehot_shape}...'.format(X_test_shape=X_test.shape,y_test_shape=y_test.shape,y_test_onehot_shape=y_test_onehot.shape))

        return (X_train, y_train, X_test, y_test, y_test_onehot, barcodes_train, barcodes_test, common_cell_types)


    def xgb_(self, n_classes, depth=6):
        
        from xgboost import XGBClassifier

        print('[INFO]..Init XGB model...')
        if n_classes > 2:
            model = XGBClassifier(max_depth=depth,learning_rate=0.1,objective="multi:softproba",n_jobs=-1)

        else:
            model = XGBClassifier(max_depth=depth,learning_rate=0.1,objective="binary:logistic",n_jobs=-1)

        print(model)
        return model


    def knn_(self, n_neighbors=11):
        print('[INFO]..Init KNN model...')
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model

    def fit_and_predict(self,model,X_train,y_train,X_test):
        print('[INFO]...Training Model...')
        model.fit(X_train, y_train)
        print('[INFO]...Testing Model...')

        y_pred = model.predict(X_test)
        #print('y_pred',y_pred)
        y_pred_proba = model.predict_proba(X_test)
        #print(y_pred_proba)

        return (y_pred,y_pred_proba)

    def eval_(self, train_adata, test_adata, batch_label, plot=True, save=True):

        run_name = 'Batch_{batch}-vs-Rest'.format(batch=batch_label)

        X_train, y_train, X_test, y_test, y_test_onehot, barcodes_train, barcodes_test, class_names = self.prepare_train_test_data(train_adata,test_adata,batch_label)
        n_classes = len(class_names)
        train_size, test_size = X_train.shape[0], X_test.shape[0]
        
        if self.classify_model == "xgb":
            model = self.xgb_(n_classes)
        elif self.classify_model == 'knn':
            model = self.knn_(self.model_params.get('n_neighbors', 11))
        else:
            raise ValueError('Model must be KNN or XGB')

        if self.grid_search is not None:
            assert self.classify_model == 'knn'
            print('[INFO]...Grid Search on KNN started...')
            param_grid = self.grid_search.get('param_grid')
            if param_grid is None:
                max_neighbors = np.sqrt(train_size)
                max_neighbors = int(max_neighbors) if train_size < 100000 else int(max_neighbors/4)
                step = 4 if train_size < 100000 else 8
                param_grid = {'n_neighbors': list(range(5, max_neighbors, step))}
            cv = self.grid_search.get('cv', 5 if train_size < 100000 else 3)
            scoring = self.grid_search.get('scoring', 'f1_macro')
            n_jobs = self.grid_search.get('n_jobs', 1)
            print(f'[INFO]...Grid Search Params: param_grid={param_grid}, cv={cv}, n_jobs={n_jobs}')
            model = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)

        print('[INFO]...Training Model...') 
        model.fit(X_train, y_train)
        if self.grid_search is not None:
            gs_results = {'best_params': model.best_params_, 'best_score': {scoring: model.best_score_}}
            print('Grid Search Results', gs_results)
            print("Best value of K: ", model.best_params_)
            print("Mean CV accuracy of best K-value: ", model.best_score_)

        class_names = list(model.classes_)
        print(f'[INFO]...Model Class Names={class_names}...')

        print('[INFO]...Predicting on Test...')
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        print('[INFO]...Evaluating Model...')
        classify_metrics_scores_df,classify_report_df = dutils.evaluate_classification_new(y_test,y_pred,batch_label,test_size,n_classes)
        print('classify_metrics_scores',classify_metrics_scores_df)

        if save or plot:
            batch_save_path = os.path.join(self.task_dir,run_name)
            if not os.path.exists(batch_save_path):
                os.makedirs(batch_save_path,exist_ok=True)
            print('batch_save_path ',batch_save_path)

        if save:
            print(f'[INFO]...Saving classify scores to {batch_save_path}...')
            classify_metrics_scores_df.to_csv(os.path.join(batch_save_path,'metrics_scores.csv'))
            classify_report_df.to_csv(os.path.join(batch_save_path,'classification_report.csv'))
            
            # Save test prediction data
            print('[INFO]...Saving prediction output data...')
            out_data = {
                'barcodes_test': barcodes_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_test_onehot': y_test_onehot,
                'y_pred_proba': y_pred_proba,
                'class_names': class_names,
                'n_classes': n_classes,
                'test_size': test_size,
                'batch_label': batch_label,
            }
            out_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in out_data.items()}
            out_datatypes = {k:type(v) for k,v in out_data.items()}
            print(f'[INFO]...Prediction output data types:{out_datatypes}')
            with open(os.path.join(batch_save_path,'predict_data.json'), 'w') as f:
                json.dump(out_data, f, indent=2)

            # Save best grid search params
            if self.grid_search is not None:
                with open(os.path.join(batch_save_path,'grid_search_results.json'), 'w') as f:
                    json.dump(gs_results, f, indent=2)
            
        if plot:
            print('[INFO]...Plotting confusion matrix...')
            dutils.plot_confmat(y_test,y_pred,class_names,cm_title=run_name,cm_fig_save_path=os.path.join(batch_save_path,'confmat.png'))


    def eval(self,plot=True,save=True):

        super().eval(plot=plot,save=save)

        for (train_adata,test_adata,batch_label) in self.batchwise_split_adata():
            self.eval_(train_adata,test_adata,batch_label)


    def eval_multi(self,plot=True,save=True,n_jobs=-1):
        super().eval(plot=plot,save=save)
        print('[INFO]...Multicore Classify eval started...')

        # Save Task config
        cfg = {
            'batch_obs_name': self.batch_obs_name,
            'cell_class_obs_name': self.cell_class_obs_name,
            'classify_model': self.classify_model,
            'model_params': self.model_params,
            'grid_search': self.grid_search,
        }

        with open(os.path.join(self.task_dir, 'task_config.yaml'), 'w') as outfile: # type: ignore
            yaml.dump(cfg, outfile, default_flow_style=False, indent=2)

        Parallel(n_jobs=n_jobs)(delayed(self.eval_)(train_adata,test_adata,batch_label) for (train_adata,test_adata,batch_label) in self.batchwise_split_adata())
        print('[INFO]...Multicore Classify eval finished...')

        #cumulative batch scores
        all_batch_scores_df = dutils.compute_mean_batch_classify_metrics(self.batch_labels,self.task_dir)

        if save:
            print('[INFO]...Saving batch cumulative classify scores...')
            all_batch_scores_df.to_csv(os.path.join(self.task_dir,'all_batches_scores.csv'))

        return all_batch_scores_df



























