import scanpy as sc
import downstream_utils as dutils
import os
import numpy as np
import time
import warnings
from sklearn.neighbors import KNeighborsClassifier

from joblib import Parallel, delayed
from evaluation_all import EvalMetrics
import datetime

#a downstream run is a an atomic process which consists of
#>evaluating perf. of a specific downstream task (cluster/batch_correct/classify) 
#on a specific version of a model trained on an anndata consisiting of multiple batches
##routine assumes our DNN models are already trained and the weight files exists 
#or the trained features are already loaded in an obsm of the anndata (for both DNN and benchmark models)
#benchmark_eval=False means it runs a task for a single feature embedding
class DownstreamTaskRun():
    def __init__(self,dataset_root_dir,adata_fname,method_ids,task,benchmark_eval=False,dataset_name=None,data_obs_name=None,\
                 cell_class_obs_name="cell_type",batch_obs_name="batch",feature_obsm_names=None,is_checkpoint=False,checkpoint_root=None,in_features=None,arch=None,sc_figdir=None):

        if sc_figdir:
            sc.settings.figdir = sc_figdir

        self.dataset_root_dir=dataset_root_dir

        self.adata_fname = adata_fname
        self.adata_path=os.path.join(self.dataset_root_dir,self.adata_fname)
        self.adata = sc.read_h5ad(self.adata_path)
        #print(self.adata,flush=True)
        self.all_tasks = ['cluster','batch_correct','classify']

        task_list = self.all_tasks + ['all']
        if task not in task_list:
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



    def classify_run_(self,task_dir,classify_model="xgb",n_jobs=-1):
        print('[INFO]...Classify run started...')


        classify = Classify(self.adata,self.method_ids,'classify',\
                            self.dataset_name,self.dataset_root_dir,self.method_root_dir,task_dir,\
                            self.feature_obsm_names,self.cell_class_obs_name,\
                                self.batch_obs_name,classify_model)

        cum_batch_scores = classify.eval_multi(n_jobs=n_jobs)
        print('[INFO]...Classify run finished...')

        return cum_batch_scores



    def cluster_run_(self,task_dir,mode='eval',save_adata=False,**kwargs):
        print('[INFO]...Cluster run started...')

        if mode not in ['eval','plot']:
            raise Exception('WrongModeError: Arg mode should be in [eval,plot]')

        cluster = Cluster(self.adata,self.method_ids,'cluster',\
                            self.dataset_name,self.dataset_root_dir,self.method_root_dir,task_dir,\
                            self.feature_obsm_names,self.cell_class_obs_name,\
                                self.batch_obs_name)

        if mode == 'eval':
            metrics = kwargs.get('cluster_metrics')
            self.adata,metric_scores = cluster.eval(metrics=metrics)

        elif mode == 'plot':
            cluster.plot()

        if save_adata:
            self.out_adata()


        print('[INFO]...Cluster run finished...')

        return self.adata,metric_scores


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

        if task == 'classify':
            classify_model = kwargs.get('classify_model','xgb')

            #n_jobs for training classifier 
            n_jobs = kwargs.get('clf_n_jobs',-1)
            self.classify_run_(task_dir=task_dir,classify_model=classify_model,n_jobs=n_jobs)

        elif task == 'cluster':
            save_adata = kwargs.get('save_adata',False)
            mode = kwargs.get('mode','eval')
            metrics = kwargs.get('cluster_metrics',None)

            self.cluster_run_(task_dir=task_dir,mode=mode,save_adata=save_adata,cluster_metrics=metrics)

        elif task == 'batch_correct':
            mode = kwargs.get('mode','eval')
            metrics = kwargs.get('batch_metrics',None)
            self.batch_correct_run_(task_dir=task_dir,mode=mode,batch_metrics=metrics)



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
        dutils.plot_clusters(self.adata, self.feature_obsm_names,self.dataset_name,\
                                            self.cell_class_obs_name,leiden=True,louvain=True,other_obs_names=[self.batch_obs_name],save_dir=self.results_dir,sc_fidgdir=sc.settings.figdir)




class Cluster(BaseDownstream):
    def __init__(self,adata,method_id,task,dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        print('task',task)
        if task != 'cluster':
            raise Exception('Wrong task name: use cluster')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)
        
        print(vars(self))


    def compute_clusters(self,resolution=0.2,seed=123):

        self.adata = dutils.compute_clusters(self.adata, self.feature_obsm_name,\
                                                resolution, seed)
        return self.adata


    def plot_umap(self):

        print('sc_fidgdir ' ,sc.settings.figdir)
        dutils.plot_clusters(self.adata, self.feature_obsm_name,self.dataset_name,\
                                            self.cell_class_obs_name,leiden=True,louvain=True,save_dir=self.task_dir,sc_fidgdir=sc.settings.figdir)

    #method for only plotting the clusters
    def plot(self):
        super().eval()
        print('[INFO]...Computing clusters...')
        self.adata = self.compute_clusters()
        print('[INFO]...Plotting UMAPs...')
        self.plot_umap()

        return self.adata



    def eval(self,metrics=None,plot=True,save=True):
        super().eval(plot=plot,save=save)
        print('[INFO]...Computing clusters...')
        self.adata = self.compute_clusters()
        print('[INFO]...Eval Started...')

        eval_metrics = EvalMetrics(self.adata, self.feature_obsm_name, self.batch_obs_name, self.cell_class_obs_name)
        metric_scores = eval_metrics.compute_cell_type_metrics(metrics)

        print('Metric scores ',metric_scores)
        if save:
            print('[INFO]...Saving clustering scores...')
            metric_scores.to_csv(os.path.join(self.task_dir,'metrics_scores'))

        if plot:
            print('[INFO]...Plotting UMAPs...')
            self.plot_umap()

        return self.adata, metric_scores







class BatchCorrect(BaseDownstream):
    def __init__(self,adata,method_id,task,dataset_name=None,dataset_root_dir=None,method_root_dir=None,task_dir=None,\
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch"):

        if task != 'batch_correct':
            raise Exception('Wrong task name: use batch_correct')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)

        print(vars(self))


    def plot_umap(self):


        dutils.plot_clusters(self.adata, self.feature_obsm_name,self.dataset_name,\
                                            self.batch_obs_name,save_dir=self.task_dir,sc_fidgdir=sc.settings.figdir)


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
                 feature_obsm_name=None,cell_class_obs_name="cell_type",batch_obs_name="batch",classify_model='xgb'):

        if task != 'classify':
            raise Exception('Wrong task name: use classify')
        if classify_model is None:
            raise Exception('Classify model not provided!')

        super().__init__(adata,method_id,task,dataset_name,dataset_root_dir,method_root_dir,task_dir,\
                 feature_obsm_name,cell_class_obs_name,batch_obs_name)

        self.classify_model = classify_model
        self.batch_labels = np.unique(self.adata.obs[self.batch_obs_name].to_numpy())

        print(vars(self))


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

            yield (train_adata, test_adata,batch_label)


    #finds common cell types in train and test adatas
    #subset train and test adatas on those common cell types
    #return X_train, y_train, X_test, y_test, y_test_onehot
    def prepare_train_test_data(self,train_adata,test_adata,batch_label):
        print('[INFO]...Preparing train, test samples...')
        print('[INFO]...Testing on {batch}, Training on the Rest'.format(batch=batch_label))

        common_cell_types = list(set(train_adata.obs[self.cell_class_obs_name].tolist()).intersection(set(test_adata.obs[self.cell_class_obs_name].tolist())))
        print('common_cell_types: ',common_cell_types)

        train_adata_common_cell_types = train_adata[train_adata.obs[self.cell_class_obs_name].isin(common_cell_types)]


        test_adata_common_cell_types = test_adata[test_adata.obs[self.cell_class_obs_name].isin(common_cell_types)]

        X_train, X_test = train_adata_common_cell_types.obsm[self.feature_obsm_name], test_adata_common_cell_types.obsm[self.feature_obsm_name]
        y_train, _ = dutils.compute_labels(train_adata_common_cell_types, self.cell_class_obs_name)
        y_test, y_test_onehot = dutils.compute_labels(test_adata_common_cell_types, self.cell_class_obs_name)
        print('[INFO]...Shape of X_train, y_train: {X_train_shape},{y_train_shape}...'.format(X_train_shape=X_train.shape,y_train_shape=y_train.shape))

        print('\n[INFO]...Shape of X_test, y_test, y_test_onehot: {X_test_shape},{y_test_shape},{y_test_onehot_shape}...'\
                            .format(X_train_shape=X_train.shape,y_train_shape=y_train.shape,\
                                 X_test_shape=X_test.shape,y_test_shape=y_test.shape,y_test_onehot_shape=y_test_onehot.shape))


        return (X_train, y_train, X_test, y_test, y_test_onehot, common_cell_types)


    def xgb_(self,class_names,depth=6):
        
        from xgboost import XGBClassifier

        print('[INFO]..Init XGB model...')
        if len(class_names) > 2:
            model = XGBClassifier(max_depth=depth,learning_rate=0.1,objective="multi:softproba",n_jobs=-1)

        else:
            model = XGBClassifier(max_depth=depth,learning_rate=0.1,objective="binary:logistic",n_jobs=-1)

        print(model)
        return model


    def knn_(self,n_neighbors=11):
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

    def eval_(self,train_adata,test_adata,batch_label,plot=True,save=True):

        run_name = 'Batch_{batch}-vs-Rest'.format(batch=batch_label)

        X_train, y_train, X_test, y_test, y_test_onehot, class_names = self.prepare_train_test_data(train_adata,test_adata,batch_label)
        n_classes = len(class_names)
        test_size = X_test.shape[0]

        if self.classify_model == "xgb":
            model = self.xgb_(class_names)
        elif self.classify_model == 'knn':
            model = self.knn_()

        y_pred,y_pred_proba = self.fit_and_predict(model,X_train,y_train,X_test)
        print('[INFO]...Evaluating Model...')
        classify_metrics_scores_df,classify_report_df = dutils.evaluate_classification(y_test,y_test_onehot,y_pred,y_pred_proba,n_classes,test_size)
        print('classify_metrics_scores_df',classify_metrics_scores_df)
        print('classify_report_df',classify_report_df)

        if save or plot:
            batch_save_path = os.path.join(self.task_dir,run_name)
            if not os.path.exists(batch_save_path):
                os.makedirs(batch_save_path,exist_ok=True)
            print('batch_save_path ',batch_save_path)

        if save:
            print('[INFO]...Saving classify scores...')
            classify_metrics_scores_df.to_csv(os.path.join(batch_save_path,'metrics_scores'))
            classify_report_df.to_csv(os.path.join(batch_save_path,'classification_report'))
            print('classify_metrics_scores_df[n_classes].iloc[0] ',classify_metrics_scores_df['n_classes'].iloc[0])

        if plot:
            print('[INFO]...Plotting confusion matrix...')

            dutils.plot_confmat(y_test,y_pred,class_names,cm_title=run_name,cm_fig_save_path=os.path.join(batch_save_path,'confmat.pdf'))



    def eval(self,plot=True,save=True):

        super().eval(plot=plot,save=save)

        for (train_adata,test_adata,batch_label) in self.batchwise_split_adata():
            self.eval_(train_adata,test_adata,batch_label)


    def eval_multi(self,plot=True,save=True,n_jobs=-1):
        super().eval(plot=plot,save=save)
        print('[INFO]...Multicore Classify eval started...')
        Parallel(n_jobs=n_jobs)(delayed(self.eval_)(train_adata,test_adata,batch_label) for (train_adata,test_adata,batch_label) in self.batchwise_split_adata())
        print('[INFO]...Multicore Classify eval finished...')

        #cumulative batch scores
        cum_batch_scores_df = dutils.compute_weighted_batch_classify_metrics(self.batch_labels,self.task_dir)

        if save:
            print('[INFO]...Saving batch cumulative classify scores...')
            cum_batch_scores_df.to_csv(os.path.join(self.task_dir,'batch_cum_scores'))

        return cum_batch_scores_df



























