from sklearn import preprocessing
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,balanced_accuracy_score,\
                                roc_auc_score,ConfusionMatrixDisplay,classification_report
from sklearn.utils.class_weight import compute_sample_weight

import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import os
import time
import shutil



#methods_tasks_list=[{'method_id':..,'feature_obsm'..,'task':..},.....]
def gen_obsmname_task_tuples_(methods_tasks_list):

    for method_dict_ in methods_tasks_list:
        method_id = method_dict_.get('method_id')
        if not method_id:
            raise Exception('method_id has to be provided!')

        feature_obsm_name = method_dict_.get('feature_obsm','X_emb_{method_id}'.format(method_id=method_id))
        task = method_dict_.get('task','all')

        print('method_id,feature_obsm_name,task ',method_id,feature_obsm_name,task)

        yield (method_id,feature_obsm_name,task)


def compute_clusters(adata,use_rep,resolution=0.2,seed=123):

    neighbors_key = '{use_rep}_neighbor'.format(use_rep=use_rep)
    sc.pp.neighbors(adata,use_rep=use_rep,key_added=neighbors_key)
    sc.tl.leiden(adata,neighbors_key=neighbors_key,resolution=resolution,key_added='leiden_{use_rep}'.format(use_rep=use_rep))
    sc.tl.louvain(adata,neighbors_key=neighbors_key,random_state=seed,resolution=resolution,key_added='louvain_{use_rep}'.format(use_rep=use_rep))

    return adata

def compute_umap(adata,use_rep):

    neighbors_key = '{use_rep}_neighbor'.format(use_rep=use_rep)

    sc.tl.umap(adata,neighbors_key=neighbors_key)

    return adata


def plot_clusters(adata,use_rep,dataset_name,obs_name,leiden=False,louvain=False,other_obs_names=[],save_dir=None,sc_fidgdir=None):

    sc.settings.figdir = sc_fidgdir
    sc._settings.ScanpyConfig.figdir = sc_fidgdir

    compute_umap(adata,use_rep)
    obs_names = [obs_name]+other_obs_names

    if leiden:
        obs_names.append('leiden')
    if louvain:
        obs_names.append('louvain')


    for obs_name_ in obs_names:
        umap_fname = "_{dataset}_{use_rep}_{obs_name}.png".format(dataset=dataset_name,use_rep=use_rep,obs_name=obs_name_)
        sc.pl.umap(adata, color=[obs_name_],save=umap_fname,show=False)

        #copy UMAP to another file
        if save_dir and sc_fidgdir:
            
            print('sc_fidgdir',sc_fidgdir)
            time.sleep(1)
            shutil.copy(os.path.join(sc_fidgdir,"umap"+umap_fname),os.path.join(save_dir,"umap_{obs_name}.png".format(obs_name=obs_name_)))

    return adata


def compute_labels(adata,label_obs_name):
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()

    y = adata.obs[label_obs_name]
    y = le.fit_transform(y)
    #print(y)
    #print('y.shape ',y.shape)
    y_onehot = ohe.fit_transform(y.reshape(-1, 1)).toarray()
    #print('y_onehot.shape ',y_onehot.shape)
    #print(y_onehot)

    return y, y_onehot

def find_class_names(adata,label_obs_name):
    labels=adata.obs[label_obs_name]
    cnt=Counter(labels)
    return set(cnt.keys())



def evaluate_classification(y_test,y_test_onehot,y_pred,y_pred_proba,n_classes=None,test_size=None):
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_test
    )

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print('balanced_accuracy_score',balanced_accuracy)
    roc_auc = roc_auc_score(y_test_onehot, y_pred_proba,sample_weight=sample_weights,multi_class='ovr')
    print('roc_auc',roc_auc)
    roc_auc = roc_auc_score(y_test_onehot, y_pred_proba,sample_weight=sample_weights,multi_class='ova')
    print('roc_auc',roc_auc)
    p,r,f1,s=precision_recall_fscore_support(y_test,y_pred,average="weighted",sample_weight=sample_weights)
    print('PRF1s',p,r,f1,s)
    scores_df = pd.DataFrame({"balanced_accuracy":balanced_accuracy,'roc_auc':roc_auc,\
                           'weighted_precision':p,'weighted_recall':r,'weighted_f1':f1,'n_classes':n_classes,'batch_size':test_size}, \
                                index=[0])
    print('scores ',scores_df)

    report = classification_report(y_test, y_pred,sample_weight=sample_weights,output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print('classification report',report_df)

    return (scores_df,report_df)


def plot_confmat(y_test,y_pred,class_names,cm_title,cm_fig_save_path):

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_test
    )

    cm_display=ConfusionMatrixDisplay.from_predictions(y_test,y_pred,sample_weight=sample_weights,normalize='pred',display_labels=class_names)
    fig, ax = plt.subplots(figsize=(20,20))

    cm_display.plot(ax=ax,cmap=plt.cm.Blues,xticks_rotation=45)
    cm_display.ax_.set_title(cm_title)

    plt.tight_layout()
    plt.savefig(cm_fig_save_path,pad_inches=2)


#cumulative F1, auc scores of evaluating classification on all batch splits
def compute_weighted_batch_classify_metrics(batch_labels,task_dir):
        ##move to dutils
        print('[INFO]...Computing cumulative scores for all batches weighted by no. of common cell types...')

        wt_n_classes_rocauc=0
        wt_n_classes_f1 = 0
        wt_batch_size_rocauc=0
        wt_batch_size_f1=0
        mean_rocauc=0
        mean_f1 = 0
        cumsum_n_classes = 0
        cumsum_batch_size = 0


        print('batch_labels ',batch_labels)

        for batch_label in batch_labels:

            run_name = 'Batch_{batch}-vs-Rest'.format(batch=batch_label)
            batch_metrics_path = os.path.join(task_dir,run_name,'metrics_scores')

            metrics_df = pd.read_csv(batch_metrics_path)

            n_classes = metrics_df['n_classes'].iloc[0]
            batch_size = metrics_df['batch_size'].iloc[0]
            roc_auc = metrics_df['roc_auc'].iloc[0]
            f1 = metrics_df['weighted_f1'].iloc[0]

            cumsum_n_classes += n_classes
            cumsum_batch_size += batch_size
            print('cumsum_n_classes ',cumsum_n_classes)
            print('cumsum_batch_size ',cumsum_batch_size)

            wt_n_classes_rocauc += n_classes*roc_auc
            wt_batch_size_rocauc += batch_size*roc_auc
            mean_rocauc += roc_auc

            wt_n_classes_f1 += n_classes*f1
            wt_batch_size_f1 += batch_size*f1
            mean_f1 += f1



        wt_n_classes_rocauc=wt_n_classes_rocauc/cumsum_n_classes
        wt_n_classes_f1 = wt_n_classes_f1/cumsum_n_classes
        score_n_classes = (wt_n_classes_rocauc+wt_n_classes_f1)/2

        wt_batch_size_rocauc=wt_batch_size_rocauc/cumsum_batch_size
        wt_batch_size_f1=wt_batch_size_f1/cumsum_batch_size
        score_batch_size = (wt_batch_size_rocauc+wt_batch_size_f1)/2

        mean_rocauc=mean_rocauc/len(batch_labels)
        mean_f1 = mean_f1/len(batch_labels)
        score_mean = (mean_rocauc+mean_f1)/2



        scores_df = pd.DataFrame({"wt_n_classes_rocauc":wt_n_classes_rocauc,"wt_n_classes_f1":wt_n_classes_f1,\
                                    "wt_batch_size_rocauc":wt_batch_size_rocauc,"wt_batch_size_f1":wt_batch_size_f1,\
                                        "mean_rocauc":mean_rocauc,"mean_f1":mean_f1,'score_n_classes':score_n_classes,\
                                            'score_batch_size':score_batch_size,'score_mean':score_mean}, index=[0])

        print('scores_df',scores_df)
        return scores_df