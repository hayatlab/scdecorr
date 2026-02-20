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
from scipy.sparse import csr_matrix
from pathlib import Path

sc_figdir = Path("/home/results/figures")
sc.settings.figdir = sc_figdir
sc._settings.ScanpyConfig.figdir = sc_figdir



def convert_to_sparse_(adata):

    from scipy.sparse._csr import csr_matrix

    if type(adata.X) == np.ndarray:
        adata.X = csr_matrix(adata.X)

    if type(adata.layers['counts']) == np.ndarray:
        adata.layers['counts'] = csr_matrix(adata.layers['counts'])

    print(type(adata.X))
    print(type(adata.layers['counts']))


    return adata


def compute_mul_embs_mul_data(datasets,root_dir,method_ids):

    from benchmarking import Benchmark
    from dataset_cfg import dataset_cfg

    for dataset in datasets:
        print('[INFO]...Processing dataset {dataset}...'.format(dataset=dataset))

        dataset_root_dir = os.path.join(root_dir,dataset)
        adata_fname = dataset_cfg[dataset]['adata_fname']
        dataset_name = dataset
        cell_class_obs_name = dataset_cfg[dataset]['cell_class_obs_name']
        batch_obs_name = dataset_cfg[dataset]['batch_obs_name']
        checkpoint_root = dataset_cfg[dataset]['checkpoint_root']
        in_features = dataset_cfg[dataset]['in_features']
        arch = dataset_cfg[dataset]['arch']


        bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)

        print('dataset_name,dataset_root_dir,adata_fname,cell_class_obs_name,batch_obs_name \n',\
            dataset_name,dataset_root_dir,adata_fname,cell_class_obs_name,batch_obs_name)


        for method_id in method_ids:
            print('[INFO]...Computing features of dataset {dataset} using {method_id} started...'.format(dataset=dataset,method_id=method_id))
            bench.compute_features(method_id,save_features=True,checkpoint_root=checkpoint_root,in_features=in_features,arch=arch)
            time.sleep(1)
            print('[INFO]...Computing features of dataset {dataset} using {method_id} finished...'.format(dataset=dataset,method_id=method_id))

        print('[INFO]...Processing dataset {dataset} finished...'.format(dataset=dataset))


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
        sc.pl.umap(adata, color=[obs_name_],save=umap_fname,legend_fontsize='xx-large',legend_fontweight='black',show=False)

        #copy UMAP to another file
        if save_dir and sc_fidgdir:
            print('sc_fidgdir',sc_fidgdir)
            time.sleep(1)
            shutil.copy(os.path.join(sc_fidgdir,"umap"+umap_fname),os.path.join(save_dir,"umap_{obs_name}.png".format(obs_name=obs_name_)))

    return adata


def plot_clusters_new(adata, feature_obsm_name, dataset_name, obs_name, leiden=False, louvain=False, res=None, other_obs_names=[], save_dir=None, sc_figdir=None, barcodes=None, label=None):
    adata_cp = adata.copy()
    sc_figdir = Path("/home/results/figures")
    sc.settings.figdir = sc_figdir
    sc._settings.ScanpyConfig.figdir = sc_figdir

    obs_names = [obs_name]+other_obs_names
    if leiden:
        obs_names.append(f'leiden_res={res:.1f}_{feature_obsm_name}')
    if louvain:
        obs_names.append(f'louvain_res={res:.1f}_{feature_obsm_name}')

    print('obs_names', obs_names)

    for obs_name_ in obs_names:
        print('obs_name_', obs_name_)
        umap_fname = f"_{dataset_name}_{label}_{obs_name_}.png" if label is not None else f"_{dataset_name}_{obs_name_}.png"
        basis=f'{feature_obsm_name}_umap'
        if barcodes is not None: # if cell barcodes is given, then plot for only those barcodes
            adata_cp.obs.loc[~adata_cp.obs.index.isin(barcodes), obs_name_] = None
        sc.pl.embedding(
            adata_cp, 
            basis=basis, 
            neighbors_key=feature_obsm_name, 
            color=[obs_name_],
            size=10,
            legend_fontsize='xx-large', 
            legend_fontweight='black',
            show=False,
            save=umap_fname
        )
        #copy UMAP to another file
        if save_dir and sc_figdir:
            print('sc_figdir',sc_figdir)
            time.sleep(1)
            plot_fname = f"{basis}{umap_fname}"
            print(f'Figure Path={Path(sc_figdir, plot_fname)}')
            print(f'UMAP SAVE PATH={Path(save_dir, f"umap_{obs_name_}.png")}')
            shutil.copy(Path(sc_figdir, plot_fname), Path(save_dir, f"umap_{obs_name_}.png"))
            assert Path(save_dir, f"umap_{obs_name_}.png").exists()


def compute_labels(adata,label_obs_name):
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    y = adata.obs[label_obs_name]
    y = le.fit_transform(y)
    y_onehot = ohe.fit_transform(y.reshape(-1, 1))
    return y_onehot


def to_onehot(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    ohe = preprocessing.OneHotEncoder()
    y_onehot = ohe.fit_transform(y.reshape(-1, 1))
    if isinstance(y_onehot, csr_matrix): 
        y_onehot = y_onehot.toarray()
    return y_onehot


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


def evaluate_classification_new(y_test,y_pred,batch_label=None,test_size=None,n_classes=None):

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print('balanced_accuracy_score',balanced_accuracy)
    scores_names = ['precision', 'recall', 'f1', 'support']
    wt_scores = precision_recall_fscore_support(y_test,y_pred,average="weighted")
    wt_scores = dict(zip(scores_names, wt_scores))
    print('Weighted PRF1 scores' , wt_scores)

    macro_scores=precision_recall_fscore_support(y_test,y_pred,average="macro")
    macro_scores = dict(zip(scores_names, macro_scores))
    print('Macro PRF1 scores ', macro_scores)

    scores_df = pd.DataFrame({
        "batch_name": batch_label,
        "balanced_accuracy":balanced_accuracy,
        'weighted_precision':wt_scores['precision'],
        'weighted_recall':wt_scores['recall'],
        'weighted_f1':wt_scores['f1'],
        'wt_classify_score': 0.5*(balanced_accuracy + wt_scores['f1']),
        'macro_precision':macro_scores['precision'],
        'macro_recall':macro_scores['recall'],
        'macro_f1':macro_scores['f1'],
        'macro_classify_score': 0.5*(balanced_accuracy + macro_scores['f1']),
        'n_classes':n_classes,
        'batch_size':test_size
    }, index=[0])

    print('scores ',scores_df)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print('classification report',report_df)
    return (scores_df, report_df)



def plot_confmat(
    y_test,
    y_pred,
    class_names,
    cm_title=None,
    cm_fig_save_path=None,
    figsize=(25, 25),
    fontsize=20,
):
    fig, ax = plt.subplots(figsize=figsize)

    # Create the confusion matrix display with formatted values
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize='pred',
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax,
        xticks_rotation=45,
        values_format=".2f",
        colorbar=False
    )

    # Set title
    #if cm_title is None:
    #    cm_title = 'Confusion Matrix'
    #cm_display.ax_.set_title(cm_title, fontsize=fontsize + 4, fontweight='bold', pad=20)

    # Set axis labels
    cm_display.ax_.set_xlabel('Predicted Label', fontsize=fontsize + 10, fontweight='bold')
    cm_display.ax_.set_ylabel('True Label', fontsize=fontsize + 10, fontweight='bold')

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize+5, width=2)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Make text annotations (heatmap values) bold and large
    for text in ax.texts:
        text.set_fontsize(fontsize)
        text.set_fontweight('bold')

    plt.grid(False)
    plt.tight_layout()

    # Save figure if path provided
    if cm_fig_save_path is not None:
        plt.savefig(cm_fig_save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Confmat plotting: Updated!!
def plot_confmat_new(
    y_test,
    y_pred,
    class_names,
    normalize='true',
    cm_title=None,
    cm_fig_save_path=None,
    figsize=None,
    fontsize=12,
    title_fontsize=None,
    title_pad=20,
    rotation=45,
):
    """
    Plot a confusion matrix with improved readability for many classes.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list
        List of class names
    cm_title : str, optional
        Title for the confusion matrix
    cm_fig_save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated based on number of classes
    fontsize : int, default=12
        Base font size for labels and values
    title_fontsize : int, optional
        Font size for title. If None, uses fontsize + 4
    title_pad : int, default=20
        Padding between title and plot
    rotation : int, default=45
        Rotation angle for x-axis labels
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np
    
    n_classes = len(class_names)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Scale figure size with number of classes (more generous spacing)
        size = max(10, min(30, n_classes * 0.9))
        figsize = (size, size)
    
    # Auto-calculate title font size if not provided
    if title_fontsize is None:
        if n_classes > 30:
            title_fontsize = fontsize + 50
        if n_classes > 20:
            title_fontsize = fontsize + 40
        else:
            title_fontsize = fontsize + 30
                 
    # Adjust font sizes based on number of classes
    if n_classes > 30:
        tick_fontsize = max(14, fontsize + 5)
        value_fontsize = max(9, fontsize)
    elif n_classes > 20:
        tick_fontsize = max(15, fontsize + 5)
        value_fontsize = max(10, fontsize + 2)
    else:
        tick_fontsize = fontsize + 2
        value_fontsize = fontsize
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the confusion matrix display
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize=normalize,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax,
        xticks_rotation=rotation,
        values_format=".2f",
        colorbar=False
    )
    
    # Set title with proper spacing
    if cm_title is not None:
        cm_display.ax_.set_title(
            cm_title, 
            fontsize=title_fontsize, 
            fontweight='bold', 
            pad=title_pad
        )
    
    # Set axis labels with adjusted positioning
    if n_classes > 30:
        # Get current tick labels and positions
        add_label_fontsize = 40
    elif n_classes > 20:
        # Get current tick labels and positions
        add_label_fontsize = 30
    else:
        add_label_fontsize = 20
    label_fontsize = fontsize + add_label_fontsize
    cm_display.ax_.set_xlabel(
        'Predicted Label', 
        fontsize=label_fontsize, 
        fontweight='bold',
        labelpad=10
    )
    cm_display.ax_.set_ylabel(
        'True Label', 
        fontsize=label_fontsize, 
        fontweight='bold',
        labelpad=10
    )
    
    # Customize tick labels with better spacing
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=tick_fontsize,
        width=2,
        length=8,
        pad=8  # Add padding between ticks and labels
    )
    
    # Make tick labels bold
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    
    # For many classes, alternate label positions or reduce crowding
    if n_classes > 15:
        # Get current tick labels and positions
        pass
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    
    # Adjust horizontal alignment for rotated labels
    for label in xlabels:
        label.set_ha('right')
            
    # Make text annotations (heatmap values) bold with adjusted size
    for text in ax.texts:
        text.set_fontsize(value_fontsize)
        text.set_fontweight('bold')
        # Improve contrast
        current_color = text.get_color()
        text.set_color(current_color)
    
    # Remove grid
    plt.grid(False)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Additional space adjustment for better spacing
    if rotation != 0:
        plt.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)
    else:
        plt.subplots_adjust(bottom=0.15, left=0.15, top=0.95, right=0.95)
    
    # Save figure if path provided
    if cm_fig_save_path is not None:
        plt.savefig(
            cm_fig_save_path, 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
    else:
        plt.show()
    plt.close()

    return fig, ax

'''
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
'''


# Mean classification scores of all batch splits
def compute_mean_batch_classify_metrics(batch_labels,task_dir):
    ##move to dutils
    print('[INFO]...Computing mean classification scores for all batches...')
    print('batch_labels ',batch_labels)

    dfs = []
    for batch_label in batch_labels:
        run_name = 'Batch_{batch}-vs-Rest'.format(batch=batch_label)
        batch_metrics_path = os.path.join(task_dir,run_name,'metrics_scores.csv')
        metrics_df = pd.read_csv(batch_metrics_path, index_col=0)
        dfs.append(metrics_df)
    
    all_batches_metrics = pd.concat(dfs).reset_index(drop=True)
    mean_metrics = all_batches_metrics.mean(axis=0)
    mean_metrics['batch_name'] = 'mean'
    all_batches_metrics = pd.concat([all_batches_metrics, pd.DataFrame(mean_metrics).T]).reset_index(drop=True)

    print('all_batches_metrics',all_batches_metrics)
    return all_batches_metrics


# Mean cluster scores of all batch splits
def compute_mean_batch_cluster_metrics(batch_labels,task_dir,res):
    ##move to dutils
    print(f'[INFO]...Computing mean cluster scores for all batches using res={res}...')
    print('batch_labels ',batch_labels)

    dfs = []
    for batch_label in batch_labels:
        run_name = 'Batch_{batch}'.format(batch=batch_label)
        batch_metrics_path = os.path.join(task_dir,run_name,f'res={res:.1f}','metrics_scores')
        metrics_df = pd.read_csv(batch_metrics_path, index_col=0)
        metrics_df['batch_name'] = batch_label
        dfs.append(metrics_df)
    
    all_batches_metrics = pd.concat(dfs).reset_index(drop=True)
    mean_metrics = all_batches_metrics.mean(axis=0)
    mean_metrics['batch_name'] = 'mean'
    all_batches_metrics = pd.concat([all_batches_metrics, pd.DataFrame(mean_metrics).T]).reset_index(drop=True)
    all_batches_metrics = all_batches_metrics[['batch_name'] + [col for col in all_batches_metrics.columns if col != 'batch_name']]
    print('all_batches_metrics',all_batches_metrics)
    return all_batches_metrics


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