from benchmarking import Benchmark
import scanpy as sc


dataset_root_dir='/home/noco0013/projects/contrastive_learning/results/runs/human_lungs'
adata_fname='Lung_atlas_public_original_adata_X_unscaled.h5ad'

dataset_name='human_lungs'
cell_class_obs_name="cell_type"
batch_obs_name="batch"
sc_figdir = "/home/noco0013/projects/contrastive_learning/results/figures"
sc.settings.figdir = sc_figdir

print('run')
print('benchmark2 run',flush=True)
bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)

method_ids = ['harmony_v1','liger_v1','bbknn_v1']
cluster_metrics=['ari','nmi','silhouette','isolated_asw','isolated_f1']

bench.eval_benchmark_cluster_and_batch(method_ids=method_ids,cluster_metrics=cluster_metrics,n_jobs=-1)