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
bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)


methods_tasks_list = [{'method_id':'liger_v1','task':'cluster'},{'method_id':'liger_v1','task':'batch_correct'},{'method_id':'scVI_v2','task':'cluster'},{'method_id':'scVI_v2','task':'batch_correct'}]
bench.eval_benchmark_multicore(methods_tasks_list)