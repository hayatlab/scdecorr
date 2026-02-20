from scdecorr.benchmarking import Benchmark
import scanpy as sc


dataset_root_dir='/data/human_lungs'
adata_fname='adata.h5ad'

dataset_name='human_lungs'
cell_class_obs_name="cell_type"
batch_obs_name="batch"
sc_figdir = "/results/figures"
sc.settings.figdir = sc_figdir

print('Started evaluation....')
bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)

methods = ['scdecorr', 'scVI', 'harmony', 'scanorama', 'liger', 'trvae-sca', 'pca', 'scalex', 'sccobra']
methods_tasks_list = [{'method_id': f'{method}_v1', 'task': 'cluster'} for method in methods]
bench.eval_benchmark_multicore(methods_tasks_list)

methods = ['scdecorr', 'scVI', 'harmony', 'scanorama', 'liger', 'trvae-sca', 'pca', 'scalex', 'sccobra']
methods_tasks_list = [{'method_id': f'{method}_v1', 'task': 'batch_correct'} for method in methods]
bench.eval_benchmark_multicore(methods_tasks_list)

methods = ['scdecorr', 'scVI', 'harmony', 'scanorama', 'liger', 'trvae-sca', 'pca', 'scalex', 'sccobra']
methods_tasks_list = [{'method_id': f'{method}_v1', 'task': 'classify'} for method in methods]
bench.eval_benchmark(methods_tasks_list)