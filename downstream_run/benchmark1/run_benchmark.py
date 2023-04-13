from benchmarking import Benchmark
import scanpy as sc


dataset_root_dir='/home/noco0013/projects/contrastive_learning/results/runs/crosstissue_immune'
adata_fname='t-cells-raw-counts.h5ad'

dataset_name='crosstissue_immune'
cell_class_obs_name="Manually_curated_celltype"
batch_obs_name="Chemistry"
sc_figdir = "/home/noco0013/projects/contrastive_learning/results/figures"
sc.settings.figdir = sc_figdir

print('run')
print('benchmark run',flush=True)
bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)

method_ids = ['bbknn_v1']
bench.eval_benchmark_cluster(method_ids)