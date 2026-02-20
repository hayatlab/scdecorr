from scdecorr.benchmarking import Benchmark
import scanpy as sc
import time

print('run')

dataset_root_dir='/data/human_lungs'
adata_fname='adata.h5ad'

dataset_name='human_lungs'
cell_class_obs_name="cell_type"
batch_obs_name="batch"

checkpoint_root = 'path/to/checkpoint'
methods = ['scdecorr', 'scVI', 'harmony', 'scanorama', 'liger', 'trvae-sca', 'pca', 'scalex', 'sccobra']
method_ids = [f'{method}_v1' for method in methods]

print('Started computing features....')
for method_id in method_ids:
    bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)
    bench.compute_features(method_id,save_features=True,checkpoint_root=checkpoint_root)
    time.sleep(1)
