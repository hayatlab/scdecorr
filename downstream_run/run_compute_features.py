from benchmarking import Benchmark
import scanpy as sc
import time

print('run')


dataset_root_dir='/home/noco0013/projects/contrastive_learning/results/runs/crosstissue_immune'
adata_fname='t-cells-raw-counts.h5ad'

dataset_name='crosstissue_immune'
scvi_max_epochs = 100
cell_class_obs_name="Manually_curated_celltype"
batch_obs_name="Chemistry"

checkpoint_root = '/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_crosstissue_immune_cells_unscaled_multi_dom_train_dsbn_minibatch=2048_optim=adam/epoch_720'
method_ids = ['harmony_v1']


for method_id in method_ids:
    bench = Benchmark(dataset_root_dir,adata_fname,dataset_name,cell_class_obs_name,batch_obs_name)
    bench.compute_features(method_id,save_features=True,scvi_max_epochs=scvi_max_epochs,checkpoint_root=checkpoint_root)
    time.sleep(1)

