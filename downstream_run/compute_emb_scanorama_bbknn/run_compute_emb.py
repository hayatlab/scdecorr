
from downstream_utils import compute_mul_embs_mul_data



print('run')

datasets = ['human_immune','human_pancreas','Muris']
root_dir='/home/noco0013/projects/contrastive_learning/results/runs'

method_ids_2 = ['bbknn_v1','scanorama_v1']

compute_mul_embs_mul_data(datasets,root_dir,method_ids_2)
