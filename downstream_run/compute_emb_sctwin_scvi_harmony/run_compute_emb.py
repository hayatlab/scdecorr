from downstream_utils import compute_mul_embs_mul_data


print('run')

method_ids = ['harmony_v1','sctwins-dsbn_v1','scVI_v1']

datasets = ['human_immune','human_pancreas','Muris']
root_dir='/home/noco0013/projects/contrastive_learning/results/runs'


compute_mul_embs_mul_data(datasets,root_dir,method_ids)
