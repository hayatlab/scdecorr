import yaml


dataset_cfg = {\
    'human_immune': {\
                        'adata_fname':'Immune_ALL_human.h5ad',\
                        'cell_class_obs_name':'final_annotation',\
                        'batch_obs_name':'batch',\
                        'checkpoint_root':'/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet11_immhuman_unscaled_multi_dom_train_dsbn_minibatch=512_projector=512-512-512_optim=adam/epoch_500',\
                        'in_features':2000,\
                        'arch':'densenet11'\
                },\

    'human_pancreas': {\
                        'adata_fname':'human_pancreas_norm_unscaled.h5ad',\
                        'cell_class_obs_name':'celltype',\
                        'batch_obs_name':'tech',\
                        'checkpoint_root':'/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet11_pancreas_unscaled_multi_dom_train_dsbn_minibatch=512_projector_512-512-512_optim=adam/epoch_499',\
                        'in_features':2000,\
                        'arch':'densenet11'\
                    },\

    'Muris': {'adata_fname':'muris_sample_filter.h5ad',\
               'cell_class_obs_name':'cell_ontology_class',\
                'batch_obs_name':'batch',\
                'checkpoint_root':'/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_muris_hvg_5000_unscaled_multi_dom_train_dsbn_minibatch=2048_optim=adam/epoch_800',\
                'in_features':5000,\
                'arch':'densenet21'\
            },\

    'human_lungs': {\
                    'adata_fname':'Lung_atlas_public_original_adata_X_unscaled.h5ad',\
                    'cell_class_obs_name':'cell_type',\
                    'batch_obs_name':'batch',\
                    'checkpoint_root':'/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet11_lungs_original_X_unscaled_multi_dom_train_dsbn_minibatch=512_projector=512-512-512_optim=adam/epoch_999',\
                    'in_features':2000,\
                    'arch':'densenet11'\
                },\


    'crosstissue_immune': {\
                    'adata_fname':'t-cells-raw-counts.h5ad',\
                    'cell_class_obs_name':'Manually_curated_celltype',\
                    'batch_obs_name':'Chemistry',\
                    'checkpoint_root':'/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_crosstissue_immune_cells_unscaled_multi_dom_train_dsbn_minibatch=2048_optim=adam/epoch_720',\
                    'in_features':2000,\
                    'arch':'densenet21'\
                },\

    'broad+hca': {'adata_fname':'',\
                    'cell_class_obs_name':'',\
                    'batch_obs_name':'',\
                    'checkpoint_root':''},\


    'kidney': {'adata_fname':'',\
                    'cell_class_obs_name':'',\
                    'batch_obs_name':'',\
                    'checkpoint_root':''},\
}\

with open('dataset_cfg.yaml','w') as f:
    yaml.dump(dataset_cfg,f)