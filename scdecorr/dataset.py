import torch
import scanpy as sc

class TwoBatchDataset(torch.utils.data.Dataset):
    def __init__(self, adata_path, batch_obs1, batch_obs2, batch_obs_name='batch',data_obsm_name=None , transform=None):
        
        adata = sc.read_h5ad(adata_path)
        print('adata loaded...',adata)

        print('Batch names',batch_obs1,batch_obs2)
        adata_batch1 = adata[adata.obs[batch_obs_name]==batch_obs1]
        print('adata batch1...',adata_batch1)
        adata_batch2 = adata[adata.obs[batch_obs_name]==batch_obs2]
        print('adata batch2...',adata_batch2)


        if not data_obsm_name:
            self.batch1 = adata_batch1.X.toarray()
            self.batch2 = adata_batch2.X.toarray()
            print('self.batch1.max(),self.batch2.max()',self.batch1.max(),self.batch2.max())
        else:
            self.batch1 = adata_batch1.obsm[data_obsm_name].toarray()
            self.batch2 = adata_batch2.obsm[data_obsm_name].toarray()

        print('self.batch1.shape ',self.batch1.shape)
        print('self.batch2.shape ',self.batch2.shape)
        self.transform = transform

    def __getitem__(self, i):
        batch1_item = self.batch1[i % len(self.batch1),:]
        batch2_item = self.batch2[i % len(self.batch2),:]

        if self.transform:
            batch1_item_transformed = self.transform(batch1_item)
            batch2_item_transformed = self.transform(batch2_item)

        #return (batch1_item_transformed[0],batch1_item_transformed[1],batch2_item_transformed[0],batch2_item_transformed[1])
        return (batch1_item_transformed,batch2_item_transformed)

    def __len__(self):
        return max(len(self.batch1),len(self.batch2))

class MultiBatchDataset(torch.utils.data.Dataset):

    #n_domains = number of experimental batches
    def __init__(self, adata_path, n_domains, domain_obs_name='batch_int',data_obsm_name=None , transform=None):

        adata = sc.read_h5ad(adata_path)
        print('adata loaded...',adata)

        #sanity check: n_domains should be eq to number of experimental batches in the adata
        assert len(set(adata.obs[domain_obs_name].tolist())) == n_domains

        #list of adatas of all the domains
        adata_domains = list(map(lambda i: adata[adata.obs[domain_obs_name]== i], range(n_domains)))

        for i in range(n_domains):
            print('adata domain ',i,' \n ', adata_domains[i])

        if not data_obsm_name:
            self.X_domains = list(map(lambda adata_domain:adata_domain.X.toarray(),adata_domains))
        else:
            self.X_domains = list(map(lambda adata_domain:adata_domain.obsm[data_obsm_name].toarray(),adata_domains))

        for i in range(n_domains):
            print('X.shape of domain ',i,' \n ', self.X_domains[i].shape)

        self.transform = transform

        print('self.__len__()',self.__len__())


    def __getitem__(self, i):

        domain_items = list(map(lambda X_domain:X_domain[i % len(X_domain),:],self.X_domains))

        if self.transform:

            domain_items_transformed = list(map(lambda domain_item:self.transform(domain_item),domain_items))

        #return (batch1_item_transformed[0],batch1_item_transformed[1],batch2_item_transformed[0],batch2_item_transformed[1])
        return domain_items_transformed

    def __len__(self):
        return max(list(map(lambda X_domain:len(X_domain),self.X_domains)))


class TwoBatchMMDDataset(TwoBatchDataset):

    def __init__(self, adata_path, batch_obs1, batch_obs2, batch_obs_name='batch',data_obsm_name=None , transform=None):

        super().__init__(adata_path, batch_obs1, batch_obs2, batch_obs_name,data_obsm_name,transform)


    def __getitem__(self, i):
        batch1_item = self.batch1[i % len(self.batch1),:]
        batch2_item = self.batch2[i % len(self.batch2),:]

        if self.transform:
            batch1_item_transformed = self.transform(batch1_item)
            batch2_item_transformed = self.transform(batch2_item)

        return ([batch1_item,batch1_item_transformed[0],batch1_item_transformed[1]],[batch2_item,batch2_item_transformed[0],batch2_item_transformed[1]])



class SingleBatchDataset(torch.utils.data.Dataset):
    def __init__(self, X, transform=None):
        """x: rows are genes and columns are samples"""
        self.X = X
        self.transform = transform
        
    def __getitem__(self, i):
        x = self.X[i,:]
        if self.transform:
            x = self.transform(x)
        return x, 0
    
    def __len__(self):
        return len(self.X)