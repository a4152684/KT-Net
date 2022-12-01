from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py

def get_dataloader_crn(phase, config):
    is_shuffle = phase == 'train'

    dataset = CRNShapeNet(config.dataset_path, phase, config.category, 'unpair')

    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    
    if is_shuffle:
        val_dataset = CRNShapeNet(config.dataset_path, 'val', config.category, 'unpair')
        return dataloader, val_dataset
    else:
        return dataloader, dataset
    
class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, dataset_path, split='train', category = 'plane', mode="unpair"):
        self.dataset_path = dataset_path
        self.category = category
        self.split = split
        
        if split == "val":
            self.split = "valid"

        if category == 'sofa':
            self.category = 'couch'
        print("using " + mode + " " + split + " CRN dataset " + category + "...")
        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

        cat_id = cat_ordered_list.index(self.category.lower())
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])

        if mode == "unpair":
            np.random.seed(0)          
            self.index_random = np.random.permutation(self.index_list)
        else:
            self.index_random = self.index_list

    def __getitem__(self, index):
        partial_full_idx = self.index_list[index]
        if self.split == "train":
            gt_full_idx = self.index_random[index]
            gt = torch.from_numpy(self.gt[gt_full_idx])
        else:
            gt = torch.from_numpy(self.gt[partial_full_idx])
        partial = torch.from_numpy(self.partial[partial_full_idx])
        return {"raw": partial.t(), "real": gt.t()}

    def __len__(self):
        return len(self.index_list)

