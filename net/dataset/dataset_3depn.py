import torch
import numpy as np
import torch.utils.data as data
import os
import random
import glob

def get_dataloader_3depn(phase, config):
    is_shuffle = phase == 'train'

    dataset = GANCycleNpy(config.dataset_path, phase, config.category, 'unpair')

    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    
    if is_shuffle:
        val_dataset = GANCycleNpy(config.dataset_path, 'val', config.category)
        return dataloader,val_dataset
    else:
        return dataloader, dataset
    
encode = {
    "chair": "03001627",
    "table": "04379243",
    "sofa": "04256520",
    "cabinet": "02933112",
    "lamp": "03636649",
    "car": "02958343",
    "plane": "02691156",
    "watercraft": "04530566"
}

class GANCycleNpy(data.Dataset):
    def __init__(self, dataset_path, phase = 'train', category = 'plane', mode='unpair'):
        super(GANCycleNpy, self).__init__()
        # train data only has input(2048) and gt(2048)

        print("using " + mode + " cycle dataset " + category)

        self.is_train = phase == "train"
        self.is_val = phase == "val"

        proj_path = dataset_path
        save_path_train = os.path.join(proj_path, category ,"partial")
        save_path_gt = os.path.join(proj_path, category, "complete")
        f_lidar = glob.glob(os.path.join(save_path_train, '*.npy'))

        if self.is_train:
            self.train_raw_filename = []
            self.train_gt_filename = []
        else:
            self.test_raw_filename = []
            self.test_gt_filename = []

        a = np.loadtxt(os.path.join(proj_path, "split_pcl2pcl.txt"), str)

        b = []
        for i in a:
            if int(i[:8]) == int(encode[category]):
                i = i[9:]
                b.append(i)

        for i in f_lidar:
            file = i.split('/')[-1].split('.')[0][:-5]+".npy"
            if file[:-4] in b:
                if self.is_train == False:
                    """ if len(self.test_gt_filename) >= 1:
                        continue """
                    self.test_raw_filename.append(i)
                    self.test_gt_filename.append(os.path.join(save_path_gt,file))
            else:
                if self.is_train:
                    """ if len(self.train_gt_filename) >= 1:
                        continue """
                    self.train_raw_filename.append(i)
                    self.train_gt_filename.append(os.path.join(save_path_gt,file))
                elif self.is_val:
                    self.test_raw_filename.append(i)
                    self.test_gt_filename.append(os.path.join(save_path_gt,file))
        
        if self.is_train:      
            if mode == "unpair":
                # shuffle training datasets
                random.seed(0)
                random.shuffle(self.train_gt_filename)
                
        # val: 200
        if self.is_val:
            rand_ids = np.random.permutation(len(self.test_gt_filename))
            val_num = 200
            self.test_gt_filename = np.array(self.test_gt_filename)[rand_ids[:val_num]]
            self.test_raw_filename = np.array(self.test_raw_filename)[rand_ids[:val_num]]
            
        if self.is_train:
            self.len = len(self.train_gt_filename)
        else:
            self.len = len(self.test_gt_filename)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.is_train:
            raw_lidar =torch.from_numpy(np.load(self.train_raw_filename[index])).float()
            gt_lidar =torch.from_numpy(np.load(self.train_gt_filename[index])).float()
            return {"raw": raw_lidar.t(), "real": gt_lidar.t()}
        else:
            raw_lidar =torch.from_numpy(np.load(self.test_raw_filename[index])).float()
            gt_lidar =torch.from_numpy(np.load(self.test_gt_filename[index])).float()
            return {"raw": raw_lidar.t(), "real": gt_lidar.t()}

if __name__ == "__main__":
    test = GANCycleNpy('test','plane')
    print(len(test))