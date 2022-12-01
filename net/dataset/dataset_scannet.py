import torch
import numpy as np
import torch.utils.data as data
import h5py

def get_dataloader_scan(phase, config):
    is_shuffle = phase == 'train'

    dataset = ScanH5(config.dataset_path, phase, config.category)

    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader, dataset

class ScanH5(data.Dataset):
    def __init__(self, dataset_path, phase="test", category="chair"):
        super(ScanH5, self).__init__()
        
        category = category + "s"
        print("using scan dataset " + category + "...")

        self.is_train = phase == "train"

        self.filename = dataset_path
        f = h5py.File(self.filename, 'r')

        """ for name in f['input']['chairs']:
            print(torch.from_numpy(np.asarray(f['input']['chairs'][name])))
            exit() """
        
        self.points = []
        for name in f['input'][category]:
            points = np.asarray(f['input'][category][name])
            furthest_distance = np.amax(
                np.sqrt(np.sum(points ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True
            )
            points = points / (2 * furthest_distance)
            self.points.append(torch.from_numpy(points).float())
        self.len = len(self.points)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {'raw': self.points[index].t()}
