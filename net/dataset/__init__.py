from dataset.dataset_crn import get_dataloader_crn
from dataset.dataset_3depn import get_dataloader_3depn
from dataset.dataset_kitti import get_dataloader_kitti
from dataset.dataset_scannet import get_dataloader_scan
from dataset.dataset_matterport import get_dataloader_mat

def get_dataloader(phase, config):
    if config.dataset_name == 'crn':
        return get_dataloader_crn(phase, config)
    elif config.dataset_name == '3depn':
        return get_dataloader_3depn(phase, config)
    elif config.dataset_name == 'kitty':
        return get_dataloader_kitti(phase, config)
    elif config.dataset_name == 'scan':
        return get_dataloader_scan(phase, config)
    elif config.dataset_name == 'mat':
        return get_dataloader_mat(phase, config)
    else:
        raise ValueError
