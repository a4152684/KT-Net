import torch
import numpy as np
import torch.utils.data as data
import os
from plyfile import PlyData

def get_dataloader_mat(phase, config):
    is_shuffle = phase == 'train'

    dataset = MatObj(config.dataset_path, config.category)

    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader, dataset

class MatObj(data.Dataset):
    def __init__(self, dataset_path, category='chair'):
        super(MatObj, self).__init__()
        print("using Matter Port 3D dataset " + category + "...")

        self.proj_dir = dataset_path
        self.val_path = os.path.join(self.proj_dir, "mat_" + category)
        
        self.val_filename_list = []
        for filename in os.listdir(self.val_path):
            self.val_filename_list.append(os.path.join(self.val_path, filename))
        
        self.len = len(self.val_filename_list)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        filename = self.val_filename_list[index]
        
        with open(filename, "r") as f:
            points = []
            while 1:
                line = f.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "vt":
                    break
        
        points = np.asarray(points)

        points = points - points.mean(0)

        points = rotate_point_cloud_by_axis_angle(points, [0,-1,0], 90)
        
        furthest_distance = np.amax(
            np.sqrt(np.sum(points ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True
        )
        points = points / (2 * furthest_distance)

        points = torch.from_numpy(points)
        return {'raw':points.t().float()}
    

def rotate_point_cloud(points, transformation_mat):

    new_points = np.dot(transformation_mat, points.T).T

    return new_points

def rotate_point_cloud_by_axis_angle(points, axis, angle_deg):

    """ angle = math.radians(angle_deg)
    rot_m = pymesh.Quaternion.fromAxisAngle(axis, angle)
    rot_m = rot_m.to_matrix() """
    rot_m = np.asarray([[0,0,1],[0,1,0],[-1,0,0]])

    new_points = rotate_point_cloud(points, rot_m)

    rot_m = np.asarray([[-1,0,0],[0,1,0],[0,0,1]])

    new_points = rotate_point_cloud(new_points, rot_m)

    return new_points
