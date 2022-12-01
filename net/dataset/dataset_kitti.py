from math import e
import torch
import numpy as np
import torch.utils.data as data
import os
from plyfile import PlyData

def get_dataloader_kitti(phase, config):
    is_shuffle = phase == 'train'

    dataset = KittiPly(config.dataset_path)

    dataloader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader, dataset

class KittiPly(data.Dataset):
    def __init__(self, dataset_path):
        super(KittiPly, self).__init__()
        print("using kitty dataset...")

        self.proj_dir = dataset_path
        self.val_path = os.path.join(self.proj_dir, "point_cloud_val")
        
        self.val_filename_list = []
        for i in range(len(os.listdir(self.val_path))):
            filename = str(i) + ".ply"
            self.val_filename_list.append(os.path.join(self.val_path, filename))
        
        self.len = len(self.val_filename_list)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        filename = self.val_filename_list[index]
        points = self.read_ply_xyz(filename)
        points = rotate_point_cloud_by_axis_angle(points, [0,-1,0], 90)
        furthest_distance = np.amax(
            np.sqrt(np.sum(points ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True
        )
        points = points / (2 * furthest_distance)
        points = torch.from_numpy(points)
        return {'raw':points.t().float()}
    
    def read_ply_xyz(self, filename):
        """ read XYZ point cloud from filename PLY file """
        if not os.path.isfile(filename):
            print(filename)
            assert(os.path.isfile(filename))
        with open(filename, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']
        return vertices

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
