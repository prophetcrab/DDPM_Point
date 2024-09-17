import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DataUtils.DataProcessUtils import voxel_downsample_point_cloud
from DataUtils.DataProcessUtils import visualize_point_cloud

def data_path_generation(directory, output_path, target_nums=50):
    count = 0  # 初始化计数器
    with open(output_path, 'w') as file:
        for root, dirs, files in os.walk(directory):
            for name in files:
                # 写入文件路径
                file.write(os.path.join(root, name) + '\n')
                count += 1  # 增加计数器

                # 如果达到指定的数量，停止写入
                if target_nums is not None and count >= target_nums:
                    return

def get_collate_fn(process_func=voxel_downsample_point_cloud, voxel_size=0.05, target_num_points=2048):

    def collate(batch):
        point_clouds = np.stack(batch, axis=0)
        results = process_func(point_clouds, voxel_size=voxel_size, target_num_points=target_num_points)
        return results

    return collate




# def collate_fn(batch, fn=voxel_downsample_point_cloud, voxel_size=0.05, target_num_points=2048):
#     point_clouds = np.stack(batch, axis=0)
#     result_clouds = fn(point_clouds, voxel_size=voxel_size, target_num_points=target_num_points)
#     return result_clouds

class PointDataSet(Dataset):
    """
    [B, N, 3]
    """
    def __init__(self,
                 file_path_list,
                 normalization=True,
                 voxel_sample=True, voxel_size=0.001,
                 random_sample=True, target_num_points=4096):
        self.normalization = normalization
        self.voxel_sample = voxel_sample
        self.voxel_size = voxel_size
        self.random_sample = random_sample
        self.target_num_points = target_num_points
        #从文本文件中读取所有文件路径
        with open(file_path_list, 'r') as f:
            self.file_paths = f.read().splitlines()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        data = data[:, :3]



        if self.voxel_sample:
            data = self.voxel_downsampling(data, voxel_size=self.voxel_size)
        if self.random_sample:
            data = self.random_sampling(data, target_num_points=self.target_num_points)

        if self.normalization:
            data = self.normalize(data)


        return data
    def normalize(self, data):
        min_coords = np.min(data, axis=0)
        max_coords = np.max(data, axis=0)

        range_coords = max_coords - min_coords

        max_range = np.max(range_coords)

        normalized_data = (data - min_coords) / max_range
        return normalized_data

    def voxel_downsampling(self, data, voxel_size):

        min_coords = np.min(data, axis=0)
        voxel_indices = np.floor((data - min_coords) / voxel_size).astype(int)
        voxel_grid = {}

        for point, voxel_idx in zip(data, voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in voxel_grid:
                voxel_grid[voxel_key] = []
            voxel_grid[voxel_key].append(point)

        downsampled_point_cloud = np.array([np.mean(points, axis=0) for points in voxel_grid.values()])


        return downsampled_point_cloud

    def random_sampling(self, data, target_num_points):
        if target_num_points >= data.shape[0]:
            return data
        selected_indices = np.random.choice(data.shape[0], target_num_points, replace=False)
        downsampled_point_cloud = data[selected_indices]
        return downsampled_point_cloud

if __name__ == '__main__':
    directory = r"D:\Data\modelnet40_normal_resampled\airplane"
    data_path = r"D:\PythonProject2\DDPM_Point\dataList.txt"
    target_nums = 300
    data_path_generation(directory, data_path, target_nums)

