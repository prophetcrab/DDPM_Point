import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(point_cloud, color=(0, 0, 255), size=0.1):
    """
    :param point_cloud: [N, C]
    :param color:
    :param size:
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color(color)

    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=(0, 0, 0))

    o3d.visualization.draw_geometries([pcd, axis_frame])

def visualize_pint_cloud_with_plt(point_cloud):
    """
        使用 matplotlib 可视化三维点云

        参数:
        - point_cloud: numpy 数组，形状为 (N, 3)，表示点云数据
        """
    # 创建一个新的图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o', s=1)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()


def read_pointcould_from_file(filepath):
    with open(filepath, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
        data = data[:, :3]
        data = data.T
        return data

def voxel_downsample_point_cloud(point_clouds, voxel_size=0.05, target_num_points=2048):
    """
    对形状为 (B, C, N) 的批次点云进行体素下采样，并调整点数到目标大小 N。

    参数:
    - point_clouds: numpy 数组，形状为 (B, C, N)，表示 B 个点云批次，每个点云有 N 个点，每个点有 C 个通道。
    - voxel_size: 体素大小，控制下采样的粒度，默认值为 0.05。
    - target_num_points: 目标点数 N，下采样后的每个点云将调整到此数量。

    返回:
    - downsampled_batch: 下采样并调整后的点云数据，形状为 (B, C, target_num_points)。
    """
    B, C, N = point_clouds.shape
    downsampled_result = np.zeros((B, C, target_num_points))

    for i in range(B):
        # 将单个点云数据提取出来，并转换为 (N, 3) 形式
        point_cloud = point_clouds[i].T  # 转置为 (N, C)

        # 创建 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # 执行体素下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # 获取下采样后的点并转换为 numpy 数组
        downsampled_points = np.asarray(downsampled_pcd.points).T  # 转置回 (C, M) 形式
        M = downsampled_points.shape[1]

        # 调整点数到目标大小
        if M >= target_num_points:
            # 如果下采样后的点数大于目标点数，随机选择 target_num_points 个点
            selected_indices = np.random.choice(M, target_num_points, replace=False)
            downsampled_points = downsampled_points[:, selected_indices]
        else:
            # 如果下采样后的点数少于目标点数，随机重复点来填充
            repeated_indices = np.random.choice(M, target_num_points - M, replace=True)
            downsampled_points = np.concatenate([downsampled_points, downsampled_points[:, repeated_indices]], axis=1)

        # 将处理后的点云添加到结果数组中
        downsampled_result[i] = downsampled_points

    return downsampled_result


if __name__ == '__main__':
    B, C, N = 128, 3, 5000
    point_cloud = np.random.rand(B, C, N)

    voxel_size = 0.05
    downsampled_point_cloud = voxel_downsample_point_cloud(point_cloud)
    #downsampled_point_cloud = downsample_point_cloud(downsampled_point_cloud)

    print(downsampled_point_cloud.shape)


