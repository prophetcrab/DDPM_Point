import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def timeit(tag, t):
    print("{} : {}".format(tag, time.time() - t))
    return time.time()


#point_cloud_normalize
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def squared_distance(src, dst):
    """
    calculate euclid distance between each tow points.
    :param src:source points [B, N, C]
    :param dst:target points [B, M, C]
    :return:dist:per-point squared distance [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    根据idx的索引从points中提取对应的点
    :param points: source points [B, N, C]
    :param idx: [B, S]
    :return:
        new_points:, index points data [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) #(B, S)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    采样最远距离点
    :param xyz:  [B, N, 3]
    :param npoint: [number of samples]
    :return:sampled pointcloud index [B, index]
    """

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #[1, 32]
    distance = torch.ones(B, N).to(device) * 1e10 #[1,128]
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) #[1]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        farthest = torch.max(distance, -1)[1]
    return centroids



def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    根据给出的点列表查询每个点半径radius内的nsample个点
    :param radius:local region radius 查询半径
    :param nsample:max sample number in local region 查询点的数量
    :param xyz: all points [B, N, 3] 点云
    :param new_xyz: query points [B, S, 3] 查询列表
    :return:
        group_index: grouped points index, [B, S, nsample] B：batch_size S:需要查询的点的数量 nsample：每个查询点前n个的idx
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = squared_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points=None, returnfps=False):
    """
    :param npoint: 最远采样点数量
    :param radius: 球查询半径
    :param nsample: 查询点数量
    :param xyz: input points position data [B, N, 3] 点云位置数据
    :param points: input points data, [B, N, D] 点云数据本身
    :param returnfps:是否返回fps采样的点的列表
    :return:
        new_xyz: sampled points position data [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]  [batch [npoint [nsample [x, y, z]]]]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B,S] 返回最远点采样fps_idx
    new_xyz = index_points(xyz, fps_idx) #根据fps_idx查询new_xyz 这个newxyz是最远点采样的结果点 [1, 32, 3] [B, S, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz) #用new_xyz和xyz进行球查询，返回的是查询idx [B, S, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C] 用球查询返回的idx获取对应的点
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  #计算查询点对应的坐标相对于查询点的偏移，

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        """
        new_xyz : 根据fps_idx查询获得的最远采样点的结果点 [B, S, C]
        new_points: group结果的点对应查询点的偏移坐标 [B,npoint, nsample, 3+D]
        grouped_xyz : group结果的点的准确坐标[B, npoint, nsample, 3]
        fps_idx:最远采样的检索idx
        """
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, grouped_xyz

def sampled_and_group_all(xyz, points):
    """
    :param xyz: input point position data [B, N, 3]
    :param points: input points data [B, N, D]
    :return:
        new_xyz: sampled position data [B, 1, 3]
        new_points: sampled points data [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
def test():
    # 生成随机3D点
    B, N, S = 1, 1024, 32  # batch size, number of points, number of query points
    nsample = 64  # 每个查询点的采样数
    radius = 0.2  # 搜索半径

    xyz = torch.rand(B, N, 3)

    new_xyz, new_points = sampled_and_group_all(xyz, points=None)
    print(new_xyz.shape)
    print(new_points.shape)



if __name__ == '__main__':
    test()


