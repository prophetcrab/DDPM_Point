import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PointNet.pointnet_utils import sample_and_group

# 测试数据 (使用之前的)
B = 1
N = 128
S = 16
C = 3  # 3D points

# 随机生成 1000 个点的 xyz
xyz = torch.rand(B, N, C)

# 参数
npoint = 8  # 采样点的数量
radius = 0.1
nsample = 10

# 运行函数
new_xyz, new_points = sample_and_group(npoint, radius, nsample, xyz, points=None)

print(new_xyz.shape)
print(new_points.shape)

print(new_xyz)
# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制所有点
ax.scatter(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], c='blue', label='All Points', s=10)

# 绘制采样点
ax.scatter(new_xyz[0, :, 0], new_xyz[0, :, 1], new_xyz[0, :, 2], c='red', label='Sampled Points', s=50)

# 连接采样点与分组点
for i in range(npoint):
    for j in range(nsample):
        ax.plot([new_xyz[0, i, 0], new_points[0, i, j, 0]],
                [new_xyz[0, i, 1], new_points[0, i, j, 1]],
                [new_xyz[0, i, 2], new_points[0, i, j, 2]], 'k--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()