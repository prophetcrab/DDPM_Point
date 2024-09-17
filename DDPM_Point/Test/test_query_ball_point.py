from PointNet.pointnet_utils import query_ball_point, pc_normalize

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

B = 1
N = 128
S = 32
C = 3  # 3D points

# Randomly generate 1000 points for xyz
xyz = torch.rand(B, N, C)

# Randomly generate 3 query points for new_xyz
new_xyz = torch.rand(B, S, C)

# Parameters
radius = 0.3
nsample = 16


group_idx = query_ball_point(radius, nsample, xyz, new_xyz)


print(group_idx.shape)
print(group_idx)
# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all points
ax.scatter(xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2], c='blue', label='All Points', s=10)

# Plot query points
ax.scatter(new_xyz[0, :, 0], new_xyz[0, :, 1], new_xyz[0, :, 2], c='red', label='Query Points', s=50)

# Plot lines connecting query points to their neighbors
for i in range(S):
    for j in group_idx[0, i]:
        if j < N:
            ax.plot([new_xyz[0, i, 0], xyz[0, j, 0]],
                    [new_xyz[0, i, 1], xyz[0, j, 1]],
                    [new_xyz[0, i, 2], xyz[0, j, 2]], 'k--')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()