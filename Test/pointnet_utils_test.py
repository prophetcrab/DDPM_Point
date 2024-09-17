import torch

def farthest_point_sample(xyz, npoint):
    """
    采样最远点，输入形状为 [B, C, N]
    :param xyz: 输入点云，形状为 [B, C, N]
    :param npoint: 需要采样的点数
    :return: 采样的点云索引，形状为 [B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape  # 获取批次大小、通道数和点数

    # 初始化采样中心索引
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)

    # 初始化距离张量
    distance = torch.ones(B, N).to(device) * 1e10

    # 随机选择初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    # 创建批次索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        # 记录当前最远点的索引
        centroids[:, i] = farthest

        # 使用 gather 函数获取每个批次中最远点的坐标
        # [B, C, 1]: 选取每个批次中的当前最远点的坐标
        centroid = xyz.gather(2, farthest.view(B, 1, 1).expand(-1, C, -1))  # 形状为 [B, C, 1]

        # 计算每个点到当前最远点的欧几里得距离的平方
        dist = torch.sum((xyz - centroid) ** 2, dim=1)

        # 使用 torch.where 更新每个点到最近采样点的距离
        distance = torch.where(dist < distance, dist, distance)

        # 选择下一个最远点
        farthest = torch.max(distance, -1)[1]

    return centroids

def main():
    # 创建测试数据
    B, C, N = 1, 3, 5  # 设置批次大小、通道数、点数量
    xyz = torch.tensor([[
        [0.0, 1.0, 2.0, 3.0, 4.0],  # 第一通道（x 坐标）
        [0.0, 1.0, 2.0, 3.0, 4.0],  # 第二通道（y 坐标）
        [0.0, 1.0, 2.0, 3.0, 4.0]   # 第三通道（z 坐标）
    ]])  # 形状为 [1, 3, 5]

    npoint = 3  # 需要采样的点数量

    # 使用函数计算采样的最远点索引
    sampled_indices = farthest_point_sample(xyz, npoint)

    # 打印结果
    print("输入点云 (xyz):", xyz)
    print("采样到的点索引 (sampled_indices):", sampled_indices)

    # 手动验证结果
    # 预期采样点索引为 [0, 4, 2]
    expected_indices = torch.tensor([[0, 4, 2]])

    print("手动计算的采样点索引 (expected_indices):", expected_indices)

    # 验证结果是否正确
    if torch.equal(sampled_indices, expected_indices):
        print("测试通过: 采样结果正确！")
    else:
        print("测试失败: 采样结果不正确！")

if __name__ == "__main__":
    main()
