import torch
import math
import torch.nn.functional as F
import torch.nn as nn

def preprocess(xt, yt):
    """
    :param xt: [B, N, D]
    :param yt: [B, N, 1]
    :return: result: [B, N, D, 2]
    """
    B, N, D = xt.shape
    yt = yt.expand(-1, -1, D)
    st = torch.stack((xt, yt), dim=-1)
    return st


def dot_product_attention(Q, K, V, mask=None):
    """
    计算点积注意力
    参数:
    - Q: 查询张量，形状为 (batch_size, num_heads, seq_len_q, d_k)
    - K: 键张量，形状为 (batch_size, num_heads, seq_len_k, d_k)
    - V: 值张量，形状为 (batch_size, num_heads, seq_len_v, d_v)
    - mask: 掩码张量，用于防止某些位置的注意力计算，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)

    返回:
    - output: 计算后的点积注意力，形状为 (batch_size, num_heads, seq_len_q, d_v)
    - attention_weights: 注意力权重，形状为 (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    #计算QK^T,并且缩放分数
    scores = torch.matmul(Q, K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device)))

    # 如果存在掩码，将掩码加到分数上
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 用注意力权重对值进行加权求和
    output = torch.matmul(attention_weights, V)

    return output


class PositionalEmbedding(nn.Module):

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 #N 8
        emb = math.log(10000) / half_dim #N 1.15
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # e^[[0,1,2...7]*(-1.15)]  [8]
        emb = torch.outer(x * self.scale, emb) #[len(x), len(emb)]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        #[len(x), len(emb)*2]
        return emb


if __name__ == '__main__':
    # 设置张量维度
    t = torch.randint(0, 1000, (1,))

    emb = PositionalEmbedding(128)

    emb_t = emb(t)
    print(t.shape)
    print(emb_t.shape)