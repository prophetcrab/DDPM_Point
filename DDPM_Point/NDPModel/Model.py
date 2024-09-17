import torch
import torch.nn as nn
import torch.nn.functional as F
from NDPModel.NDPUtils import dot_product_attention
from NDPModel.NDPUtils import preprocess
from NDPModel.NDPUtils import PositionalEmbedding
from NDPModel.SparseAttention.flash_attention import *
from NDPModel.SparseAttention.memory_efficient_attention import *

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        #先别用其他类型的注意力，只用点积
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        #assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)


        self.attention = dot_product_attention


    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)


        q = q.view(q.size(0), -1, self.num_heads, self.depth).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.depth).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.depth).transpose(1, 2)


        if mask is not None:
            # 重新排列掩码以适应多头形状
            mask = mask.unsqueeze(1).unsqueeze(2)

        # print("q shape", q.shape)
        # print("k shape", k.shape)
        # print("v shape", v.shape)
        # 计算注意力
        scaled_attention = self.attention(q, k, v, mask)


        # 将多头的结果合并并进行最后的线性变换
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(q.size(0), -1, self.d_model)
        output = self.dense(scaled_attention)


        return output

class BiDimensionalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.linear_d = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.linear_n = nn.Linear(self.hidden_dim, self.hidden_dim * 2)

        # 初始化线性层
        self.linear_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_d = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)
        self.attention_n = MultiHeadAttention(2 * self.hidden_dim, self.num_heads)

        #norm层
        self.group_norm1 = nn.GroupNorm(1,  3)
        self.group_norm2 = nn.GroupNorm(1,  3)

    def forward(self,
                s: torch.Tensor, t: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        双维度注意力块，主要用于NDP噪声模型的计算。
        :param s: [B, N, D, H]
        :param t: [B, H]
        :param mask:
        :return: [B, N, D, H]
        """
        B, N, D, H = s.shape

        t = self.linear_t(t)[:, None, None, : ]
        #y : (B, N, D, H)
        y = s + t
        #reshape 每个数据的维度,令其适应d计算
        y_d = y.view(B * N, D, H)
        y_r = y.permute(0, 2, 1, 3).view(B * D, N, H)

        y_d = self.linear_d(y_d)
        y_r = self.linear_n(y_r)

        #第一次注意力计算(D维度)
        y_attn_d = self.attention_d(y_d, y_d, y_d)

        if mask is not None:
            mask = mask.unsqueeze(1)

        #第二次注意力计算(N维度)
        y_attn_n = self.attention_n(y_r, y_r, y_r, mask)

        #reshape回标准格式
        y_attn_d = y_attn_d.view(B, N, D, H * 2)
        y_attn_n = y_attn_n.view(B, D, N, H * 2).permute(0, 2, 1, 3)

        y = y_attn_d + y_attn_n

        # 分割y得到残差连接和跳跃连接部分
        residual, skip = torch.split(y, self.hidden_dim, dim=-1)

        """"""
        # residual = residual.permute(0, 2, 1, 3).contiguous()
        skip = skip.permute(0, 2, 1, 3).contiguous()

        # residual = self.group_norm1(residual)
        #skip = self.group_norm2(skip)

        # residual = residual.permute(0, 2, 1, 3).contiguous()
        skip = skip.permute(0, 2, 1, 3).contiguous()
        """"""

        residual = F.gelu(residual)
        skip = F.gelu(skip)

        # 返回残差连接后的输出和跳跃连接部分
        return (s + residual) / math.sqrt(2.0), skip

class ResAttentionBlock(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.first_layer = BiDimensionalAttentionBlock(hidden_dim, num_heads)
        self.attn_layers = nn.ModuleList([BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)])


    def forward(self, s, t, mask = None):
        """
        :param s: [B, N, D, H]
        :param t: [B, H]
        :param mask:
        :return: skip.mean() [B, N, D, H]
        """
        skip = None
        res, skip = self.first_layer(s, t, mask)

        for layer in self.attn_layers:
            res, skip_connection = layer(res, t, mask)
            skip = skip_connection + skip



        return skip


class AttentionModel(nn.Module):
    def __init__(self, n_layers: int = 8, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.linear_1 = nn.Linear(2, hidden_dim)
        self.linear_2 = nn.Linear(128, hidden_dim)
        self.t_embed = PositionalEmbedding(128)
        self.model = ResAttentionBlock(n_layers, hidden_dim, num_heads)

        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, 1)

    def forward(self, xt, t, mask = None):
        """+
        :param xt: [B, N, D]
        :param yt: [B, N, 1]
        :param t: [B, 1]
        :param mask:
        :return:
        """
        B, N, D = xt.shape
        device = xt.device
        yt = torch.arange(0, N).unsqueeze(0).unsqueeze(2).to(device)
        st = preprocess(xt, yt) #[B, N, D, 2]
        st = self.linear_1(st) #[B, N, D, H]
        st = F.gelu(st)

        t = self.t_embed(t)  # [B, 128]
        t = self.linear_2(t) #[B, H]


        eps = self.model(st, t, mask)

        eps = eps / math.sqrt(self.n_layers * 1.0)

        """"""
        eps = F.gelu(self.linear_3(eps))

        eps = F.gelu(self.linear_4(eps))
        eps = eps.view(B, N, D)

        #eps = F.normalize(eps, p=2, dim=2)
        """"""
        #eps = F.gelu(eps)

        return eps





if __name__ == '__main__':
    B, N, D = 1, 128, 3
    xt = torch.randn(B, N, D)
    yt = torch.arange(0, N).unsqueeze(0).unsqueeze(2)
    st = preprocess(xt, yt)
    t = torch.randint(0, 1000, (B,))

    d_model = 512
    num_heads = 8
    n_layers = 4

    Model = AttentionModel()
    y = Model(xt, t)
    print(y.shape)
