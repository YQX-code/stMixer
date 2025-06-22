import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
from tqdm import tqdm

import torch
import numpy as np
from tqdm import tqdm
import torch_geometric

import pandas as pd


def build_graph(positions, modality, k=6):
    """
    构建图的函数，基于位置的k近邻和模态相似度。

    参数：
        positions (torch.Tensor): 细胞的位置张量，形状为 [num_cells, position_dim]。
        modality (torch.Tensor): 细胞的模态特征张量，形状为 [num_cells, modality_dim]。
        k (int): 每个细胞的邻居数量。

    返回：
        edge_index (torch.LongTensor): 边的索引，形状为 [2, num_edges]。
        edge_weight (torch.FloatTensor): 边的权重，形状为 [num_edges]。
    """
    device = positions.device  # 获取当前设备
    num_cells = positions.shape[0]
    import torch.nn.functional as F
    # 归一化模态特征以计算余弦相似度
    modality_normalized = F.normalize(modality, p=2, dim=1)

    # 计算所有细胞之间的欧氏距离
    with torch.no_grad():
        distance_matrix = torch.cdist(positions, positions, p=2)  # [num_cells, num_cells]

    # 计算模态相似度矩阵
    with torch.no_grad():
        similarity_matrix = torch.mm(modality_normalized, modality_normalized.t())  # [num_cells, num_cells]

    edges = []
    edge_weights = []

    # 使用 tqdm 进行进度条显示
    for i in tqdm(range(num_cells), desc="Constructing Graph"):
        # 获取当前细胞到所有其他细胞的距离
        distances = distance_matrix[i]  # [num_cells]

        # 获取距离排序的索引
        sorted_distances, sorted_indices = torch.sort(distances)

        # 排除自身（距离为0）
        sorted_distances = sorted_distances[1:]
        sorted_indices = sorted_indices[1:]

        # 找出所有唯一的距离值，并保持排序
        unique_distances, inverse_indices = torch.unique(sorted_distances, sorted=True, return_inverse=True)

        neighbors = []
        remaining_k = k

        for dist in unique_distances:
            # 获取当前距离的所有细胞索引
            same_dist_mask = (sorted_distances == dist)
            same_dist_indices = sorted_indices[same_dist_mask]  # [num_same_dist]

            if same_dist_indices.numel() <= remaining_k:
                # 如果当前距离的细胞数量不超过剩余需要选择的数量，全部添加
                neighbors.extend(same_dist_indices.tolist())
                remaining_k -= same_dist_indices.numel()
            else:
                # 如果超过剩余需要选择的数量，根据模态相似度选择
                # 获取这些细胞的相似度
                similarities = similarity_matrix[i, same_dist_indices]  # [num_same_dist]

                # 获取相似度排序的索引（降序）
                sorted_similarities, sim_sorted_indices = torch.sort(similarities, descending=True)

                # 选择前remaining_k个细胞
                top_k_indices = same_dist_indices[sim_sorted_indices[:remaining_k]]

                neighbors.extend(top_k_indices.tolist())
                remaining_k = 0
                break  # 已经选择了k个邻居，退出循环
        # # 打印 neighbors 和它的类型
        # print(f"Neighbors: {neighbors}")
        # print(f"Type of neighbors: {type(neighbors)}")
        # 计算边权重（使用距离的倒数）
        selected_distances = distance_matrix[i, torch.tensor(neighbors, device=device)]  # [k]
        weights = 1.0 / (selected_distances + 1e-12)  # 避免除以零

        # 添加到边列表
        for neighbor, weight in zip(neighbors, weights):
            edges.append((i, neighbor))
            edge_weights.append(weight.item())

    # 全局归一化边权重
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-12)
    # 转换为 PyTorch Geometric 所需的格式
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)  # [2, num_edges]
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).to(device)  # [num_edges]

    return edge_index, edge_weight


import torch
import torch.nn.functional as F


def build_feature_graph(features, k=6):
    """
    基于细胞特征的构图函数，通过计算细胞间的余弦相似度选择最近的k个邻居。

    参数：
        features (torch.Tensor): 细胞的特征张量，形状为 [num_cells, feature_dim]。
        k (int): 每个细胞的邻居数量。

    返回：
        edge_index (torch.LongTensor): 边的索引，形状为 [2, num_edges]。
        edge_weight (torch.FloatTensor): 边的权重，形状为 [num_edges]。
    """
    device = features.device  # 获取当前设备
    num_cells = features.shape[0]

    # 归一化特征向量，计算余弦相似度
    features_normalized = F.normalize(features, p=2, dim=1)  # 对每个细胞的特征进行L2归一化

    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(features_normalized, features_normalized.T)  # [num_cells, num_cells]

    edges = []
    edge_weights = []

    for i in range(num_cells):
        # 获取当前细胞到所有其他细胞的相似度
        similarities = similarity_matrix[i]  # [num_cells]

        # 获取相似度排序的索引（降序）
        sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)

        # 排除自身（相似度为1）
        sorted_similarities = sorted_similarities[1:]
        sorted_indices = sorted_indices[1:]

        # 获取前k个邻居
        neighbors = sorted_indices[:k].tolist()

        # 计算边权重（使用相似度的倒数）
        selected_similarities = similarity_matrix[i, torch.tensor(neighbors, device=device)]  # [k]
        weights = selected_similarities

        # 添加到边列表
        for neighbor, weight in zip(neighbors, weights):
            edges.append((i, neighbor))
            edge_weights.append(weight.item())

    # 全局归一化边权重
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-12)

    # 转换为 PyTorch Geometric 所需的格式
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)  # [2, num_edges]
    edge_weight = torch.tensor(edge_weights, dtype=torch.float).to(device)  # [num_edges]

    return edge_index, edge_weight


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.transformer_layer(x)
        return x
class scNet(nn.Module):
    def __init__(self, hidden_dim, print_freq=100):
        super(scNet, self).__init__()
        self.gcn_encoder_atac = GCNEncoder(30, hidden_dim, hidden_dim)
        self.gcn_encoder_atac_feature = GCNEncoder(30, hidden_dim, hidden_dim)
        self.gcn_encoder_rna = GCNEncoder(50, hidden_dim, hidden_dim)
        self.gcn_encoder_rna_feature = GCNEncoder(50, hidden_dim, hidden_dim)
        self.print_freq = print_freq
        self.forward_count = 0

    def forward(self, modality1, modality2, position):

        # 构建图
        graph_atac = build_graph(position, modality2)  # 使用ATAC-seq数据构建图
        graph_rna = build_graph(position, modality1)   # 使用RNA-seq数据构建图

        graph_atac_feature = build_feature_graph(modality2)  # 使用ATAC-seq数据构建图
        graph_rna_feature = build_feature_graph(modality1)  # 使用RNA-seq数据构建图


        # 将数据转换为PyG的数据格式
        data_atac = torch_geometric.data.Data(x=modality2.to('cuda'), edge_index=graph_atac[0].to('cuda'),
                                              edge_attr=graph_atac[1].to('cuda'))
        data_rna = torch_geometric.data.Data(x=modality1.to('cuda'), edge_index=graph_rna[0].to('cuda'),
                                             edge_attr=graph_rna[1].to('cuda'))

        data_atac_feature = torch_geometric.data.Data(x=modality2.to('cuda'), edge_index=graph_atac_feature[0].to('cuda'),
                                              edge_attr=graph_atac_feature[1].to('cuda'))
        data_rna_feature = torch_geometric.data.Data(x=modality1.to('cuda'), edge_index=graph_rna_feature[0].to('cuda'),
                                             edge_attr=graph_rna_feature[1].to('cuda'))

        # 使用GCN编码
        enc_output_atac = self.gcn_encoder_atac(data_atac.x, data_atac.edge_index, data_atac.edge_attr)
        enc_output_rna = self.gcn_encoder_rna(data_rna.x, data_rna.edge_index, data_rna.edge_attr)

        enc_output_atac_feature = self.gcn_encoder_atac(data_atac_feature.x, data_atac_feature.edge_index, data_atac_feature.edge_attr)
        enc_output_rna_feature = self.gcn_encoder_rna(data_rna_feature.x, data_rna_feature.edge_index, data_rna_feature.edge_attr)


        enc_output_atac = F.normalize(enc_output_atac, p=2, dim=-1)
        enc_output_rna = F.normalize(enc_output_rna, p=2, dim=-1)

        enc_output_atac_feature = F.normalize(enc_output_atac_feature, p=2, dim=-1)
        enc_output_rna_feature = F.normalize(enc_output_rna_feature, p=2, dim=-1)


        # 计算相似度矩阵
        similarity_matrix = torch.matmul(enc_output_rna, enc_output_atac.T)

        self.forward_count += 1

        return enc_output_atac, enc_output_rna, enc_output_atac_feature, enc_output_rna_feature, similarity_matrix
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=2):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def _compute_attn_score(self, query, key):
        # 获取query和key的维度信息
        batch_size, seq_length, latent_dim = query.size()

        # 将query和key按照多头注意力机制进行reshape
        query_reshaped = query.view(batch_size, self.num_heads, seq_length, latent_dim // self.num_heads)
        key_reshaped = key.view(batch_size, self.num_heads, seq_length, latent_dim // self.num_heads)

        # 计算注意力分数: matmul(Q, K^T) / sqrt(d_k)
        attn_score = torch.matmul(query_reshaped, key_reshaped.transpose(-2, -1)) / (
                    latent_dim // self.num_heads) ** 0.5

        return attn_score
    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return self.layer_norm(attn_output + query)

class Model2(nn.Module):
    def __init__(self, input_dim_rna, input_dim_atac, latent_dim, hidden_dim, batch_size, num_heads=2):
        super(Model2, self).__init__()
        # 定义初始查询向量 N
        self.initial_query = nn.Parameter(torch.randn(latent_dim))
        self.num_heads = num_heads
        # 编码层，将输入映射到潜在空间
        self.rna_encoder = nn.Linear(input_dim_rna, latent_dim)
        self.atac_encoder = nn.Linear(input_dim_atac, latent_dim)

        # 跨注意力机制，用于融合 RNA 和 ATAC 的特征
        self.cross_attention_rna = CrossAttention(latent_dim, num_heads)
        self.cross_attention_atac = CrossAttention(latent_dim, num_heads)
        self.cross_attention_z = CrossAttention(latent_dim, num_heads)

        # 定义 Output Query Array
        self.output_query = nn.Parameter(torch.randn(latent_dim))

        # 解码层，从融合后的表示中重构输出
        self.rna_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim_rna)
        )

        self.atac_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim_atac)
        )
    def _compute_attn_score(self, query, key):
        # 获取query和key的维度信息
        batch_size, seq_length, latent_dim = query.size()

        # 将query和key按照多头注意力机制进行reshape
        query_reshaped = query.view(batch_size, self.num_heads, seq_length, latent_dim // self.num_heads)
        key_reshaped = key.view(batch_size, self.num_heads, seq_length, latent_dim // self.num_heads)

        # 计算注意力分数: matmul(Q, K^T) / sqrt(d_k)
        attn_score = torch.matmul(query_reshaped, key_reshaped.transpose(-2, -1)) / (
                    latent_dim // self.num_heads) ** 0.5
        return attn_score

    def forward(self, input_rna, input_atac, position, m1, m2):
        # 初始化查询向量 N
        batch_size = input_rna.size(0)

        N = self.initial_query.unsqueeze(0).repeat(batch_size, 1)

        # 将输入数据映射到潜在空间
        x_rna = self.rna_encoder(input_rna)
        x_atac = self.atac_encoder(input_atac)

        N = N.unsqueeze(0)  # (1, batch_size, latent_dim)
        x_rna = x_rna.unsqueeze(0)  # (1, batch_size, latent_dim)
        x_atac = x_atac.unsqueeze(0)  # (1, batch_size, latent_dim)

        
        N = self.cross_attention_rna(N, x_rna, x_rna)
        N = self.cross_attention_atac(N, x_atac, x_atac)




        # 使用 Output Query Array 生成最终的融合表示 Z
        output_query = self.output_query.unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)

        Z = self.cross_attention_z(output_query, N, N)

        # 计算 q * k.T 注意力得分
        attn_score_matrix = self._compute_attn_score(output_query, N)
       
        attn_score_atac = 0
        attn_score_rna = 0


        Z = Z.squeeze(0)
        # 使用 Z 解码 RNA 和 ATAC 输出
        output_rna = self.rna_decoder(Z)
        output_atac = self.atac_decoder(Z)

        return Z, output_rna, output_atac, attn_score_rna, attn_score_atac, attn_score_matrix
