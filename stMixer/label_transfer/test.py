#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
import harmonypy as hm
import numpy as np
import pandas as pd
from anndata import AnnData
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from collections import Counter

# -------- 参数区 --------
# 多组学低维嵌入
emb0_file     = ''    # slice1 嵌入保存h5ad路径
emb1_file     = ''    # slice2 嵌入保存h5ad路径

# H&E 图像编码
he0_file      = ''      # slice1 H&E嵌入保存h5ad路径
he1_file      = ''      # slice2 H&E嵌入保存h5ad路径
# 真实标签
anno0_file    = '/home/yangqx/SC/data/BAS/BAS1/bas1_annotated.h5ad'
anno1_file    = '/home/yangqx/SC/data/BAS/BAS2/bas2_annotated.h5ad'

mn           = '' # method name 保存名称标记
resolution   =      #leiden分辨率
k_neighbors  = 5
n_pcs_he     = 20   # Harmony 用 PCA 到多少维
R = 1000   #连通半径设置
# -------- 1. 读取 AnnData --------
adata0    = sc.read_h5ad(emb0_file)
adata1    = sc.read_h5ad(emb1_file)
adata_he0 = sc.read_h5ad(he0_file)
adata_he1 = sc.read_h5ad(he1_file)
anno0     = sc.read_h5ad(anno0_file)
anno1     = sc.read_h5ad(anno1_file)

# -------- 2. 合并真实标签到 adata0/adata1 --------
for ad, an in [(adata0, anno0), (adata1, anno1)]:
    an.obs['cluster_code'] = an.obs['cluster_code'].astype('category')
    ad.obs['true_cluster'] = pd.Categorical(
        an.obs['cluster_code'].values,
        categories=an.obs['cluster_code'].cat.categories
    )
    an.obs['path_code'] = an.obs['path_code'].astype('category')
    ad.obs['true_path'] = pd.Categorical(
        an.obs['path_code'].values,
        categories=an.obs['path_code'].cat.categories
    )

# -------- 3. 多组学聚类（Leiden） --------
sc.pp.neighbors(adata0, n_neighbors=15, metric='cosine')
sc.tl.leiden(adata0, resolution=resolution)
print(f"Leiden (res={resolution}) 在 slice1 上共聚出了 {adata0.obs['leiden'].nunique()} 个簇")

# -------- 4. H&E 模态 Harmony 校正 --------
def get_array(X):
    return X.toarray() if sp.issparse(X) else X

# 4.1 拼接原始编码
Xhe0 = get_array(adata_he0.X)
Xhe1 = get_array(adata_he1.X)
cells_he = list(adata_he0.obs_names) + list(adata_he1.obs_names)
batches  = ['slice1'] * Xhe0.shape[0] + ['slice2'] * Xhe1.shape[0]
Xhe_all = np.vstack([Xhe0, Xhe1])

# 4.2 构造 AnnData 做 PCA + Harmony
adata_he = AnnData(X=Xhe_all, obs=pd.DataFrame({'batch': batches}, index=cells_he))
sc.pp.scale(adata_he, max_value=10)
sc.tl.pca(adata_he, n_comps=n_pcs_he, svd_solver='arpack')

pca_mat = adata_he.obsm['X_pca']
# Harmony 要求 (n_dims, n_cells)
pca_in = pca_mat.T if pca_mat.shape[0] < pca_mat.shape[1] else pca_mat
ho = hm.run_harmony(pca_in, adata_he.obs, 'batch')

Z_corr = ho.Z_corr
# 转回 (n_cells, n_dims)
if Z_corr.shape[0] == n_pcs_he:
    Z_corr = Z_corr.T

# 4.3 拆分并统一归一化
Zhe0 = Z_corr[:Xhe0.shape[0], :]
Zhe1 = Z_corr[Xhe0.shape[0]:, :]
Xhe0n = normalize(Zhe0, axis=1)
Xhe1n = normalize(Zhe1, axis=1)

# -------- 5. 计算 Top‑k 余弦相似度 --------
sim = Xhe0n @ Xhe1n.T
topk_idx = np.argpartition(-sim, kth=k_neighbors-1, axis=1)[:, :k_neighbors]
np.save('topk_idx_withhm.npy', topk_idx)
# -------- 6. 簇级投票 --------
adata0.obs['pred_path']    = ''
adata0.obs['pred_cluster'] = ''

for clu in adata0.obs['leiden'].cat.categories:
    idx0 = np.where(adata0.obs['leiden']==clu)[0]
    nbrs = topk_idx[idx0].ravel()
    names2 = adata1.obs_names[nbrs]
    vote_path    = Counter(adata1.obs.loc[names2, 'true_path']).most_common(1)[0][0]
    vote_cluster = Counter(adata1.obs.loc[names2, 'true_cluster']).most_common(1)[0][0]
    mask = adata0.obs['leiden']==clu
    adata0.obs.loc[mask, 'pred_path']    = vote_path
    adata0.obs.loc[mask, 'pred_cluster'] = vote_cluster

# —— 5.5 按连通分量二次投票 ——
coords = adata0.obsm['spatial']  # (N, 2) 全体坐标
# 拷贝一份，下面直接覆盖
final_path    = adata0.obs['pred_path'].copy()
final_cluster = adata0.obs['pred_cluster'].copy()

# 遍历每个 Leiden 簇
for clu in adata0.obs['leiden'].cat.categories:
    # global indices of this簇中所有点
    mask_clu = (adata0.obs['leiden'] == clu).values
    sub_idx  = np.where(mask_clu)[0]
    if len(sub_idx) == 0:
        continue
    sub_coords = coords[sub_idx]  # 这个簇的所有坐标

    # 构建半径 R 的邻接图（包括自己）R
    A = radius_neighbors_graph(
        sub_coords, radius=R, mode='connectivity', include_self=True
    )
    n_comp, labels_comp = connected_components(A, directed=False)

    # 如果只有一个连通分量，则跳过
    if n_comp <= 1:
        continue

    # 对每个分量单独重新投票
    for comp_id in range(n_comp):
        # 这一分量在 sub_idx 中对应的全局行号
        this_cells = sub_idx[labels_comp == comp_id]
        # 把它们在 slice2 中的 top-k 邻居拉出来
        nbrs = topk_idx[this_cells].ravel()
        names2 = adata1.obs_names[nbrs]

        # 多数投票
        vote_path    = Counter(adata1.obs.loc[names2, 'true_path']).most_common(1)[0][0]
        vote_cluster = Counter(adata1.obs.loc[names2, 'true_cluster']).most_common(1)[0][0]

        # 写回覆盖
        cell_names = adata0.obs_names[this_cells]
        final_path.loc[cell_names]    = vote_path
        final_cluster.loc[cell_names] = vote_cluster

# 覆盖回 adata0.obs
adata0.obs['pred_path']    = final_path
adata0.obs['pred_cluster'] = final_cluster
# -------- 7. 统计命中率 --------
vote_color = []
for idx in adata0.obs_names:
    tp = adata0.obs.at[idx, 'true_path']
    tc = adata0.obs.at[idx, 'true_cluster']
    pp = adata0.obs.at[idx, 'pred_path']
    pc = adata0.obs.at[idx, 'pred_cluster']
    if pc==tc:
        vote_color.append('both_correct')
    elif pc==tc and pp!=tp:
        vote_color.append('cluster_only')
    elif pp==tp:
        vote_color.append('path_only')
    else:
        vote_color.append('both_wrong')

cnt = Counter(vote_color)
for k in ['both_correct','cluster_only','path_only','both_wrong']:
    print(f"{mn}:{k}: {cnt[k]}/{len(vote_color)} = {cnt[k]/len(vote_color):.2%}")

adata0.obs['vote_result'] = vote_color

# -------- 8. 空间可视化 --------
coords = adata0.obsm['spatial']
pal = {
    'both_correct':'green',
    'cluster_only':'yellow',
    'path_only':'purple',
    'both_wrong':'red'
}
plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c=[pal[c] for c in vote_color], s=5)
plt.gca().invert_yaxis()
plt.axis('off')
plt.title(f'{mn} Voting 结果')
plt.savefig(f'slice0_vote_{mn}.png', dpi=300, bbox_inches='tight')
plt.close()


# -------- 7. 基于固定 20 色调色板的可视化 & 簇映射统计 --------
import matplotlib.pyplot as plt

# 方法名，用于文件命名
method_name = mn  # mn 在脚本开头已经定义

# 1) 准备调色板
# 原始标签类别
unique_clusters = sorted(adata0.obs['true_cluster'].cat.categories.tolist())
n_clusters = len(unique_clusters)
cmap = plt.get_cmap('tab20')
palette = {lab: cmap(i) for i, lab in enumerate(unique_clusters)}

# 坐标
coords = adata0.obsm['spatial']

# Groundtruth 标签
labels0 = adata0.obs['true_cluster'].values
colors0 = [palette[l] for l in labels0]

plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c=colors0, s=5)
plt.gca().invert_yaxis(); plt.axis('off')
plt.title('Slice0 Groundtruth')
plt.savefig(f'slice0_groundtruth_{method_name}.png', dpi=300, bbox_inches='tight')
plt.close()

# Voted Cluster 标签
voted_labels = adata0.obs['pred_cluster'].values
colors_vote = [palette[l] for l in voted_labels]

plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1], c=colors_vote, s=5)
plt.gca().invert_yaxis(); plt.axis('off')
plt.title('Slice0 Voted Cluster Labels')
plt.savefig(f'slice0_voted_clusters_{method_name}.png', dpi=300, bbox_inches='tight')
plt.close()

# 2) 统计原始标签到投票标签的映射并归一化
import numpy as np
mapping_counts = np.zeros((n_clusters, n_clusters), dtype=int)

for orig_label, vote_label in zip(labels0, voted_labels):
    i = unique_clusters.index(orig_label)
    j = unique_clusters.index(vote_label)
    mapping_counts[i, j] += 1

# 行归一化
mapping_counts_normalized = mapping_counts / mapping_counts.sum(axis=1, keepdims=True)

# 打印归一化后的映射频率
# print("Original Cluster -> Voted Cluster Mapping Counts (Normalized):")
# for i, orig in enumerate(unique_clusters):
#     for j, voted in enumerate(unique_clusters):
#         print(f"From {orig} to {voted}: {mapping_counts_normalized[i, j]:.2f}")

# 3) 绘制热图
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(mapping_counts_normalized, origin='lower', aspect='auto', cmap='Reds')
fig.colorbar(im, ax=ax)
ax.set_xticks(np.arange(n_clusters))
ax.set_xticklabels(unique_clusters, rotation=90)
ax.set_yticks(np.arange(n_clusters))
ax.set_yticklabels(unique_clusters)
ax.set_xlabel('Voted Cluster')
ax.set_ylabel('Original Cluster')
ax.set_title('Normalized Mapping: Original vs Voted Cluster')
plt.tight_layout()
plt.savefig(f'slice0_cluster_mapping_normalized_{method_name}.png', dpi=300)
plt.close()

# 4) 导出为 CSV
import pandas as pd
mapping_df = pd.DataFrame(
    mapping_counts_normalized,
    index=unique_clusters,
    columns=unique_clusters
)
mapping_df.to_csv(f'slice0_cluster_mapping_{method_name}.csv')
