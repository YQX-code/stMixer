import os
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from anndata import AnnData
import random
from collections import defaultdict
import scipy.sparse as sp

# ========== 随机数种子 ==========
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== 参数配置 ==========
embed1_path  = '/home/yangqx/mount2T_dev_sdh/cluster_plots_spleen1_single_pro_broken/data_epoch_10/Z_data_epoch_10_epoch_1.h5ad'
embed2_path  = '/home/yangqx/mount2T_dev_sdh/cluster_plots_spleen2/data_epoch_27/Z_data_epoch_27_epoch_48.h5ad'
pro1_path    = '/home/yangqx/SC/data/spleen/spleen1_protein_with_spatial.h5ad'
pro2_path    = '/home/yangqx/SC/data/spleen/spleen2_protein_with_spatial.h5ad'
# labels1_path = '/home/yangqx/SC/test/test_print_fig/stMixer_spleen1_spleen1_single_pro_broken_10,1.npy'
labels1_path = '/home/yangqx/SC/test/test_print_fig/stMixer_spleen1_spleen1_single_pro_broken_10,1_km6.npy'
labels2_path = '/home/yangqx/SC/test/test_print_fig/stMixer_spleen2_27,48.npy'
out_dir      = '/home/yangqx/mount2T_dev_sdh/simple_spleen_mapping_spleen1_single_mmd'
os.makedirs(out_dir, exist_ok=True)

# 超参数
epochs       = 1500      # 总训练轮数
switch_epoch = 500      # 动态采样开始轮次
g_mnn        = 1.0      # MNN loss 权重
g_cluster    = 1     # 同簇 loss 权重
margin       = 1.0      # Triplet margin
lr           = 1e-3     # 学习率
n_clusters   = 5        # KMeans 聚类数
k_mnn        = 10       # MNN 邻居数
k_cluster    = 6        # 同簇最近邻数
blend_epochs = 100
# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch

def print_gpu_mem(info=''):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        max_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"{info} 当前分配: {allocated:.1f} MB, 峰值分配: {max_alloc:.1f} MB")



# ========== 数据读取 ==========
adata1 = sc.read_h5ad(embed1_path)
adata2 = sc.read_h5ad(embed2_path)
pro1   = sc.read_h5ad(pro1_path)
pro2   = sc.read_h5ad(pro2_path)

# 预先加载切片标签，用于 cluster 内构建
labels1 = np.load(labels1_path)
labels2 = np.load(labels2_path)

mask = labels1 != 5
# 3. 子集过滤
adata1 = adata1[mask].copy()
pro1 = pro1[mask].copy()
labels1 = labels1[mask]

# 原始高维嵌入
X1 = adata1.X.astype(np.float32)
X2 = adata2.X.astype(np.float32)
# 蛋白表达
if sp.issparse(pro1.X):
    prof1 = pro1.X.toarray().astype(np.float32)
    prof2 = pro2.X.toarray().astype(np.float32)
else:
    prof1 = pro1.X.astype(np.float32)
    prof2 = pro2.X.astype(np.float32)
# 空间坐标
sp1 = adata1.obsm['spatial']
sp2 = adata2.obsm['spatial']
# 预先加载切片标签，用于 cluster 内构建

# ========== 初始 MNN 构建 ==========
nbrs12 = NearestNeighbors(n_neighbors=k_mnn).fit(prof2)
_, idx12 = nbrs12.kneighbors(prof1)
nbrs21 = NearestNeighbors(n_neighbors=k_mnn).fit(prof1)
_, idx21 = nbrs21.kneighbors(prof2)
# 互为 MNN 对列表
mnn_pairs = [(i,j) for i in range(len(prof1)) for j in idx12[i] if i in idx21[j]]
# 构建反向索引: slice1 idx -> list of slice2 mnn
mnn_dict = defaultdict(list)
for i,j in mnn_pairs:
    mnn_dict[i].append(j)

# ========== 初始同簇最近邻构建 ==========
cluster_nns = {0: {}, 1: {}}
for slc, (X, labels) in enumerate([(X1, labels1), (X2, labels2)]):
    for cl in np.unique(labels):
        pts = np.where(labels == cl)[0]
        if len(pts) <= k_cluster: continue
        nbrs = NearestNeighbors(n_neighbors=k_cluster+1).fit(X[pts])
        _, ids = nbrs.kneighbors(X[pts])
        for p, neigh in zip(pts, ids):
            cluster_nns[slc][p] = pts[neigh[1:]]

# ========== 初始三元组构建 ==========
# ========== 初始三元组构建（含同切片最近非同簇3个点负样本） ==========
anc_init     = []
pos_mnn_init = []
pos_cls_init = []
neg_init     = []
neg_same_slice_ncls = []

for i, js in mnn_dict.items():
    for j in js:
        anc_init.append(X1[i])
        pos_mnn_init.append(X2[j])
        # 同簇正样本：slice1 内的同簇邻居
        if i in cluster_nns[0]:
            pos_cls_init.append(X1[cluster_nns[0][i][0]])
        else:
            pos_cls_init.append(X1[i])
        # 负样本1：随机选取不同簇的 slice2
        neg_c = np.where(labels2 != labels1[i])[0]
        neg_init.append(X2[np.random.choice(neg_c)])
        # 负样本2：同切片内最近的非同簇3个点
        ncls_mask = labels1 != labels1[i]
        ncls_indices = np.where(ncls_mask)[0]
        if len(ncls_indices) == 0:
            neg_same_slice_ncls.append(np.repeat(X1[i][None], 3, axis=0))
        else:
            dists = np.linalg.norm(X1[ncls_indices] - X1[i], axis=1)
            nn3 = ncls_indices[np.argsort(dists)[:3]]
            neg_same_slice_ncls.append(X1[nn3])

anc_init     = torch.tensor(np.stack(anc_init),device=device)
pos_mnn_init = torch.tensor(np.stack(pos_mnn_init),device=device)
pos_cls_init = torch.tensor(np.stack(pos_cls_init),device=device)
neg_init     = torch.tensor(np.stack(neg_init),device=device)
neg_same_slice_ncls = torch.tensor(np.stack(neg_same_slice_ncls),device=device)
# 新增三元组构建 (slice2内的正负样本)
pos_cls_slice2_init = []
neg_cls_slice2_init = []
anc_slice2_init = []

for idx in range(len(X2)):
    anc_slice2_init.append(X2[idx])

    # 同簇最近的6个点
    same_cluster = np.where(labels2 == labels2[idx])[0]
    same_cluster = same_cluster[same_cluster != idx]  # 排除自己
    if len(same_cluster) == 0:
        pos_cls_slice2_init.append(np.repeat(X2[idx][None], 6, axis=0))
    else:
        dists_pos = np.linalg.norm(X2[same_cluster] - X2[idx], axis=1)
        nn6_pos = same_cluster[np.argsort(dists_pos)[:6]]
        if len(nn6_pos) < 6:  # 防止不足6个点的情况
            nn6_pos = np.pad(nn6_pos, (0, 6 - len(nn6_pos)), 'wrap')
        pos_cls_slice2_init.append(X2[nn6_pos])

    # 不同簇最近的6个点
    diff_cluster = np.where(labels2 != labels2[idx])[0]
    dists_neg = np.linalg.norm(X2[diff_cluster] - X2[idx], axis=1)
    nn6_neg = diff_cluster[np.argsort(dists_neg)[:6]]
    neg_cls_slice2_init.append(X2[nn6_neg])

anc_slice2_init = torch.tensor(np.stack(anc_slice2_init), device=device)
pos_cls_slice2_init = torch.tensor(np.stack(pos_cls_slice2_init), device=device)
neg_cls_slice2_init = torch.tensor(np.stack(neg_cls_slice2_init), device=device)
# batch_labels: 0表示slice1, 1表示slice2
batch_labels = np.concatenate([
    np.zeros(len(X1), dtype=int),
    np.ones(len(X2), dtype=int)
])
# ========== 模型定义 ==========
class SimpleEmbedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1  = nn.Linear(dim, 50)
        self.relu1 = nn.ReLU()
        self.fc2  = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 50)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(50, 50)
    def forward(self, x):
        return self.fc4(self.relu3(self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))))
        # return self.fc2(self.relu1(self.fc1(x)))
model   = SimpleEmbedNet(X1.shape[1]).to(device)
opt     = optim.Adam(model.parameters(), lr=lr)
triplet = nn.TripletMarginLoss(margin=margin)

# 动态三元组占位符
dyn_anchor = dyn_pos = dyn_neg = None
lbl1 = lbl2 = None
def compute_mmd_torch(x, y, gamma=1.0):
    """
    x: [N, D] torch.Tensor
    y: [M, D] torch.Tensor
    返回: 标量 torch.Tensor
    """
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    rx = (x ** 2).sum(1).view(-1, 1)
    ry = (y ** 2).sum(1).view(-1, 1)
    K_xx = torch.exp(-gamma * (rx + rx.t() - 2 * xx))
    K_yy = torch.exp(-gamma * (ry + ry.t() - 2 * yy))
    K_xy = torch.exp(-gamma * (rx + ry.t() - 2 * xy))
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd
# ========== 训练循环 ==========
for epoch in range(1, epochs+1):
    model.train()
    opt.zero_grad()
    if epoch <= switch_epoch:
        # 多组三元组，一一配对
        ea = model(anc_init)
        ep_mnn = model(pos_mnn_init)
        ep_cls = model(pos_cls_init)
        en = model(neg_init)
        en_ncls = model(neg_same_slice_ncls)

        # 明确定义 D
        D = ea.shape[1]

        # loss均用TripletMarginLoss
        loss = (
                g_mnn * triplet(ea, ep_mnn, en)
                + g_cluster * triplet(ea, ep_cls, en)
        )

        ea_rep = ea.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        ep_cls_rep = ep_cls.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        ep_mnn_rep = ep_mnn.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        en_ncls_flat = en_ncls.view(-1, D)

        loss += triplet(ea_rep, ep_cls_rep, en_ncls_flat)
        loss += triplet(ea_rep, ep_mnn_rep, en_ncls_flat)

        # 明确定义维度
        input_dim = X2.shape[1]

        # 新增切片2内部的loss
        ea_slice2 = model(anc_slice2_init)
        ep_slice2 = model(pos_cls_slice2_init.view(-1, input_dim))
        en_slice2 = model(neg_cls_slice2_init.view(-1, input_dim))

        ea_slice2_rep = ea_slice2.unsqueeze(1).repeat(1, 6, 1).view(-1, D)

        loss += triplet(ea_slice2_rep, ep_slice2, en_slice2)
    elif epoch <= switch_epoch + blend_epochs:
        # 多组三元组，一一配对
        ea = model(anc_init)
        ep_mnn = model(pos_mnn_init)
        ep_cls = model(pos_cls_init)
        en = model(neg_init)
        en_ncls = model(neg_same_slice_ncls)

        # 明确定义 D
        D = ea.shape[1]

        # loss均用TripletMarginLoss
        triplet_loss = (
                g_mnn * triplet(ea, ep_mnn, en)
                + g_cluster * triplet(ea, ep_cls, en)
        )

        ea_rep = ea.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        ep_cls_rep = ep_cls.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        ep_mnn_rep = ep_mnn.unsqueeze(1).repeat(1, 3, 1).view(-1, D)
        en_ncls_flat = en_ncls.view(-1, D)

        triplet_loss += triplet(ea_rep, ep_cls_rep, en_ncls_flat)
        triplet_loss += triplet(ea_rep, ep_mnn_rep, en_ncls_flat)

        # 明确定义维度
        input_dim = X2.shape[1]

        # 新增切片2内部的loss
        ea_slice2 = model(anc_slice2_init)
        ep_slice2 = model(pos_cls_slice2_init.view(-1, input_dim))
        en_slice2 = model(neg_cls_slice2_init.view(-1, input_dim))

        ea_slice2_rep = ea_slice2.unsqueeze(1).repeat(1, 6, 1).view(-1, D)

        triplet_loss += triplet(ea_slice2_rep, ep_slice2, en_slice2)
        if epoch == switch_epoch + 1:
            # 第一次动态阶段：KMeans 聚类
            with torch.no_grad():
                emb1_np = model(torch.tensor(X1, device=device)).detach().cpu().numpy()
                emb2_np = model(torch.tensor(X2, device=device)).detach().cpu().numpy()
            all_emb = np.vstack([emb1_np, emb2_np])
            km = KMeans(n_clusters=n_clusters, random_state=0).fit(all_emb)
            lbl1 = km.labels_[:len(emb1_np)]
            lbl2 = km.labels_[len(emb1_np):]

        # 动态阶段每轮都要用 torch.Tensor embedding
        model.train()
        emb1 = model(torch.tensor(X1, device=device))  # [N1, D]，torch.Tensor
        emb2 = model(torch.tensor(X2, device=device))  # [N2, D]，torch.Tensor

        mmd_loss = torch.tensor(0.0, device=device)
        for cl in np.unique(lbl1):
            idx1 = np.where(lbl1 == cl)[0]
            idx2 = np.where(lbl2 == cl)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                x1 = emb1[idx1]
                x2 = emb2[idx2]
                mmd_loss = mmd_loss + compute_mmd_torch(x1, x2, gamma=1.0)
        blend = (epoch - switch_epoch) / blend_epochs  # 从0变化到1
        loss = (1 - blend) * triplet_loss + blend * mmd_loss
    else:
        # === 只用 MMD Loss ===
        model.train()
        emb1 = model(torch.tensor(X1, device=device))
        emb2 = model(torch.tensor(X2, device=device))
        mmd_loss = torch.tensor(0.0, device=device)
        for cl in np.unique(lbl1):
            idx1 = np.where(lbl1 == cl)[0]
            idx2 = np.where(lbl2 == cl)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                x1 = emb1[idx1]
                x2 = emb2[idx2]
                mmd_loss = mmd_loss + compute_mmd_torch(x1, x2, gamma=1.0)
        loss = mmd_loss
    loss.backward()
    opt.step()
    print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
    print_gpu_mem(f"Epoch {epoch}")  # 打印显存
    # ========== 保存 & 可视化 ==========
    model.eval()
    with torch.no_grad():
        new1 = model(torch.tensor(X1,device=device)).cpu().numpy()
        new2 = model(torch.tensor(X2,device=device)).cpu().numpy()
    vars = [f'emb_{i}' for i in range(new1.shape[1])]
    for emb, ad, sp, tag in [(new1,adata1,sp1,'slice1'),(new2,adata2,sp2,'slice2')]:
        out = AnnData(X=emb,obs=ad.obs.copy(),var=pd.DataFrame(index=vars),obsm={'spatial':sp})
        out.write_h5ad(os.path.join(out_dir,f'epoch_{epoch}_{tag}.h5ad'))
    all_emb = np.vstack([new1,new2])
    km2     = KMeans(n_clusters=n_clusters,random_state=0).fit(all_emb)
    labs    = km2.labels_
    lab1,lab2 = labs[:len(new1)],labs[len(new1):]
    np.save(os.path.join(out_dir,f'epoch_{epoch}_slice1_kmeans.npy'),lab1)
    np.save(os.path.join(out_dir,f'epoch_{epoch}_slice2_kmeans.npy'),lab2)
    cmap    = matplotlib.colors.ListedColormap(["#2BA02D","#FD7E12","#D52728","#1D78B6","#8B564C"])
    for spc,lab,tag in [(sp1,lab1,'slice1'),(sp2,lab2,'slice2')]:
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(spc[:,0],spc[:,1],c=lab,cmap=cmap,s=20)
        ax.set_title(f'{tag} - Epoch {epoch}');ax.axis('off')
        plt.savefig(os.path.join(out_dir,f'epoch_{epoch}_{tag}_spatial.png'),dpi=150)
        plt.close(fig)
    model.eval()
    if epoch >= switch_epoch and epoch%50==0:
        # 1. AnnData 拼接
        ad1 = AnnData(X=new1, obs=adata1.obs.copy(), var=pd.DataFrame(index=vars), obsm={'spatial': sp1})
        ad2 = AnnData(X=new2, obs=adata2.obs.copy(), var=pd.DataFrame(index=vars), obsm={'spatial': sp2})

        ad1.obs['type'] = lab1.astype(str)
        ad2.obs['type'] = lab2.astype(str)
        ad1.obs['batch'] = 'slice1'
        ad2.obs['batch'] = 'slice2'

        adata = ad1.concatenate(ad2, batch_key='batch', batch_categories=['slice1', 'slice2'], join='inner',
                                index_unique=None)
        adata.obsm["X_emb"] = adata.X
        # from scib.metrics import ilisi_graph
        #
        # # adata: AnnData，已经运行过 sc.pp.neighbors
        # # batch_key: 批次标签名，比如 'batch'
        # ilisi_scores = ilisi_graph(
        #                                         adata, batch_key="batch",
        #                                         type_="embed",
        #                                         use_rep="X_emb",
        #                                         k0=90,
        #                                         n_cores=1
        #                                     )
        # print("iLISI (graph) 均值：", ilisi_scores.mean())
        # print(f"epoch:{epoch}  iLISI均值：", ilisi_scores.mean())

        # 2. UMAP计算（直接用模型嵌入）
        sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=15)
        sc.tl.umap(adata)

        # 3. 保存 UMAP 坐标为 csv
        umap_res = adata.obsm['X_umap']
        umap_out = pd.DataFrame(umap_res, columns=['umap1', 'umap2'], index=adata.obs_names)
        umap_out['batch'] = adata.obs['batch'].values
        umap_out['type'] = adata.obs['type'].values
        umap_csv_path = os.path.join(out_dir, f'epoch_{epoch}_umap.csv')
        umap_out.to_csv(umap_csv_path)

        # 4. UMAP by batch（手动保存到 out_dir）
        fig, ax = plt.subplots(figsize=(6, 6))
        for b in adata.obs['batch'].unique():
            idx = adata.obs['batch'] == b
            ax.scatter(umap_res[idx, 0], umap_res[idx, 1], s=12, label=b, alpha=0.7)
        ax.set_title(f'UMAP by Batch, epoch {epoch}')
        ax.legend()
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_umap_by_batch.png'), dpi=150)
        plt.close(fig)

        # 5. UMAP by type（聚类标签，手动保存到 out_dir）
        palette = ["#2BA02D", "#FD7E12", "#D52728", "#1D78B6", "#8B564C"]
        type_labels = np.unique(adata.obs['type'])
        type_to_color = {lab: palette[i % len(palette)] for i, lab in enumerate(type_labels)}
        colors = [type_to_color[t] for t in adata.obs['type']]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(umap_res[:, 0], umap_res[:, 1], c=colors, s=12, alpha=0.7)
        ax.set_title(f'UMAP by Cluster Type, epoch {epoch}')
        for lab, color in type_to_color.items():
            ax.scatter([], [], c=color, label=f'Type {lab}')
        ax.legend(markerscale=1, fontsize='small', loc='best')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_umap_by_type.png'), dpi=150)
        plt.close(fig)

print("训练完成，结果保存于：", out_dir)

