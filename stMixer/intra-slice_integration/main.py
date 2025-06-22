import os

# 设置线程数
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import numpy as np

import pandas as pd
import torch.nn as nn
import random
from configurations import Stage1_Options
from configurations import *
from sc_dataset import SC_dataset
from sc_dataset import Encoded_dataset
from torch.utils.data import random_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset
from torch.utils.data import Dataset, ConcatDataset
from sklearn.cluster import KMeans

from esda.moran import Moran

import time
# from stage1_test import test
from scnet import *
import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
import umap

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import torch.optim as optim

from anndata import AnnData


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='threadpoolctl')

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    # 保存原来的 stderr
    stderr_fileno = sys.stderr
    try:
        # 将 stderr 重定向到空设备
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        # 恢复原来的 stderr
        sys.stderr = stderr_fileno

# 使用 suppress_stderr() 只在需要时 suppress 错误输出
with suppress_stderr():
    # 在这个代码块中的所有 stderr 输出都会被忽略




    def set_seed(seed=10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True



    if __name__ == '__main__':

        options = Stage1_Options()
        parser = options.get_parser().parse_args()
        set_seed(parser.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpuID
        # model_save_path=os.path.join(checkpoints_path,parser.expName)
        import os
        import torch
        import os

        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        # 打印 PyTorch 是否检测到可用的 CUDA
        print(f"CUDA is available: {torch.cuda.is_available()}")

        # 打印可以使用的 GPU 数量
        print(f"GPU count: {torch.cuda.device_count()}")

        # 打印每个 GPU 的名称
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(torch.cuda.is_available())  # 应该返回 True
        print(torch.cuda.device_count())  # 查看可用 GPU 数量
        print(parser.gpuID)
        print(torch.cuda.device_count())
        # ----------------------------------define Dataset,Dataloader---------------------------------------------------------------------------

        dataset = SC_dataset()
        batch_size=parser.batchSize
        print(batch_size)
        trainset = dataset
        trainloader = DataLoader(trainset, batch_size=batch_size)

        # ----------------------------------define Dataset,Dataloader---------------------------------------------------------------------------

        # ----------------------------------define model,optimizer---------------------------------------------------------------------------
        model_name = parser.model
        model_dict = {
            'sc': scNet(hidden_dim=50)

        }
        if not parser.continueTrain:
            assert parser.model in model_dict.keys(), 'provide the proper name for model'
            model = model_dict[parser.model]

        print(model.named_modules())

        if parser.continueTrain:
            assert os.path.exists(os.path.join(stage1_model_save_path, parser.continueExpName,
                                               parser.continueSavedModelNmae)), 'continueExpName and continueSavedModelName need to be provided properly for continue train'
            model_load_path = os.path.join(stage1_model_save_path, parser.continueExpName, parser.continueSavedModelName)

        optimizer = torch.optim.Adam(model.parameters(), lr=parser.learningRate)
        scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)

        def new_loss_function(input_rna, input_atac, output_rna, output_atac):
            loss_rna = F.mse_loss(output_rna, input_rna)
            loss_atac = F.mse_loss(output_atac, input_atac)
            loss = (loss_rna + loss_atac)/2
            return loss
        def feature_loss_function(input_rna, input_atac, output_rna, output_atac):
            loss_rna = F.mse_loss(output_rna, input_rna)
            loss_atac = F.mse_loss(output_atac, input_atac)
            loss = (loss_rna + loss_atac)/2
            return loss,loss_rna,loss_atac

        def loss_function(similarity_matrix, position = 0, enc_output_atac = 0, labels = 0):
            batch_size = similarity_matrix.size(0)
            labels_true = torch.arange(batch_size, device=similarity_matrix.device)

            # 计算交叉熵损失
            loss_rna = F.cross_entropy(similarity_matrix, labels_true)
            loss_atac = F.cross_entropy(similarity_matrix.T, labels_true)
            return (loss_rna + loss_atac) / 2


        import torch
        import torch.nn.functional as F

        device = torch.device("cuda" if parser.gpuID != 'cpu' else "cpu")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)


        import numpy as np
        import umap
        import scanpy as sc
        from anndata import AnnData


        def visualize_and_save_Z(Z, q, position, embedding, embedding_q, labels, labels_q, epoch, data_epoch, attn_score_matrix,
                                 save_path='../result'):
            """
            可视化聚类结果和降维结果，将 Z 和 q 的可视化合并到一张图。
            """
            # 创建以 data_epoch 命名的文件夹
            data_epoch_path = os.path.join(save_path, f'data_epoch_{data_epoch}')
            os.makedirs(data_epoch_path, exist_ok=True)

            # 保存 Z 为 h5ad 文件
            z_filename = os.path.join(data_epoch_path, f'Z_data_epoch_{data_epoch}_epoch_{epoch}.h5ad')
            adata_z = anndata.AnnData(X=Z)  # 将 Z 转为 AnnData 对象
            adata_z.obsm['spatial'] = position  # 将 position 填充为 spatial 信息
            adata_z.obs['annotation'] = labels.astype(str)  # 保存聚类标签到 obs 中
            adata_z.write_h5ad(z_filename)
            print(f"Saved Z to {z_filename}")

            # 保存 q 为 h5ad 文件
            q_filename = os.path.join(data_epoch_path, f'q_data_epoch_{data_epoch}_epoch_{epoch}.h5ad')
            adata_q = anndata.AnnData(X=q)  # 将 q 转为 AnnData 对象
            adata_q.obsm['spatial'] = position  # 将 position 填充为 spatial 信息
            adata_q.obs['annotation'] = labels_q.astype(str)  # 保存聚类标签到 obs 中
            adata_q.write_h5ad(q_filename)
            print(f"Saved q to {q_filename}")

            # 聚类对比图 (Cluster Visualization)
            plt.figure(figsize=(12, 6))

            # 子图 1：Z 的聚类结果
            plt.subplot(1, 2, 1)
            sns.scatterplot(
                x=position[:, 0],
                y=position[:, 1],
                hue=labels,
                palette='Paired',
                style=labels,
                markers='o',
                s=20
            )
            plt.title(f'Cluster Visualization of Z at Epoch {epoch}')
            plt.xlabel('Position X')
            plt.ylabel('Position Y')
            plt.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))
            plt.grid(True)

            # 子图 2：q 的聚类结果
            plt.subplot(1, 2, 2)
            sns.scatterplot(
                x=position[:, 0],
                y=position[:, 1],
                hue=labels_q,
                palette='Paired',
                style=labels_q,
                markers='o',
                s=20
            )
            plt.title(f'Cluster Visualization of q at Epoch {epoch}')
            plt.xlabel('Position X')
            plt.ylabel('Position Y')
            plt.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))
            plt.grid(True)

            cluster_comparison_filename = os.path.join(data_epoch_path, f'cluster_comparison_epoch_{epoch}.png')
            plt.tight_layout()
            plt.savefig(cluster_comparison_filename)
            plt.close()
            print(f"Saved cluster comparison plot of Z and q to {cluster_comparison_filename}")

            # UMAP 对比图
            plt.figure(figsize=(12, 6))

            # 子图 1：Z 的 UMAP 结果
            plt.subplot(1, 2, 1)
            sns.scatterplot(
                x=embedding[:, 0],
                y=embedding[:, 1],
                hue=labels,
                palette='Paired',
                markers='o',
                s=10
            )
            plt.title(f'UMAP Visualization of Z at Epoch {epoch}')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))
            plt.grid(True)

            # 子图 2：q 的 UMAP 结果
            plt.subplot(1, 2, 2)
            sns.scatterplot(
                x=embedding_q[:, 0],
                y=embedding_q[:, 1],
                hue=labels_q,
                palette='Paired',
                markers='o',
                s=10
            )
            plt.title(f'UMAP Visualization of q at Epoch {epoch}')
            plt.xlabel('UMAP1')
            plt.ylabel('UMAP2')
            plt.legend(title='Cluster', loc='best', bbox_to_anchor=(1, 1))
            plt.grid(True)

            umap_comparison_filename = os.path.join(data_epoch_path, f'umap_comparison_epoch_{epoch}.png')
            plt.tight_layout()
            plt.savefig(umap_comparison_filename)
            plt.close()
            print(f"Saved UMAP comparison plot of Z and q to {umap_comparison_filename}")

            # # 转换 attn_score_matrix 为 numpy 数组
            # attn_score_matrix = attn_score_matrix.cpu().detach().numpy().squeeze(0)
            #
            # # 遍历每个头，分别生成热力图
            # num_heads = attn_score_matrix.shape[0]
            # for head_idx in range(num_heads):
            #     head_attention_matrix = attn_score_matrix[head_idx]
            #
            #     # 生成每个头的热力图
            #     plt.figure(figsize=(8, 6))
            #     sns.heatmap(head_attention_matrix, cmap="YlGnBu", cbar=True)
            #     plt.title(f"Attention Heatmap for Head {head_idx + 1} (Data Epoch {data_epoch}, Epoch {epoch})")
            #     plt.xlabel('Cells')
            #     plt.ylabel('Queries')
            #     plt.savefig(os.path.join(data_epoch_path,
            #                              f'attention_heatmap_head_{head_idx + 1}_data_epoch_{data_epoch}_epoch_{epoch}.png'))
            #     plt.close()
            #
            #     # 找到总关注度最大的行索引
            #     max_attention_row_index = np.argmax(np.sum(head_attention_matrix, axis=1))  # 找到最大关注度的行索引
            #     max_attention_values = head_attention_matrix[max_attention_row_index, :]  # 获取该行的关注度值
            #
            #     # 将热力图覆盖到 Z 空间聚类结果上
            #     plt.figure(figsize=(8, 6))
            #     sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="viridis", s=30,
            #                     alpha=0.7)
            #     plt.title(f"Attention Heatmap for Query (Head {head_idx + 1}, Data Epoch {data_epoch}, Epoch {epoch})")
            #
            #     # 绘制热力图，亮度代表关注度
            #     plt.scatter(position[:, 0], position[:, 1], c=max_attention_values, cmap="hot", s=30, edgecolors='w',
            #                 alpha=0.3)
            #     plt.colorbar(label='Attention Value')  # 添加颜色条，表示关注度值
            #     plt.savefig(os.path.join(data_epoch_path,
            #                              f'query_attention_heatmap_head_{head_idx + 1}_with_space_data_epoch_{data_epoch}_epoch_{epoch}.png'))
            #     plt.close()


        def save_losses_to_txt(epoch, loss_feature, loss_rna, loss_atac, loss_function_output, total_loss,
                               filename='losses.txt'):
            with open(filename, 'a') as f:
                f.write(f"Epoch {epoch}: Loss_feature = {loss_feature:.4f}, Loss_rna = {loss_rna:.4f}, "
                        f"Loss_atac = {loss_atac:.4f}, Loss_function_output = {loss_function_output:.4f}, "
                        f"Total_Loss = {total_loss:.4f}\n")
        if parser.continueTrain:
            start_epoch = parser.continueTrainStartEpoch
        else:
            start_epoch = 1
        state_dict_previous = model.state_dict()
        m1 = None
        m2 = None
        for iter, (modality1, modality2, position) in enumerate(trainloader): #1:RNA,2:ATAC
            m1 = modality1
            m2 = modality2
            break
        # 初始化用于保存所有epoch结果的列表
        all_output_atac = []
        all_output_rna = []
        for epoch in range(start_epoch, 301):
            epoch_start_time = time.time()

            for iter, (modality1, modality2, position) in enumerate(trainloader): #1:RNA,2:ATAC
                epoch_outputs = []
                epoch_outputs1 = []
                model.train()  # make Dropout available
                modality1 = modality1.to(device).float()  # 确保数据为 float 类型
                modality2 = modality2.to(device).float()  # 确保数据为 float 类型
                position = position.to(device).float()
                # print(modality2.shape)
                # exit()
                output_atac, output_rna, output_atac_feature, output_rna_feature,output= model(modality1, modality2, position)
                # output_np = output_rna.cpu().detach().numpy()
                #
                # 将每个epoch的output_atac和output_rna保存到列表中
                all_output_atac.append(output_atac.cpu().detach().numpy())
                all_output_rna.append(output_rna.cpu().detach().numpy())
                loss_feature,loss_rna,loss_atac = feature_loss_function(output_rna, output_atac, output_rna_feature, output_atac_feature)
                optimizer.zero_grad()
                loss_feature = loss_feature * 4
                loss = loss_function(output)/10 + loss_feature
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                position_np = position.cpu().detach().numpy()  # 确保position是从GPU转移并准备好的numpy array
                # 保存每个 epoch 的损失到 .txt 文件
                save_losses_to_txt(epoch, loss_feature.item(), loss_rna.item(), loss_atac.item(),
                                   loss_function(output).item()/10, loss.item())

                final_position = position_np
            if epoch == 100:
                break

        # 保存所有 epoch 的结果

        np.save('all_output_atac.npy', all_output_atac)
        np.save('all_output_rna.npy', all_output_rna)
        np.save('final_position.npy', final_position)
        print("Final output_atac  output_rna and final_positionsaved successfully.")




        ####################################################################model2#################################################################################
        all_output_atac = np.load('all_output_atac.npy', allow_pickle=True)
        all_output_rna = np.load('all_output_rna.npy', allow_pickle=True)

        position = np.load('final_position.npy')
        for data_epoch  in range(0, len(all_output_atac)):
            # print(f'第 {data_epoch} 组数据开始训练')
            input_atac = all_output_atac[data_epoch ]
            input_rna = all_output_rna[data_epoch ]

            # input_atac = np.load('final_output_atac.npy')
            # input_rna = np.load('final_output_rna.npy')
            # position = np.load('final_position.npy')

            # 转换为 PyTorch 张量
            input_atac = torch.tensor(input_atac).float()
            input_rna = torch.tensor(input_rna).float()
            position = torch.tensor(position).float()

            encodeddataset = Encoded_dataset(input_atac, input_rna, position)
            trainloader = DataLoader(encodeddataset, batch_size=batch_size)



            # 初始化新模型
            model = Model2(50,50,50,50,batch_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=parser.learningRate)
            scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)
            start_epoch = 1
            for epoch in range(start_epoch, 30):
                epoch_start_time = time.time()
                for iter, (input_rna, input_atac, position) in enumerate(trainloader):
                    model.train()  # 进入训练模式
                    m1 = m1.float().to(device)
                    m2 = m2.float().to(device)
                    input_rna = input_rna.to(device)
                    input_atac = input_atac.to(device)
                    position = position.to(device)
                    # 新模型的输入可以是之前的编码结果，也可以是其他组合
                    Z, output_rna, output_atac, attn_score_rna, attn_score_atac, attn_score_matrix= model(input_rna, input_atac,  position, m1, m2)

                    # 获取 initial_query（N）的值
                    output_query_np = model.initial_query.repeat(batch_size, 1).cpu().detach().numpy()  # 取出并转为numpy数组

                    loss_recon = new_loss_function(input_rna, input_atac,output_rna, output_atac)

                    loss = loss_recon
                    optimizer.zero_grad()
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数

                    print(f"DataEpoch {data_epoch}, Epoch {epoch}, Iteration {iter}, Loss: {loss.item()}, Loss_recon: {loss_recon.item()}")
                    # 初始参数
                    resolution = 0.27
                    target_clusters = 6
                    step_size = 0.01  # 初始分辨率调整步长
                    min_step_size = 0.01  # 最小步长
                    tolerance = 2  # 最大允许的聚类差异
                    max_iterations = 20  # 最大迭代次数
                    flag_cluster = 0  # 缩小标记


                    output_np = Z.cpu().detach().numpy()

                    # 转换为AnnData对象
                    adata = sc.AnnData(X=output_np)

                    # 计算最近邻图
                    sc.pp.neighbors(adata, n_neighbors=40, metric='correlation')


                    # UMAP降维
                    reducer = umap.UMAP(n_neighbors=40, min_dist=0.3, metric='correlation')
                    embedding = reducer.fit_transform(output_np)
                    # UMAP降维（对 q 进行降维）
                    reducer_q = umap.UMAP(n_neighbors=40, min_dist=0.3, metric='correlation')
                    embedding_q = reducer_q.fit_transform(output_query_np)

                    # Leiden聚类，动态调整分辨率
                    for i in range(max_iterations):
                        sc.tl.leiden(adata, resolution=resolution)
                        num_clusters = adata.obs['leiden'].nunique()
                        print(f"Iteration {i + 1}: Resolution = {resolution:.2f}, Clusters = {num_clusters}")

                        # 如果聚类数已经等于目标聚类数，则停止调整
                        if num_clusters == target_clusters:
                            print(f"聚类数已经达到目标聚类数! 当前聚类数：{num_clusters}")
                            break
                        # 判断聚类数与目标聚类数的差异并调整分辨率
                        if num_clusters < target_clusters:
                            resolution += step_size  # 增大分辨率
                        elif num_clusters > target_clusters:
                            resolution -= step_size  # 减小分辨率

                    # 最终的聚类标签
                    labels = adata.obs['leiden'].astype(int).to_numpy()
                     # KMeans 聚类（对 q 进行聚类）
                    kmeans_q = KMeans(n_clusters=target_clusters, random_state=0)
                    labels_q = kmeans_q.fit_predict(output_query_np)

                    position_np = position.cpu().detach().numpy()  # 确保position是从GPU转移并准备好的numpy array
                    visualize_and_save_Z(output_np, output_query_np, position_np, embedding, embedding_q, labels, labels_q, epoch, data_epoch, attn_score_matrix)

