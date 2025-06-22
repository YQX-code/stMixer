import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
import pickle
import os
from configurations import *
from configurations import file_fold
import numpy as np
import pandas as pd
import warnings
import random
from tqdm import tqdm
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
import psutil
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

import os
import psutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


class SC_dataset(Dataset):
    def __init__(self):
        self.adata_omics1, self.adata_omics2, self.position = self.process()

    def pca(adata, use_reps=None, n_comps=10):
        """Dimension reduction with PCA algorithm"""

        from sklearn.decomposition import PCA
        from scipy.sparse.csc import csc_matrix
        from scipy.sparse.csr import csr_matrix
        pca = PCA(n_components=n_comps)
        print('pca初始化 finish')

        if use_reps is not None:
            feat_pca = pca.fit_transform(adata.obsm[use_reps])
        else:
            if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
                print('choose 2')
                feat_pca = pca.fit_transform(adata.X.toarray())
            else:
                print('choose 3')
                feat_pca = pca.fit_transform(adata.X)
        print('pca finish')
        return feat_pca

    def lsi(
            adata: anndata.AnnData, n_components: int = 20,
            use_highly_variable: Optional[bool] = None, **kwargs
    ) -> None:
        r"""
        LSI analysis (following the Seurat v3 approach)
        """
        if use_highly_variable is None:
            use_highly_variable = "highly_variable" in adata.var
        adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
        # X = tfidf(adata_use.X)
        X = adata_use.X
        X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        # adata.obsm["X_lsi"] = X_lsi
        adata.obsm["X_lsi"] = X_lsi[:, 1:]

    def clr_normalize_each_cell(adata, inplace=True):

        """Normalize count vector for each cell, i.e. for each row of .X"""

        import numpy as np
        import scipy

        def seurat_clr(x):
            # TODO: support sparseness
            s = np.sum(np.log1p(x[x > 0]))
            exp = np.exp(s / len(x))
            return np.log1p(x / exp)

        if not inplace:
            adata = adata.copy()

        # apply to dense or sparse matrix, along axis. returns dense matrix
        adata.X = np.apply_along_axis(
            seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
        )
        return adata

    def process(self):

        adata_omics1 = sc.read_h5ad('/home/yangqx/SC/data/real_annotation/RNA_with_annotation.h5ad') #模态1保存的h5ad文件
        adata_omics2 = sc.read_h5ad('/home/yangqx/SC/data/real_annotation/ADT_with_annotation.h5ad') #模态2保存的h5ad文件



        # print('read finish')
        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()
        # print('make_unique finish')



        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = SC_dataset.pca(adata_omics1_high, n_comps=50)

        # Protein
        adata_omics2 = SC_dataset.clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = SC_dataset.pca(adata_omics2, n_comps=30)
        # print(adata_omics1)
        # print(adata_omics2)
        # exit()
        return adata_omics1.obsm['feat'], adata_omics2.obsm['feat'], adata_omics1.obsm['spatial']
    def __getitem__(self, index):
        return self.adata_omics1[index], self.adata_omics2[index], self.position[index]

    def __len__(self):
        return len(self.adata_omics1)


class Encoded_dataset(Dataset):
    def __init__(self, atac_data, rna_data, position):
        self.atac_data = atac_data
        self.rna_data = rna_data
        self.position = position

    def __len__(self):
        return len(self.atac_data)

    def __getitem__(self, idx):
        atac = self.atac_data[idx]
        rna = self.rna_data[idx]
        position = self.position[idx]
        return rna, atac, position
