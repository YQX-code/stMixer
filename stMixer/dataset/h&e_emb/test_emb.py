import os
import numpy as np
import torch
from PIL import Image
import sys, os

# 把包含 miso/ 目录的父目录加入 sys.path
base_dir = "/home/yangqx/SC/data/BAS/BAS_copy/BAS1/miso"
sys.path.insert(0, base_dir)

# 验证一下，miso 应该在这个目录里：
assert os.path.isdir(os.path.join(base_dir, "miso")), \
    f"没找到 miso 包，请检查 {base_dir}/miso"
from torchvision import transforms
import scanpy as sc
from anndata import AnnData
from miso.hist_features import get_embeddings
from miso.hipt_model_utils import get_vit256, eval_transforms
from tqdm import tqdm

# 1. 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = "/home/yangqx/SC/data/BAS/BAS_copy/BAS1/miso/miso/checkpoints/vit256_small_dino.pth"

# 2. 加载模型
model = get_vit256(pretrained_weights=ckpt, arch="vit_small", device=device)
model.to(device).eval()

# 3. 预处理：和 miso 一致的 transform
transform = eval_transforms()  # 里面已经包含 Resize/CenterCrop/ToTensor/Normalize

def extract_spot_features(
    adata: AnnData,
    ckpt_path: str,
    arch: str = "vit_small",
    device: str = "cuda"
) -> np.ndarray:
    # 1) 加载 ViT-Small 模型，只返回 CLS token (384维)
    model = get_vit256(pretrained_weights=ckpt_path, arch=arch, device=device)
    model.to(device).eval()

    # 2) miso 自带的预处理：Resize(256), CenterCrop(256), ToTensor, Normalize
    transform = eval_transforms()

    feats = []
    for loc_path, glb_path in tqdm(
        zip(adata.obs["local_tile_path"], adata.obs["global_tile_path"]),
        total=adata.n_obs,
        desc="Extracting embeddings"
    ):
        # 3) 读取、预处理
        img_loc = Image.open(loc_path).convert("RGB")
        img_glb = Image.open(glb_path).convert("RGB")
        x_loc = transform(img_loc).unsqueeze(0).to(device)  # [1,3,256,256]
        x_glb = transform(img_glb).unsqueeze(0).to(device)

        # 4) 前向，只取 CLS token
        with torch.no_grad():
            out_loc = model(x_loc)
            out_glb = model(x_glb)
            # model 返回要么 Tensor，要么 (Tensor,) 或者 [Tensor]
            # 统一取第一个张量：
            emb_loc = out_loc[0] if isinstance(out_loc, (tuple,list)) else out_loc
            emb_glb = out_glb[0] if isinstance(out_glb, (tuple,list)) else out_glb

        # 5) 到 NumPy 并 reshape
        v_loc = emb_loc.cpu().numpy().reshape(-1)     # (384,)
        v_glb = emb_glb.cpu().numpy().reshape(-1)[:192]  # (192,)

        # 6) 拼成 576
        feat = np.concatenate([v_loc, v_glb], axis=0)  # (576,)
        feats.append(feat)

    return np.stack(feats, axis=0)  # (n_spots, 576)

adata = sc.read_h5ad("/home/yangqx/SC/data/BAS/BAS_copy/BAS1/bas1_with_two_scales.h5ad")
feats = extract_spot_features(adata,"/home/yangqx/SC/data/BAS/BAS_copy/BAS1/miso/miso/checkpoints/vit256_small_dino.pth")

# 4. 构造一个独立的 image-AnnData
var_names = [f"feat{i}" for i in range(feats.shape[1])]
adata_img = AnnData(
    X=feats,
    obs=adata.obs.copy(),
    var=Path.var if False else None  # 不保留 var
)
adata_img.obsm["spatial"] = adata.obsm["spatial"]

# 5. 保存
out_fp = "adata_image_modality.h5ad"
adata_img.write_h5ad(out_fp, compression="gzip")
print(f"✅ 已生成图像模态 AnnData：{out_fp}")

