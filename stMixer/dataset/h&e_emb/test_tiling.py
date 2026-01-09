import os
from pathlib import Path
import scanpy as sc
import numpy as np
from PIL import Image
from tqdm import tqdm

def crop_with_padding(img: Image.Image, x0: int, y0: int, w: int, h: int) -> Image.Image:
    """
    从 img 上以 (x0,y0) 大小 (w,h) 做裁剪，超出边界部分用黑色填充，返回大小恒定为 (w,h) 的 Image。
    """
    W, H = img.size
    left   = max(x0, 0)
    upper  = max(y0, 0)
    right  = min(x0 + w, W)
    lower  = min(y0 + h, H)
    crop   = img.crop((left, upper, right, lower))
    out    = Image.new(img.mode, (w, h))
    paste_x = left - x0
    paste_y = upper - y0
    out.paste(crop, (paste_x, paste_y))
    return out

def tiling_two_scales(
    adata,
    out_path: str,
    library_id: str = None,
    local_size: int = 256,
    global_size: int = 4096,
    quality: str = "hires",
    copy: bool = False
):
    """
    对每个 spot：
      1) 读取 spot 在原图上的像素中心 (cx, cy)
      2) 裁‘local’ 256×256 窗口；裁‘global’ 4096×4096 窗口
      3) 字段写入 adata.obs["local_tile_path"] 和 ["global_tile_path"]
    """
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    # 读取 hires 图像
    img_arr = adata.uns["spatial"][library_id]["images"][quality]
    if img_arr.dtype in (np.float32, np.float64):
        img_arr = (img_arr * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_arr)
    H, W  = img_arr.shape[:2]

    # 计算每个 spot 的像素坐标
    sf = adata.uns["spatial"][library_id]["scalefactors"]
    scale = sf["tissue_hires_scalef"]
    # print(scale)
    # exit()
    cols  = adata.obsm["spatial"][:,0] * scale
    rows  = adata.obsm["spatial"][:,1] * scale

    os.makedirs(out_path, exist_ok=True)
    local_paths, global_paths = [], []

    for spot, cx, cy in tqdm(zip(adata.obs_names, cols, rows),
                             total=len(adata), desc="Tiling spots"):
        cx, cy = float(cx), float(cy)

        # 1. Local
        lx0 = int(cx - local_size/2)
        ly0 = int(cy - local_size/2)
        local_img = crop_with_padding(img_pil, lx0, ly0, local_size, local_size)
        local_fname = f"{spot}_local_{local_size}.jpeg"
        local_path  = Path(out_path)/local_fname
        local_img.save(local_path, "JPEG")

        # 2. Global
        gx0 = int(cx - global_size/2)
        gy0 = int(cy - global_size/2)
        global_img = crop_with_padding(img_pil, gx0, gy0, global_size, global_size)
        global_fname = f"{spot}_global_{global_size}.jpeg"
        global_path  = Path(out_path)/global_fname
        global_img.save(global_path, "JPEG")

        local_paths.append(str(local_path))
        global_paths.append(str(global_path))

    # 写回 adata.obs
    adata.obs["local_tile_path"]  = local_paths
    adata.obs["global_tile_path"] = global_paths

    if copy:
        return adata
    # 否则就直接在原 adata 上修改

# ------------------------------
# 在你的脚本中替换原 tiling 调用
# ------------------------------

# adata = sc.read_h5ad("/home/yangqx/SC/data/BAS/BAS_copy/BAS1/adata_image_modality.h5ad")
# print(adata)
# exit()

adata = sc.read_h5ad("/home/yangqx/SC/data/BAS/BAS1/bas1_s.h5ad")

# 假设你的 library_id 就叫 None，或者按你实际改
lib = list(adata.uns['spatial'].keys())[0]
hires = adata.uns['spatial'][lib]['images']['hires']
# 水平+垂直翻转
hires_fixed = hires.swapaxes(0, 1)
# 覆盖回去
adata.uns['spatial'][lib]['images']['hires'] = hires_fixed


library_id = list(adata.uns["spatial"].keys())[0]
adata.uns["spatial"][library_id]["use_quality"] = "hires"

# 计算 imagerow, imagecol 同你原来那段代码
sf = adata.uns["spatial"][library_id]["scalefactors"]
scale = sf["tissue_hires_scalef"]
coords = adata.obsm["spatial"]
adata.obs["imagecol"]  = coords[:,0] * scale
adata.obs["imagerow"]  = coords[:,1] * scale

# 调用新的 tiling_two_scales
out_dir = "tiles_two_scales"
tiling_two_scales(
    adata,
    out_path=out_dir,
    library_id=library_id,
    local_size=16,
    global_size=256,
    quality="hires",
    copy=False      # 直接在 adata 上写入
)

# 保存结果
adata.write_h5ad("bas1_with_two_scales.h5ad")
print("完成：每个 spot 已拥有 local 256×256 及 global 4096×4096 窗口。")
