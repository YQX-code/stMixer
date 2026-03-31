Dataset and input notes
=======================

This directory stores helper material related to preparing inputs for stMixer workflows.

What the current scripts expect
-------------------------------

The repository currently works with locally prepared h5ad and npy files instead of a built-in downloader.

Typical inputs used by the scripts include:

1. Intra-slide integration
   - Two modalities from the same slide stored as h5ad files.
   - Shared spot or cell ordering across modalities.
   - Spatial coordinates stored in adata.obsm["spatial"].

2. Mosaic integration
   - Precomputed embeddings for two slides saved as h5ad files.
   - Cluster labels for both slides saved as npy files.
   - A bridging modality shared across slides, for example protein.

3. Label transfer
   - Query and reference embeddings saved as h5ad files.
   - Reference annotations stored in adata.obs.
   - Optional H&E embeddings for histology-guided transfer.

H&E embedding utilities
-----------------------

The h&e_emb directory contains two helper scripts:

- test_tiling.py
  Creates local and global image tiles for each spot from a Visium-style histology image.

- test_emb.py
  Extracts H&E features from the generated tiles and saves them as an AnnData object.

Expected AnnData fields
-----------------------

Depending on the workflow, the following fields are typically required:

- adata.X
  Feature matrix or embedding matrix.

- adata.obsm["spatial"]
  Two-dimensional spot or cell coordinates.

- adata.obsm["feat"]
  Optional precomputed modality features used before training.

- adata.obs["cluster_code"] or related annotation columns
  Used in some label transfer experiments.

- adata.obs["local_tile_path"]
- adata.obs["global_tile_path"]
  Used by the H&E feature extraction scripts.

Practical recommendation
------------------------

If you want to adapt stMixer to a new dataset, the easiest path is:

1. Convert each modality into h5ad format.
2. Ensure the spot order is aligned across paired modalities.
3. Store spatial coordinates in adata.obsm["spatial"].
4. Prepare any needed cluster labels or reference annotations.
5. Then run the corresponding script under intra-slice_integration, mosaic_integration, or label_transfer.
