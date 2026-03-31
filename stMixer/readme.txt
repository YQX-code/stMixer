stMixer source tree
===================

This directory contains the main task-oriented scripts used in the stMixer study.

Subdirectories
--------------

1. intra-slice_integration
   Learns slide-specific representations from paired spatial modalities such as RNA + protein or RNA + ATAC.

   Main files:
   - configurations.py: command-line arguments for training.
   - sc_dataset.py: data loading and modality preprocessing.
   - scnet.py: graph encoder and cross-attention related model components.
   - main.py: training script for intra-slide integration.

2. mosaic_integration
   Aligns embeddings from two slides into a shared latent space for cross-slide mosaic integration.

   Main files:
   - test.py: training and evaluation script for triplet/MMD-based slide alignment.

3. label_transfer
   Transfers labels from a reference slide or reference modality to a query slide.

   Main files:
   - test.py: cluster-level voting and spatial refinement script for label transfer.

4. dataset
   Helper notes and preprocessing scripts for preparing histology-related inputs.

Usage notes
-----------

- The repository is organized around manuscript workflows rather than a packaged API.
- Most scripts expect local data files to be prepared in advance.
- Before running a script, check the file path variables at the top of the file and replace them with your local inputs.
- Input objects are generally stored as h5ad files with spatial coordinates in adata.obsm["spatial"].

Suggested reading order
-----------------------

1. Read the repository root README.md for installation and workflow overview.
2. Read dataset/readme.txt for expected input files.
3. Start with intra-slice_integration if you want to reproduce the basic embedding workflow.
4. Use mosaic_integration or label_transfer after embeddings and annotations are prepared.
