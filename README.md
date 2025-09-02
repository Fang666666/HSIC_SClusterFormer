# PyTorch Demo of the Hyperspectral Image Classification method - SClusterFormer.

This repository contains the PyTorch implementation of the paper:

**​​Deformable Convolution-Enhanced Hierarchical Transformer With Spectral-Spatial Cluster Attention for Hyperspectral Image Classification​**

Using the code should cite the following paper:

Y. Fang, L. Sun, Y. Zheng and Z. Wu, "Deformable Convolution-Enhanced Hierarchical Transformer With Spectral-Spatial Cluster Attention for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 34, pp. 701-716, 2025, doi: 10.1109/TIP.2024.3522809. 

@ARTICLE{10820058,
  author={Fang, Yu and Sun, Le and Zheng, Yuhui and Wu, Zebin},
  journal={IEEE Transactions on Image Processing}, 
  title={Deformable Convolution-Enhanced Hierarchical Transformer With Spectral-Spatial Cluster Attention for Hyperspectral Image Classification}, 
  year={2025},
  volume={34},
  number={},
  pages={701-716},
  keywords={Feature extraction;Transformers;Convolution;Computational modeling;Data mining;Sun;Mixers;Computer science;Support vector machines;Redundancy;Hyperspectral image classification;hierarchical Transformer;multi-feature;cluster attention},
  doi={10.1109/TIP.2024.3522809}}

This project has referred to the following works:

[1] L. Sun, G. Zhao, Y. Zheng and Z. Wu, "Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5522214, doi: 10.1109/TGRS.2022.3144158.

[2] Y. Fang, Q. Ye, L. Sun, Y. Zheng and Z. Wu, "Multiattention Joint Convolution Feature Representation With Lightweight Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-14, 2023, Art no. 5513814, doi: 10.1109/TGRS.2023.3281511.

# Description.
Vision Transformer (ViT), known for capturing non-local features, is an effective tool for hyperspectral image classification (HSIC). However, ViT’s multi-head selfattention (MHSA) mechanism often struggles to balance local details and long-range relationships for complex highdimensional data, leading to a loss in spectral-spatial information representation. To address this issue, we propose a deformable convolution-enhanced hierarchical Transformer with spectral-spatial cluster attention (SClusterFormer) for HSIC. The model incorporates a unique cluster attention mechanism that utilizes spectral angle similarity and Euclidean distance metrics to enhance the representation of fine-grained homogenous local details and improve discrimination of non-local structures in 3D HSI and 2D morphological data, respectively. Additionally, a dual-branch multiscale deformable convolution framework augmented with frequency-based spectral attention is designed to capture both the discrepancy patterns in high-frequency and overall trend of the spectral profile in low-frequency. Finally, we utilize a cross-feature pixel-level fusion module for collaborative cross-learning and fusion of the results from the dual-branch framework. Comprehensive experiments conducted on multiple HSIC datasets validate the superiority of our proposed SClusterFormer model, which outperforms existing methods.
