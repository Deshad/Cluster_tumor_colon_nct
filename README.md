# Model Selection for Clustering

This project explores clustering techniques applied to high-dimensional datasets derived from various feature extraction methods used on images processed by machine learning models. The primary objective is to assess how different dimensionality reduction techniques and clustering algorithms impact the quality and structure of the resulting clusters.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Conclusion](#conclusion)

---

## Introduction

Clustering is a critical task in machine learning, particularly when working with high-dimensional data. This project investigates the clustering of feature representations extracted from deep learning models—PathologyGAN, ResNet50, InceptionV3, and VGG16. These features, derived from 5,000 colorectal cancer tissue images, were analyzed using dimensionality reduction methods such as PCA and UMAP, followed by clustering.

Key metrics for evaluation:
- **Silhouette Score**: Measures cluster separation and cohesion.
- **V-Measure Score**: Evaluates clustering alignment with ground-truth labels.

---

## Methodology

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Retained the top 100 principal components with the highest variance.
- **UMAP (Uniform Manifold Approximation and Projection)**: Reduced data to 100 components, capturing both local and global structures.

### Clustering Algorithms
1. **K-Means**: Partition-based clustering optimized by minimizing intra-cluster variance.
2. **Hierarchical Clustering**: Agglomerative clustering using the Ward linkage method.
3. **Louvain Clustering**: A graph-based method for community detection optimized for modularity.

### Evaluation Framework
- Randomly sampled subsets of 200 points for K-Means and Louvain clustering.
- Larger subset of 1,000 points for Hierarchical clustering.
- Evaluation using Silhouette and V-Measure scores.

---

## Results

| Representation          | K-Means | Hierarchical Clustering | Louvain Clustering |
|--------------------------|---------|--------------------------|---------------------|
| **VGG16 (UMAP)**         | 0.63    | 0.71                     | 0.67                |
| **VGG16 (PCA)**          | 0.63    | 0.58                     | 0.68                |
| **PathologyGAN (PCA)**   | 0.43    | 0.548                    | -                   |
| **PathologyGAN (UMAP)**  | 0.43    | 0.6147                   | -                   |

---

## Conclusion

This study demonstrates the importance of dimensionality reduction and clustering algorithm selection in high-dimensional data analysis:
- **UMAP** outperformed PCA in producing well-defined clusters and higher V-Measure scores.
- **Hierarchical Clustering** paired with UMAP was the most consistent and effective approach, particularly for the VGG16 dataset.
- **K-Means** performed adequately but struggled with non-spherical cluster shapes.
- **Louvain Clustering** excelled in fine-grained clustering with optimal resolution tuning.

These findings highlight the interplay between dimensionality reduction techniques, clustering algorithms, and parameter optimization in achieving robust clustering performance.
