# SciViz Assigment #2

### Dataset:
4 Matrices which share the same rows (the genes). 
Each matrix, however, represents different, correlated aspects of the gene:
- BP: the Biological Processes
- CC: the Cellular Components
- MF: the Molecular Functions
- HPO: the Symptoms (Phenotypic Abnormality)

All matrices share the same genes (rows) but different sets of features (columns).These represent different “views” of the same entities (genes), describing different biological aspects.

### The plan:

1. Integrate the 4 binary matrices using two strategies:
   - Early Integration: simple concatenation into one large matrix. Then apply MOFA to the single-view.
   - Multi-View Integration: apply MOFA to the four, separate view.
2. Analyze the Latent Factors in order to understand the benefit of an approach such as MOFA:
   - Visualize the Latent Factors Embeddings (UMAP, PCA)
   - Variance Explained Plots
   - Heatmaps
   - Clustering, if we want
   - Check out mofax for some visualizations on MOFA


