# Data Integration with MOFA

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

### TODO:

1. Run MOFA training again, preserving labels (for analysis) DONE
2. Run cluster 2-100 on the MOFA factors matrix. Save as csv.
    - Real valued-data so KMeans? DONE
3. Run clusters 2-100 on each of the 4 matrices. Save as csv.
    - Binary data so KModes?
    - Issue: KModes is very slow on large datasets. 
    - Use MiniBatchKMeans as an approximation? Yes, it is appropriate because:
      - It is the same algorithm used for MOFA clustering, making for a fair comopairson
      - It has been proven that it works reasonably well on binary data


https://geneontology.org/docs/ontology-documentation/

https://github.com/bioFAM/MOFA2

#### MOFA2 tutorial to fit the model:
https://github.com/bioFAM/mofapy2/blob/master/mofapy2/notebooks/getting_started_python.ipynb

#### Visualization notebook:
https://github.com/bioFAM/mofax/blob/master/notebooks/getting_started_pbmc10k.ipynb


