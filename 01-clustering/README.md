# Explainable Clustering

Dataset: [ILPD (Indian Liver Patient Dataset)](https://uci-ics-mlr-prod.aws.uci.edu/dataset/225/ilpd%2Bindian%2Bliver%2Bpatient%2Bdataset)

Authors:    
Filippo Corti   
Carlotta Donato   
Giorgio Dal Santo   


## Repository structure:

- `/clustering` contains utility functions for the implementation of some clustering algorithms.
- `/data` contains the raw csv data, containing both the original dataset and the clustering labels.
- The notebooks contains:
  - `clustering.ipynb`: dataset loading and clustering.
  - `clustering_visualization.ipynb`: techniques to visualize the clusters graphically
  - `shap_analysis.ipynb`: training of a Random Forest and analysis of feature importance and SHAP values.
  - `clustering_summaries.ipynb`: statistics about the clusters and hypothesis tests on clusters' relevance.

