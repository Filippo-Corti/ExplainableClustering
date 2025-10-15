# Explainable Clustering

Dataset: [ILPD (Indian Liver Patient Dataset)](https://uci-ics-mlr-prod.aws.uci.edu/dataset/225/ilpd%2Bindian%2Bliver%2Bpatient%2Bdataset)

Students:    
Filippo Corti   
Carlotta Donato   
Giorgio Dal Santo   


### Notes:

1. Different clustering techniques were tried. Some of the most important questions were:
   - Does the clustering algorithm support mixed data types (numeric + categorical)?
   - Does the clustering algorithm requires/suggests scaling?
     - Which scaling is ideal?
   - What are the parameters of the algorithms? How do I pick the best ones?
     - Elbow rule
     - Silhouette score
   - How can we check if the clustering algorithm did a somewhat good job, looking only
   at the clustering result (as a preliminary analysis)?
     - We generally looked for classifications were clusters had balanced sizes. Most of the 
     algorithms failed in the sense that 90% of data points fell in the same cluster.

    The first phase revealed Spectral Clustering to perform the most promising clustering, with 
    results for k = 4, 6 where all clusters had meaningful sizes. 

1b. Why did the Spectral clustering solve this much better than any other algorithm?
   - The most important aspect was combining Spectral Clustering with Gower's Distance, which works 
    for mixed features and is particularly good at handling different types and scales.
   - Then, the Spectral Clustering works as follows:  
     1. Build a Full Connected Graph from the Dataset. The Graph should be weighted so that each
        connection (u, v) has a weight equal to the Gower's Distance between the datapoints u and v.
     2. Compute the Graph Laplacian Matrix, which corresponds to L = D - A (D = diagonal degree matrix; 
        A = adjacency matrix).
     3. Compute the eigenvalues of L and take the eigenvectors for the first n non-negligible eigenvalues.
     4. The eigenvectors can be used to compose a new matrix, which is necessarily 2D. (?)
     5. Use a classic clustering algorithm on the points of the 2D matrix. 
   
1c. Can we understand visually what the algorithm has seen?
   - Going from the 10D space of the features to a 2D space didn't produce meaningful results.
   - One idea to understand the algorithm visually was to obtain the 2D matrix of eigenvectors which was
    clustered during the Spectral Clustering. This however was also not clear.
   - A 3-step plan was conducted:
     1. Take the 10D space and compute Gower Distance. THIS ALLOWS TO "FOLLOW" THE SPECTRAL CLUSTERING LOGIC
     2. Use the Gower Distances to perform Spectral Embedding (same thing the clustering does) but 
        get to a 6D space instead of directly to a 2D space. THIS PRESERVES MANIFOLD STRUCTURE (CONNECTED
        DATA POINTS IN THE GRAPH APPEAR AT A SMALL DISTANCHE IN THE SPACE)
     3. Apply on the 6D space the t-SNE dimensionality reduction technique to get a 2D space.
        THIS PRESERVES LOCAL NEIGHBOURHOODS (BUT NOT DISTANCES), ALLOWING FOR A MORE SEPARABLE VIEW.
        Then visualize it.

2. Important questions now is "What do these clusters mean?":
   - First, can we classify clusters as "affected" and "not affected"?
   - Assuming we can:
     - What are the (biological) characteristics of the affected clusters?
     - What are the (biological) characteristics of the not affected clusters?