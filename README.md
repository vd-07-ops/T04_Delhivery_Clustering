# T04_Delhivery_Clustering

## ML Assignment 3: T04 (The Predictive Squad)
### Team Members:
* #####  Vedant Dave (202418014)
* #####  Sujal Dhrangadhariya (202418017)
* #####  Kashish Patel (202418044)
* #####  Jatin Sindhi (202418055)

---

### Problem Statement
According to Delhivery's dataset, we have applied clustering algorithms on basis of trip's duration and trip's distance


## Documentation


## 1. Data Preparation:
-> Read delivery data from the dataset file provided. The data contains various features related to trip details, such as trip durations, distances, actual times, and more.

### Dataset Link: https://www.kaggle.com/datasets/devarajv88/delhivery-logistics-dataset

## 2. Data Preprocessing:
-> Cleaned the data by handling missing values, removing irrelevant columns (e.g., 'source_name', 'destination_name') to focus on relevant trip features.

-> Removed outliers using distance calculations based on cluster centroids, identifying and excluding extreme values that might skew clustering results.

-> Applied normalization (StandardScaler) to ensure that the features are on the same scale for clustering algorithms.

## 3. Feature Engineering:
-> Created new features such as trip duration, actual time, and trip distance, and converted categorical features into numerical values where necessary.

-> Extracted date and time-based features like start hour and month for further analysis.

## 4. Visualization:
A distribution of trip durations was plotted using a histogram.
A correlation heatmap was created for numerical features to identify relationships between variables.
A boxplot visualized actual times across different route types.
Visualized the clustering results using PCA (Principal Component Analysis) after K-Means clustering to understand the groupings.
Used a pie chart to display the distribution of trips across different route types.

## 5. Clustering:

### (i) K-Means Clustering:

-> Performed K-Means clustering on the dataset to identify patterns in trip duration, distance, and other features.
Applied the Elbow Method to determine the optimal number of clusters.
Evaluated the clustering results using silhouette score and Davies-Bouldin score.

### (ii) Gaussian Mixture Model (GMM):

-> Used Gaussian Mixture Models for a probabilistic approach to clustering.
Evaluated GMM clustering results using silhouette and Davies-Bouldin scores.
Agglomerative Clustering:  Applied Agglomerative Clustering and visualized the results using a dendrogram.
Evaluated the clustering performance using silhouette and Davies-Bouldin scores.

### (iii) Agglomerative Clustering:

-> Applied Agglomerative Clustering and visualized the results using a dendrogram.

-> Evaluated the clustering performance using silhouette and Davies-Bouldin scores.

## 6. Model Training & Pseudocode:
### (i) K-Means Clustering:

#### Psuedocode:

-> Input: Data, K (number of clusters), random_state

-> Train:

Initialize centroids randomly

Assign each point to the nearest centroid

Recalculate centroids

Repeat until convergence

Predict: For each new data point, assign to the nearest centroid

### (ii) Gaussian Mixture Model (GMM):

#### Psuedocode:

-> Input: Data, K (number of clusters), random_state

-> Train:

Initialize Gaussian distributions for each cluster

Calculate the likelihood of each point belonging to each Gaussian

Update Gaussian parameters until convergence

Predict: For each data point, calculate the posterior probability of each cluster

### (iii) Agglomerative Clustering:

#### Psuedocode:

-> Input: Data, n_clusters

-> Train:

Start by treating each data point as a separate cluster

Merge the closest clusters iteratively

Stop when the desired number of clusters is reached

Predict: Assign each data point to a cluster based on the final merged clusters

## 7. Contributions / Novelty:

-> This project aims to segment trip data into meaningful clusters, identifying patterns related to trip durations, distances, and routes. By using various clustering algorithms like K-Means, GMM, and Agglomerative Clustering, the project provides valuable insights into delivery processes, potentially improving efficiency and route optimization.

-> K-Means was used for its simplicity and speed, while GMM offered a probabilistic approach that could model uncertainty in the data. Agglomerative Clustering provided a hierarchical perspective on clustering, revealing the structure in the data at various levels.

-> The novelty of this work lies in combining multiple clustering methods and evaluating their effectiveness using performance metrics such as silhouette score and Davies-Bouldin score, offering a more comprehensive understanding of the trip data.

## 8. Citations:

For Dataset: https://www.kaggle.com/datasets/devarajv88/delhivery-logistics-dataset

For Clustering Methods: https://developers.google.com/machine-learning/clustering/clustering-algorithms

For K-Means: https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

For Gaussian Mixture Model (GMM): https://scikit-learn.org/1.5/modules/mixture.html

For Agglomerative Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

