# K-Means Clustering On Customer Data
Implemented K-Means clustering algorithm on customer segmentation data from Kaggle using numpy, matplotlib, and pandas.

## Overview

K-Means Clustering is an unsupervised machine learning algorithm that groups together unlabeled data of various attributes via intrinsic similarities. The goal of this algorithm is to separate data into $K$ clusters where each data point is clustered into one of $K$ centroids. To do this, the algorithm aims to minimize the sum of square distances given by the following: 

$$
\sum_{i=1}^{K} \sum_{x \in C_i} \| x - \mu_i \|^2
$$

Where:
- $C_i$ is the set of points assigned to cluster $i$,
- $\mu_i$ is the centroid of cluster $i$.

The algorithm is as follows:

1. Assign each data point to the nearest centroid
2. Recalculate each centroid as the mean of the points in its cluster
3. Once the centroids do not change after step 2, the algorithm is finished

## Why it works

In step 1, we assign each point to the nearest centroid. It immediately becomes obvious that this step minimizes the squared distance for each point, since by definition we are choosing the $\mu_i$ that minimizes $\|x - \mu_i\|^2$.

In step 2, we set the centroids to the means of each cluster. To see why this minimizes the sum of square distances, we take the partial derivative with respect to the mean:

$$
\frac{\partial}{\partial \mu_i} \sum_{x \in C_i} \|x - \mu_i\|^2 = \sum_{x \in C_i} -2(x - \mu_i)
$$

Setting this derivative to 0:

$$
\sum_{x \in C_i} (x - \mu_i) = 0 \Rightarrow \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
$$

Therefore, the centroid that minimizes the sum of square distances is simply the mean of each cluster.

## Finding an Optimal K value Using the Elbow Method

Choosing a right value of $K$ is a common challenge in K-Means. One common method to do this is the **Elbow Method** in which one plots the total within-cluster sum of squares (WCSS) against specific $K$ values. The value of $K$ at the "elbow", or the point in the graph at which the magnitude of the derivative sharply decreases, is optimal. The "elbow" is a optimal due to a good tradeoff between the complexity, or the number of centroids, and minimizing the square error. 

## Results
![Elbow Method Plot](/elbow.png)
![Results](/results.png)