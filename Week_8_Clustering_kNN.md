# Week 8: Clustering and Nearest Neighbors

This week contrasts one of the simplest supervised learning algorithms alongside one of the most foundational unsupervised learning algorithms. The common, unifying thread between them is their fundamental reliance on measuring multi-dimensional geometric **distance**.

## 1. K-Nearest Neighbors (kNN) - Supervised

kNN represents the ultimate "memory-based" or "lazy" learning framework. Unlike regression, it does completely away with constructing a generalized global mathematical equation ($Y = \beta X$) during a rigorous training phase.

### 1.1 Core Characteristics of kNN
*   **Non-parametric**: kNN does not derive a set of fixed parameters (like $\beta$ coefficients). It uses the data points themselves for classification. The only "parameter," $K$ (number of neighbors), is a tuning parameter selected by the user, not learned from the data.
*   **Lazy Learning**: Computation is deferred until a new data point needs to be classified. There is no explicit "training phase" involving complex optimization.
*   **Instance-based**: The algorithm approximates the classification function locally based on the immediate neighbors.

### 1.2 The Algorithm Steps
kNN operates on the intuitive assumption that "birds of a feather flock closely together." 
When a new test point ($X_{new}$) is introduced, the algorithm follows these four steps:
1.  **Calculate Distance:** Use a distance metric (e.g., Euclidean, Manhattan, or Mahalanobis) to find the distance between $X_{new}$ and *every single* labeled point in the training set. (Euclidean Distance / L2 Norm is the default standard for continuous variables: $d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$).
2.  **Order Neighbors:** Sort all calculated distances from smallest (closest) to largest (farthest).
3.  **Pick K Neighbors:** Select the top $k$ points with the smallest distances.
4.  **Majority Vote / Averaging Mechanism:**
    *   *For Classification Tasks*: Assign $X_{new}$ to the class that is most common among its $k$ nearest neighbors (Simple majority vote).
    *   *For Regression Tasks*: It strictly calculates the mean (or median) of the numeric target values possessed by the $k$ neighbors. 

### 1.3 The Crucial Importance of Feature Scaling
Because kNN calculates spherical distance blindly across all axes, it is critical to scale features so that one attribute with large values doesn't dominate.
*Example:* A dataset predicting buying behavior includes `Salary ($40,000 - $150,000)` and `Age (18 - 60)`. Without rigid standardization, the algorithm will functionally ignore the `Age` variable completely. 
*   **Mandatory Step**: You must apply **Standardization** (Z-score scaling: $\frac{x-\mu}{\sigma}$) or **Min-Max Normalization** prior to executing kNN.
*   **Curse of Dimensionality:** In very high-dimensional spaces, points become sparsely distributed, making the concept of "nearest" less meaningful and increasing computational costs.

### 1.4 Choosing the Optimal 'k' (The Bias-Variance Tradeoff)
*   **Small $k$ (e.g., $k=1$)**: The model rigidly memorizes the noise/outliers within the specific training set. It builds highly complex, jagged, "crisp" decision boundaries. Yields low bias, but overwhelmingly **High Variance** (Overfitting).
*   **Large $k$**: The model generalizes aggressively. Decision boundaries become completely smoothed out, diffuse, and more stable against noise. Yields low variance, but fundamentally **High Bias** (Underfitting).
*   *Rule of Thumb*: $K$ is often chosen as an odd number to avoid ties in binary classification.
*   *Optimization*: The optimal $k$ is structurally found by iteratively plotting cross-validated error rates against differing $k$ values and identifying the "valley" where the error minimizes.

## 2. K-Means Clustering - Unsupervised

K-Means is a radically different paradigm. We possess a vast database of independent variables ($X$), but absolutely no target labels ($Y$). The goal is strictly exploratory: organically segment $N$ observations into $K$ heterogeneous, non-overlapping subgroups (clusters) by finding hidden patterns and shared characteristics.

*   **Likeness vs. Distance**: "Likeness" between data points is measured using a distance metric (typically Euclidean distance). If a point is closer to the mean (centroid) of Cluster A than Cluster B, it is considered "more like" the points in Cluster A.
*   **Engineering Application**: K-means is powerful for anomaly detection. By clustering raw sensor data (temperature, pressure, etc.), you can identify "normal" operational clusters. Any new data that falls into a separate, distinct cluster might indicate an unstable or faulty process.

### 2.1 The Mathematical Foundation: Minimizing WCSS
Mathematically, the core objective of K-Means is to completely minimize the **Within-Cluster Sum of Squares (WCSS)**. 
We want the data points residing within a single cluster to be rigidly tight and as compact as possible, while keeping the geometric distance between fundamentally different clusters decisively large.
*   **Optimization**: The mathematical goal is to minimize the sum of squared distances between each data point and its assigned cluster centroid. This means high intra-cluster similarity and low inter-cluster similarity.

**The Core Variance Equation:**
For every K-means evaluation, the total variance is physically partitioned into two components:
$$ \text{Total SS} = \text{WCSS} + \text{BCSS} $$

| Term | R Code Equivalent | Meaning |
| :--- | :--- | :--- |
| **Total SS** | `km$totss` | Total variance in the data |
| **WCSS** | `km$tot.withinss` | Variance *inside* clusters (Intra-cluster: we want to **minimize** this) |
| **BCSS** | `km$betweenss` | Variance *between* clusters (Inter-cluster: we want to **maximize** this) |

### 2.2 The Iterative Lloyd's Algorithm
The algorithm cannot analytically jump to the correct answer; it follows a simple, repetitive iterative logic:
1.  **Initialization**: The user explicitly defines the hyperparameter $K$. The algorithm randomly selects $K$ points as the initial cluster centers (centroids).
2.  **Assignment**: For every single data point in the set, calculate its distance to all $K$ centroids. Assign the point to the cluster with the nearest centroid.
3.  **Update**: Once all points are assigned, calculate the new mean (average) of all points currently in each cluster. These new means become the updated centroids. 
4.  **Convergence**: Repeat the assignment and update steps iteratively. The loop permanently terminates when the centroids no longer move significantly or assignments stop changing.

### 2.3 Choosing the Optimal 'K' (The Elbow Method)
Since we possess no ground truth, we use the Elbow Method to find the best fit for $K$.
1.  Run the full K-Means algorithm repeatedly for a range of values (e.g., $K=1$ to $10$) and calculate the total WCSS for each.
2.  Plot the variance explained (or WCSS) directly against the number of clusters $K$. Note that WCSS strictly drops as $K$ increases.
3.  Look visually for the **"Elbow" or "Kink"** in the line graph—the specific coordinate where adding explicitly another cluster marginally stops yielding a drastically massive drop in variance.

### 2.4 Challenges and Limitations
*   **Sensitivity to Initial Guess**: Because the starting centroids are random, different starts can lead to different results. In practice, the algorithm is often run multiple times with different random starts (using `nstart` in R) to find the best overall fit.
*   **Cluster Shape**: K-means works best for spherical or compact clusters. It can struggle with complex, non-spherical shapes (like interlocking circles or long paths) because it relies solely on distance from a single center point.
*   **Feature Scaling**: Like kNN, since the algorithm relies on Euclidean distance metrics, it is vital to scale variables so that one attribute doesn't blindly dominate the calculations.

## 3. R Implementations

### 3.1 Developing kNN
```r
# library(class)
# Features must cleanly exclude the target column; 'cl' specifies the true labels
y_pred <- knn(train = scale(train_set[, -3]), 
              test = scale(test_set[, -3]), 
              cl = train_set[, 3], 
              k = 5)
```

### 3.2 Developing K-Means (Case Study: Uber Trips)
This demonstrates finding driving clusters for 91 Uber trips using parameters like speed, braking, duration, etc.

#### 1. Setup and Exploration
```r
# Clear existing variables
rm(list = ls())

# Set working directory and load data
# row.names=1 specifies the first column contains indexing, not feature variables
tripDetails <- read.csv("uber_data.csv", row.names = 1) 

# Explore structure
View(tripDetails) # Opens spreadsheet view
str(tripDetails)  # Verifies observations and variable types
summary(tripDetails) # Returns 5-point summaries (min, max, mean, etc.) for each attribute
```

#### 2. K-Means Implementation & Interpretation
```r
set.seed(42) # Guarantees reproducible starting centroids

# Run K-Means with 3 clusters
# x = dataset matrix, centers = K, nstart = multiple random starts
tripCluster <- kmeans(x = tripDetails, centers = 3, nstart = 20)

# Interpreting the Output Object
tripCluster$size    # Number of trips in each of the 3 clusters (e.g. 46, 15, 30)
tripCluster$centers # A table showing the average value of each variable within each cluster (e.g. Cluster 1="short trips")
tripCluster$cluster # A list showing exactly which group each of the 91 trips was assigned to
tripCluster$withinss # WCSS: The compactness of each formed group
```

#### 3. Finding Optimal K via Elbow Method Loop
To find the optimal K=3, you can use a loop:
```r
# Initialize an empty vector to store WCSS values
wcss <- vector()

# Run the K-means algorithm repeatedly from K=1 to K=10
for (i in 1:10) {
  wcss[i] <- sum(kmeans(tripDetails, centers = i)$withinss)
}

# Plot the calculated WCSS against K to visually locate the elbow
plot(1:10, wcss, type = 'b', main = paste('The Elbow Method'), xlab = 'Number of clusters', ylab = 'WCSS')
```

## 4. Key Takeaways from Practice Assignment

1.  **Distance Metrics in K-Means**: The most commonly used distance metric to calculate the distance between the centroid of each cluster and data points in the K-means algorithm is **Euclidean distance**.
2.  **kNN Training Phase**: It is **NOT TRUE** that explicit training and testing phases are involved while implementing kNN. Due to its "lazy learning" nature, computation is deferred until a new point needs classification. 
3.  **Importance of Scaling**: Scaling is absolutely vital in distance-based algorithms literally because **variables with higher magnitude will influence the results more**, artificially dominating variables with smaller ranges (e.g., Salary vs. Age).
4.  **K-Means Optimization Goal**: The fundamental objective of K-means clustering is that the **inter-cluster distance is maximized** (clusters are far apart) and the **intra-cluster distance is minimized** (points within a cluster are tightly grouped).
5.  **Determining Optimal Clusters**: Statistical techniques like the **Elbow method** and **Dendrograms** are formally useful for calculating the optimal number of clusters in unsupervised algorithms. A simple visual **Scatter plot** is **NOT useful** for determining this metric analytically.