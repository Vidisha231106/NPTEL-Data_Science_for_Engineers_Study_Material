# Week 8: Clustering and Nearest Neighbors

This week contrasts one of the simplest supervised learning algorithms alongside one of the most foundational unsupervised learning algorithms. The common, unifying thread between them is their fundamental reliance on measuring multi-dimensional geometric **distance**.

## 1. K-Nearest Neighbors (kNN) - Supervised

kNN represents the ultimate "memory-based" or "lazy" learning framework. Unlike regression, it does completely away with constructing a generalized global mathematical equation ($Y = \beta X$) during a rigorous training phase.

### 1.1 The Theoretical Mechanism
kNN operates on the intuitive assumption that "birds of a feather flock closely together." If a new data point perfectly structurally mirrors a set of known positive data points, it is extremely likely to also be positive.

1.  **Ingest new observation:** Given an entirely unclassified data point $X_{new}$.
2.  **Calculate Distance:** The mathematical distance between $X_{new}$ and *every single* data point existing in the known training database is meticulously computed. 
    *   **Euclidean Distance (L2 Norm)** is the default standard for continuous variables: $d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$.
    *   Other options include Manhattan distance, especially in higher dimensionality.
3.  **Find "k" Neighbors:** Sort the entire dataset globally by calculated distance in ascending order. Isolate the top $k$ nearest data points.
4.  **Voting/Averaging Mechanism:**
    *   *For Classification Tasks*: The algorithm holds a simple majority vote. If $k=5$ and the nearest neighbors are [Cat, Cat, Dog, Cat, Dog], $X_{new}$ is definitively classified as a **Cat**.
    *   *For Regression Tasks*: It strictly calculates the mean (or median) of the numeric target values possessed by the $k$ neighbors. 

### 1.2 The Crucial Importance of Feature Scaling
Because kNN calculates spherical distance blindly across all axes, variables with massive numerical ranges will structurally dominate the mathematical output. 
*Example:* A dataset predicting buying behavior includes `Salary ($40,000 - $150,000)` and `Age (18 - 60)`. Without rigid standardization, the algorithm will functionally ignore the `Age` variable completely because the coordinate differences in `Salary` are tens of thousands of units larger. 
*   **Mandatory Step**: You must apply **Standardization** (Z-score scaling: $\frac{x-\mu}{\sigma}$) or **Min-Max Normalization** prior to executing kNN.

### 1.3 Choosing the Optimal 'k' (The Bias-Variance Tradeoff)
*   **Highly Small $k$ (e.g., $k=1$)**: The model rigidly memorizes the noise/outliers within the specific training set. It builds highly complex, jagged decision boundaries. Yields low bias, but overwhelmingly **High Variance** (Overfitting).
*   **Highly Large $k$**: The model generalizes aggressively. Decision boundaries become completely smoothed out. Yields low variance, but fundamentally **High Bias** (Underfitting).
*   *Optimization*: The optimal $k$ is structurally found by iteratively plotting cross-validated error rates against differing $k$ values and identifying the "valley" where the error minimizes.

## 2. K-Means Clustering - Unsupervised

K-Means is a radically different paradigm. We possess a vast database of independent variables ($X$), but absolutely no target labels ($Y$). The mathematical goal is strictly exploratory: organically segment the data into $K$ heterogeneous, non-overlapping subgroups (clusters).

### 2.1 The Underlying Objective Function
Mathematically, the core objective of K-Means is to completely minimize the **Within-Cluster Sum of Squares (WCSS)**. 
We want the data points residing within a single cluster to be rigidly tight and highly similar, while keeping the geometric distance between fundamentally different clusters decisively large.

### 2.2 The Iterative Lloyd's Algorithm
The algorithm cannot analytically jump to the correct answer; it must systematically guess and dynamically update:
1.  **Initialization**: The user explicitly defines the hyperparameter $K$ (e.g., 3). The algorithm procedurally picks $K$ random observations from the dataset to act as the initial cluster "centroids" (the physical centers of the groups).
2.  **Assignment Step (Expectation)**: For strictly every individual data point in the entire dataset, the algorithm recalculates its Euclidean distance to all $K$ centroids. It rigidly assigns the point to the single cluster whose centroid is mathematically closest.
3.  **Update Step (Maximization)**: Now that points are temporarily assigned, the structurally old centroids are obsolete. The algorithm computes the exact mathematical mean of all coordinates for all points belonging to a cluster. This new mean coordinate becomes the dynamically updated centroid. 
4.  **Convergence**: Steps 2 and 3 iteratively repeat on a loop. The centroids physically "move" across the scatterplot. The loop permanently terminates when the centroid locations functionally stop changing (or shift invisibly), signaling mathematical convergence to a local minimum.

### 2.3 Choosing the Optimal 'K' (The Elbow Method)
Since we possess no ground truth, we cannot calculate "accuracy." How do we know how many clusters legitimately exist?
1.  Run the full K-Means algorithm repeatedly for $K=1, K=2, \ldots, K=10$.
2.  Compute the total calculated WCSS for every iteration.
3.  Plot the WCSS value directly against the number of clusters $K$. Note that WCSS strictly drops as $K$ increases (if $K$ equals population size, WCSS is literally 0).
4.  Look visually for the **"Elbow" or "Kink"** in the line graph—the specific coordinate where adding explicitly another cluster marginally stops yielding a drastically massive drop in variance.

## 3. R Implementations (Conceptual)

### 3.1 Developing kNN
```r
# library(class)
# Features must cleanly exclude the target column; 'cl' specifies the true labels
y_pred <- knn(train = scale(train_set[, -3]), 
              test = scale(test_set[, -3]), 
              cl = train_set[, 3], 
              k = 5)
```

### 3.2 Developing K-Means
```r
set.seed(42) # Guarantees the random starting centroids are reproducible
# nstart = 20 forces the algorithm to independently run 20 times with 20 distinct random starting configurations, returning exclusively the one mathematically best result to prevent "bad randomized luck."
kmeans_model <- kmeans(dataset, centers = 3, nstart = 20)

# The assigned clusters for every row can be aggressively pulled out via:
# kmeans_model$cluster
```