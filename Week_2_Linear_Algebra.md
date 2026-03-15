# Week 2: Linear Algebra for Data Science

Linear algebra is the foundational branch of mathematics that powers almost all modern data science and machine learning algorithms—from simple linear regression to complex deep learning architectures. It provides a highly efficient way to represent and manipulate large datasets (matrices) mathematically.

## 1. Algebraic View: Vectors and Matrices

### 1.1 Vector Operations
A vector is an array of numbers that can represent a point in space or a specific feature across data points.
*   **Vector Addition**: If $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}$, then $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$. Crucial for translating points in space.
*   **Scalar Multiplication**: Multiplying a vector by a real number $c$: $c\mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \end{bmatrix}$. Scales the magnitude of the vector without changing its direction (unless $c$ is negative, which reverses it).
*   **Dot Product (Inner Product)**: Measures how aligned two vectors are. It returns a scalar value.
    $$ \mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^{n} u_i v_i $$
    If the dot product is $0$, the vectors are orthogonal (perpendicular), meaning they share no direction.

### 1.2 Matrix Operations
A matrix is a 2D array of numbers. In data science, a dataset is typically represented as a matrix $\mathbf{X}$ composed of $n$ rows (observations/samples) and $p$ columns (features/variables).
*   **Matrix Multiplication**: If $\mathbf{A}$ is an $m \times n$ matrix and $\mathbf{B}$ is an $n \times p$ matrix, their product $\mathbf{C} = \mathbf{AB}$ is an $m \times p$ matrix.
    *   **Rule**: The number of columns in the first matrix must equal the number of rows in the second matrix.
    *   **Calculation**: Element $c_{ij}$ is the dot product of the $i$-th row of $\mathbf{A}$ and the $j$-th column of $\mathbf{B}$.
    *   **Non-commutative**: $\mathbf{AB} \neq \mathbf{BA}$ in general.
*   **Transpose ($\mathbf{A}^T$)**: Flips the matrix over its diagonal. Rows become columns and vice versa. $(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$.
*   **Identity Matrix ($\mathbf{I}$)**: A square matrix with $1$s on the diagonal and $0$s elsewhere. $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$.
*   **Inverse Matrix ($\mathbf{A}^{-1}$)**: For a square matrix $\mathbf{A}$, if an inverse exists, $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$. This is heavily used to solve systems of linear equations ($\mathbf{Ax} = \mathbf{b} \implies \mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$).
*   **Determinants Features**: 
    *   The determinant of a diagonal matrix is simply the explicitly calculated mathematically product of its distinct diagonal entries. 
    *   If one specific row of a given matrix is exactly a mathematical scalar multiple of a different row, the matrix explicitly physically collapses, mathematically yielding a determinant of strictly precisely $0$.

## 2. Rank and Null Space

### 2.1 Rank of a Matrix
The rank of a matrix is the maximum number of linearly independent row or column vectors. It essentially tells us the "true" dimensionality of the feature space the matrix represents.
*   **Linear Independence**: A set of vectors is linearly independent if no vector in the set can be written as a linear combination of the others.
*   **Full Rank**: An $n \times n$ square matrix is full rank if its rank is $n$. Such a matrix is invertible (its determinant is non-zero).
*   **Data Science Implication**: If your feature matrix $\mathbf{X}$ does not have full column rank, it means some features are perfectly predictable from others (perfect multicollinearity). This leads to redundancy, instabilty in models (like linear regression), and the matrix $\mathbf{X}^T\mathbf{X}$ being non-invertible.

### 2.2 Null Space
The null space of a matrix $\mathbf{A}$ is the set of all vectors $\mathbf{x}$ such that $\mathbf{A}\mathbf{x} = \mathbf{0}$.
*   It represents the dimensions that get collapsed to the origin when transformed by the matrix $\mathbf{A}$.
*   The dimension of the null space plus the rank of the matrix equals the number of columns (Rank-Nullity Theorem).

## 3. Over-determined Systems and Pseudo-inverse

### 3.1 Over-determined Systems
A system of linear equations $\mathbf{Ax} = \mathbf{b}$.
*   **Square System**: Number of equations equals unknowns ($m=n$). Typically has a single exact solution.
*   **Over-determined System**: More equations than unknowns ($m > n$). This is almost always the case in data science where you have far more samples (rows) than features (columns).
*   An exact solution $\mathbf{x}$ rarely exists because the vector $\mathbf{b}$ might not lie perfectly in the column space of $\mathbf{A}$ due to real-world noise.

### 3.2 The Pseudo-inverse (Moore-Penrose)
Since an exact solution is impossible, we seek the "best approximate solution" that minimizes the error $||\mathbf{Ax} - \mathbf{b}||^2$.
*   This is the Least Squares solution, foundational to Linear Regression.
*   The solution is found using the **Pseudo-inverse** ($\mathbf{A}^+$):
    $$ \mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b} $$
    Here, $\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$. It allows us to mathematically "solve" rectangular matrices.

## 4. Geometric View

### 4.1 Distances (Norms)
Norms measure the length or magnitude of a vector, which equates to calculating distances between data points.
*   **L2 Norm (Euclidean Distance)**: The straight-line distance. Most common.
    $$ ||\mathbf{x}||_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2} $$
*   **L1 Norm (Manhattan/City Block Distance)**: The sum of absolute differences. Useful in high-dimensional spaces or sparse data.
    $$ ||\mathbf{x}||_1 = |x_1| + |x_2| + ... + |x_n| $$
*   **L$\infty$ Norm (Chebyshev Distance)**: The maximum absolute value among the components.

### 4.2 Projections
Projection is the process of casting a vector $\mathbf{b}$ onto a line or a subspace defined by a vector $\mathbf{a}$.
*   Used heavily in dimensionality reduction (like PCA) where you want to project high-dimensional data points onto lower-dimensional surfaces that minimize the projection error (distance between point and subspace).

## 5. Eigenvalues and Eigenvectors

When a square matrix $\mathbf{A}$ acts on a specific vector $\mathbf{x}$, it typically changes both its magnitude and direction. However, for certain special vectors, the matrix only scales them (changes their length) without altering their fundamental direction.
These special vectors are **Eigenvectors**, and the scalar they are stretched by is the **Eigenvalue** ($\lambda$).
$$ \mathbf{Ax} = \lambda \mathbf{x} $$

### Finding Them:
1.  Solve the characteristic equation: $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ to find the eigenvalues $\lambda$.
2.  Plug each $\lambda$ back into $(\mathbf{A} - \lambda\mathbf{I})\mathbf{x} = \mathbf{0}$ to solve for the eigenvector $\mathbf{x}$.

### Data Science Significance: Principal Component Analysis (PCA)
Eigen decomposition is the mathematical engine behind PCA.
*   If we calculate the Covariance Matrix of our data, its eigenvectors represent the *principal components* (the directions in space where data varies the most).
*   The corresponding eigenvalues represent the amount of variance captured along that component.
*   By keeping only eigenvectors with the largest eigenvalues, we reduce the dimensionality of our data while preserving the most important information.

## 6. Hyperplanes and Halfspaces

*   **Hyperplane**: In a 2D space, it's a line. In 3D space, it's a 2D plane. In an N-dimensional space, it is a flat, $(N-1)$-dimensional subspace. Its equation is generally $\mathbf{w}^T\mathbf{x} + b = 0$.
*   **Halfspaces**: A hyperplane cleanly divides the space into two halves. A point $\mathbf{x}$ lies in the positive halfspace if $\mathbf{w}^T\mathbf{x} + b > 0$, and in the negative if $< 0$.
*   **Application**: This concept is the soul of linear classifiers, particularly **Support Vector Machines (SVMs)** and logistic regression, which try to find the best hyperplane to separate data belonging to different classes.