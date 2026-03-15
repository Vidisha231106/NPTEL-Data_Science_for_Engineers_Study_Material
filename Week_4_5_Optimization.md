# Week 4 & 5: Optimization for Data Science

Optimization sits at the mathematical core of nearly all machine learning problems. When we say an algorithm is "learning" or "training," it is actually running an optimization routine to find the parameters that minimize an error or loss function.

## 1. Optimization Basics

An optimization problem is formulated as:
$$ \text{Minimize (or Maximize) } f(\mathbf{x}) $$
$$ \text{subject to } \mathbf{x} \in \mathcal{S} $$

*   **Objective Function ($f(\mathbf{x})$)**: The function indicating the cost, loss, or profit we want to minimize or maximize. In machine learning, this is usually a Loss Function (like Mean Squared Error or Cross-Entropy).
*   **Decision Variables ($\mathbf{x}$)**: The variables we can change/control to achieve the goal (e.g., the weights and biases in a neural network or a regression model).
*   **Feasible Set ($\mathcal{S}$)**: The set of all possible values the variables can take. If $\mathcal{S}$ equals all real numbers, it is an *unconstrained* problem. Otherwise, it is a *constrained* problem.

### 1.1 Unconstrained Multivariate Optimization
Looking for the absolute best point across the entire continuous parameter space.

#### Determining Optimality
To find local minima or maxima analytically drawn from calculus:
1.  **First-Order Necessary Condition (Stationary Point)**: For a point $\mathbf{x}^*$ to be an extremum, its tangent slope must be flat in all directions. Analytically, the gradient vector (matrix of first partial derivatives) must be zero:
    $$ \nabla f(\mathbf{x}^*) = 0 $$
2.  **Second-Order Sufficient Condition**: To determine if the stationary point is a minimum, a maximum, or a saddle point, we evaluate the **Hessian matrix** ($\nabla^2 f(\mathbf{x}^*)$, the matrix of second partial derivatives).
    *   If the Hessian is **Positive Definite** (all eigenvalues > 0), the point is a **Local Minimum** (bowl shape).
    *   If the Hessian is **Negative Definite** (all eigenvalues < 0), the point is a **Local Maximum** (hill shape).
    *   If the Hessian is **Indefinite**, it is a **Saddle Point**.

## 2. Gradient Descent (Steepest Descent)

Analytically solving $\nabla f(\mathbf{x}) = 0$ is computationally impossible for complex machine learning models with thousands or millions of parameters. We therefore use numerical, iterative methods. **Gradient Descent** is the most renowned.

### 2.1 The Concept
Imagine you are blindfolded on a mountainous terrain and want to reach the lowest valley. Your best strategy is to feel the slope of the ground under your feet (the gradient) and take a step in the direction of the steepest downward slope (the negative gradient).

### 2.2 The "Learning Rule"
The math dictates an iterative update of our parameters $\mathbf{x}$:
$$ \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k) $$

*   $\mathbf{x}_k$: The current position/parameters.
*   $\nabla f(\mathbf{x}_k)$: The gradient of the objective function at the current position. It points in the direction of steepest *ascent*. We subtract it to move downwards.
*   $\alpha$ (Alpha): The **Learning Rate** or step size.

### 2.3 The Impact of the Learning Rate ($\alpha$)
The choice of $\alpha$ is arguably the most critical hyperparameter in basic model training.
*   **Too Small**: The steps are tiny. The algorithm will eventually converge but it will take an excruciatingly long computational time.
*   **Too Large**: The steps overshoot the valley. Instead of converging, the algorithm bounces back and forth across the valley, potentially diverging towards infinity.

## 3. Advanced Optimization: Handling Constraints

Many real-world problems bound the decision variables (e.g., probabilities must be between 0 and 1, resource allocations must be non-negative).

### 3.1 Equality Constraints ($h(\mathbf{x}) = 0$)
Dealt with using the **Method of Lagrange Multipliers**.
*   Instead of minimizing just $f(\mathbf{x})$, we construct the **Lagrangian Function**:
    $$ \mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \sum_{i} \lambda_i h_i(\mathbf{x}) $$
*   We introduce a new variable $\lambda_i$ (the Lagrange multiplier) for every equality constraint.
*   We then find the stationary points by setting the gradient of $\mathcal{L}$ with respect to both $\mathbf{x}$ and $\lambda$ to zero.

### 3.2 Inequality Constraints ($g(\mathbf{x}) \leq 0$)
Dealt with using the **Karush-Kuhn-Tucker (KKT) Conditions**.
*   The KKT conditions are a set of mathematical requirements (stationarity, primal feasibility, dual feasibility, and complementary slackness) that generalize Lagrange multipliers to inequality constraints.
*   These are foundational to algorithms like **Support Vector Machines (SVMs)**, which maximize the margin between classes *subject to the constraint* that all data points are correctly classified.

## 4. Data Science Typology (Structured Problem Framework)

Optimization algorithms serve different "Types" of machine learning problems. Identifying your problem type ensures you select the correct objective function and metric.

1.  **Supervised Learning**
    *   You have an input feature matrix $\mathbf{X}$ and a known target variable $\mathbf{Y}$. The algorithm learns the mapping from $\mathbf{X}$ to $\mathbf{Y}$.
    *   **Regression**: The target $\mathbf{Y}$ is continuous (e.g., temperature, stock price).
        *   Objective function to minimize: Mean Squared Error (MSE).
    *   **Classification**: The target $\mathbf{Y}$ is discrete/categorical (e.g., Yes/No, Cat/Dog/Mouse).
        *   Objective function to minimize: Cross-Entropy (Log-Loss).

2.  **Unsupervised Learning**
    *   You only have input features $\mathbf{X}$, with no target labels. The goal is to uncover hidden structure in the data.
    *   **Clustering**: Grouping similar observations (e.g., customer segmentation).
    *   **Dimensionality Reduction**: Compressing the feature space while retaining the most information (e.g., PCA, where optimization is maximizing variance).