# Study Notes: Unconstrained Multivariate Optimization
**Source:** NPTEL-NOC IITM Lectures (Data Science Series)

---

## 1. Introduction to Multivariate Optimization
Multivariate optimization involves finding the maximum or minimum of a function $f(x_1, x_2, \dots, x_n)$ where there are multiple decision variables. Unlike univariate (single-variable) optimization, these problems require vector and matrix algebra.

### Key Concepts in Visualization
* **3D Surfaces:** For two variables ($x_1, x_2$), the function value is represented as the height (Z-axis).
* **Contour Plots:** A 2D representation where lines (contours) connect points with the same function value.
    * Moving **along** a contour results in no change to the objective function.
    * To **improve** the function, you must move perpendicular to the contours toward the center (minimum) or outward (maximum).



---

## 2. Analytical Conditions for a Minimum
To find the optimum point ($x^*$) analytically, we use the multivariate equivalents of first and second derivatives.

### The Gradient Vector (First-Order Condition)
The **Gradient ($\nabla f$)** is a vector of partial derivatives for every variable.
* **Necessary Condition:** At the minimum point, the gradient must be zero ($\nabla f = 0$).
* This signifies a "stationary point" where the slope is zero in every direction.

### The Hessian Matrix (Second-Order Condition)
The **Hessian ($H$)** is a symmetric $n \times n$ matrix of second-order partial derivatives.
* **Sufficient Condition for Minimum:** The Hessian matrix must be **Positive Definite**.
* **The Check:** A matrix is positive definite if all its **eigenvalues** ($\lambda$) are strictly greater than zero ($\lambda > 0$).



---

## 3. Numerical Methods: Gradient Descent
In practice, most data science problems are too complex to solve analytically. Instead, we use iterative numerical methods like **Steepest Descent**.

### The Update Rule (The "Learning Rule")
To find the next best point ($x_{k+1}$), we use the following formula:
$$x_{k+1} = x_k + \alpha_k s_k$$

1.  **Current Point ($x_k$):** Your starting position in the current iteration.
2.  **Search Direction ($s_k$):** For steepest descent, this is the **negative gradient** ($-\nabla f$). This is the direction that goes "downhill" the fastest.
3.  **Step Length / Learning Rate ($\alpha_k$):** A scalar determining how far you move. This can be a fixed value or calculated via a "line search."



### Local vs. Global Optima
* **Global Minimum:** The absolute lowest point of the function.
* **Local Minimum:** A "valley" lower than its surroundings but not the absolute lowest.
* **Saddle Points:** Points that look like a minimum from one axis but a maximum from another.



---

## 4. Connections to Machine Learning
* **Learning Rule:** The update formula is exactly how neural networks adjust weights to minimize error.
* **Backpropagation:** An efficient way to calculate the gradient in multi-layered networks using the calculus chain rule.
* **Initialization:** Since many problems are "non-convex," the starting point ($x_0$) often determines if the algorithm finds the global best or gets stuck in a local minimum.

---

### Summary Table: Univariate vs. Multivariate
| Feature | Univariate (1 Variable) | Multivariate ($n$ Variables) |
| :--- | :--- | :--- |
| **First-Order** | $f'(x) = 0$ | $\nabla f = 0$ (Gradient Vector) |
| **Second-Order** | $f''(x) > 0$ | $H$ is Positive Definite (Eigenvalues $> 0$) |
| **Search Path** | Move left or right | Direction ($s$) and Step Size ($\alpha$) |
