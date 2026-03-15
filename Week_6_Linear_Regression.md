# Week 6: Linear Regression & Model Assessment

Linear regression is the quintessential statistical method for modeling the mathematical relationship between a dependent (response) variable and one or more independent (predictor) variables. It is the building block for neural networks and predictive analytics.

## 1. Simple Linear Regression (SLR)

SLR deals with a single independent variable $X$ forecasting a continuous continuous dependent variable $Y$.
The theoretical population model is defined as:
$$ Y = \beta_0 + \beta_1X + \epsilon $$
*   **$Y$**: The dependent variable (what we want to predict).
*   **$X$**: The independent variable.
*   **$\beta_0$**: The population intercept (the expected value of $Y$ when $X = 0$).
*   **$\beta_1$**: The population slope (the expected change in $Y$ for a 1-unit increase in $X$).
*   **$\epsilon$**: The unobservable error term (noise that the model cannot capture).

### 1.1 Building the Model: Ordinary Least Squares (OLS)
Because we only have a sample, we estimate the theoretical model as:
$$ \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i $$
*   **$\hat{y}_i$** is the *predicted* value.
*   The difference between the actual value $y_i$ and the predicted value $\hat{y}_i$ is the **Residual ($e_i$)**: $e_i = y_i - \hat{y}_i$.

The **Ordinary Least Squares (OLS)** method determines the best $\hat{\beta}_0$ and $\hat{\beta}_1$ by minimizing the **Sum of Squared Errors (SSE)**:
$$ \text{Minimize } \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1x_i)^2 $$
Using calculus (taking derivatives with respect to the betas and setting them to zero), OLS yields deterministic formulas for the coefficients.

### 1.2 Verifying Linear Regression Assumptions
For the OLS estimates to be "Best Linear Unbiased Estimators" (BLUE) and for hypothesis tests on the coefficients to be valid, four strictly defined assumptions (often remembered as **LINE**) must hold:

1.  **L - Linearity**: The relationship between $X$ and the mean of $Y$ is linear.
    *   *Check*: Plot residuals vs. fitted values. It should look like a random cloud without a discernible structural pattern (like a U-shape).
2.  **I - Independence**: Observations are independent of each other (especially important for time-series data).
    *   *Fix*: If temporal correlation is found, one might need Time Series models (ARIMA).
3.  **N - Normality**: The residuals ($e_i$) must be normally distributed around a mean of zero.
    *   *Check*: Q-Q (Quantile-Quantile) Plot. The points should lie closely on the 45-degree reference line.
4.  **E - Equal Variance (Homoscedasticity)**: The variance of the residuals remains constant across all values of the independent variable $X$.
    *   *Check*: Residuals vs. Fitted plot. If the cloud of points "fans out" like a megaphone, you have **Heteroscedasticity**. This renders standard errors and p-values unreliable.

## 2. Model Assessment

How much of the reality does our model actually capture?

1.  **Total Sum of Squares (SST)**: The total variance in $Y$ around its mean. $\sum (y_i - \bar{y})^2$.
2.  **Regression Sum of Squares (SSR)**: The variance in $Y$ explained by the regression line. $\sum (\hat{y}_i - \bar{y})^2$.
3.  **Sum of Squared Errors (SSE)**: The unexplained variance. $\sum (y_i - \hat{y}_i)^2$.
*Mathematically, $SST = SSR + SSE$.*

### 2.1 R-squared ($R^2$)
Known as the **Coefficient of Determination**. It represents the proportion of variance in the dependent variable explained by the independent variable(s).
$$ R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST} $$
*   Ranges from $0$ (predicts no better than the simple mean line) to $1$ (perfect prediction).

### 2.2 Adjusted R-squared
A major flaw with standard $R^2$ in Multiple Linear Regression is that it *always* increases (or stays the same) when you add more predictors, even if those predictors are purely random noise. This tricks you into overfitting.
**Adjusted $R^2$** incorporates a penalty for adding extra, non-contributing variables:
$$ \text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right] $$
($n$ = sample size, $k$ = number of predictors). Use Adjusted $R^2$ to compare models with different numbers of predictors.

## 3. Multivariate Linear Regression
Extends SLR to handle $k$ predictors.
$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_kX_k + \epsilon $$

### The Curse of Multicollinearity
This is a critical issue where two or more independent variables are highly correlated with each other.
*   **Consequence**: The model cannot accurately mathematically distinguish the individual effect of each variable on $Y$. Coefficient estimates become wildly unstable and standard errors artificially inflate.
*   **Detection**: Calculate the **Variance Inflation Factor (VIF)**. If VIF $> 10$ for a variable, severe multicollinearity exists, and that variable should likely be dropped or combined via PCA.

## 4. Diagnostics: Outliers and Leverage

Regression relies on means and variances, which are sensitive to extreme values.
1.  **Outliers**: Observations where the actual $Y$ value is very far from the predicted $\hat{Y}$ value (large residual).
2.  **Leverage Points**: Observations that have an extreme or unusual value for an $X$ predictor. They have the *potential* to drag the regression line heavily towards them.
3.  **Influential Points**: A data point that fundamentally changes the slope/intercept of the regression line if it is removed. A point is typically influential if it is *both* an outlier and has high leverage.
    *   *Metric*: Measured using **Cook’s Distance**. Points with high Cook's Distance warrant deep investigation.

## 5. Subset Selection
Finding the "best" combination of variables from a large pool of potential predictors.
1.  **Best Subset Selection**: Bruteforce fits every possible model ($2^k$ models). Computationally impossible if $k$ is large.
2.  **Stepwise Selection (Heuristic approach)**:
    *   **Forward Stepwise**: Start with a strictly intercept-only model. Add the predictor that gives the lowest p-value / highest adj-$R^2$. Repeat until no remaining variables statistically improve the model.
    *   **Backward Stepwise**: Start with all variables. Remove the variable with the highest p-value (least significant). Repeat until all remaining variables are statistically significant.