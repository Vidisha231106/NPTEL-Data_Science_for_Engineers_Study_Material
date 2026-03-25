# Week 6: Linear Regression & Model Assessment

Linear regression is the quintessential statistical method for modeling the mathematical relationship between a dependent (response) variable and one or more independent (predictor) variables. It is the building block for neural networks and predictive analytics.

## 1. Predictive Modelling Foundations: Measuring Relationships

Before building complex models, it is essential to measure the relationship between variables.

### 1.1 Basic Statistical Measures
*   **Sample Mean ($\bar{X}$)**: The average value of a set of observations.
*   **Sample Variance ($s^2$)**: A measure of how much the values deviate from the mean.
*   **Sample Covariance ($s_{XY}$)**: Measures how two variables change together. Positive covariance means both increase or decrease together.

### 1.2 Visualization via Scatter Plots
Before calculating numbers, a scatter plot provides a qualitative check:
*   **Positive Trend**: Points move up and to the right.
*   **Negative Trend**: Points move down and to the right.
*   **No Correlation**: Points are scattered randomly with no clear pattern.

### 1.3 Correlation Coefficients
1.  **Pearson’s Correlation Coefficient ($r$)**: Quantifies linear dependency between two variables (ranges from -1 to +1). $+1$ is a perfect positive linear relationship, $-1$ is a perfect negative linear relationship, and $0$ indicates no linear relationship. *Limitations*: Not robust against outliers and cannot be used for ordinal (ranked) variables.
2.  **Spearman’s Rank Correlation ($\rho$)**: Used for ordinal variables or when the relationship is nonlinear but monotonic (consistently increasing or decreasing). Raw data is converted into ranks before calculation.
3.  **Kendall’s Rank Correlation ($\tau$)**: Another measure for ordinal variables based on concordant (moving in the same direction) and discordant (moving in opposite directions) pairs. Often used for expert agreement.

**The Anscombe's Quartet Warning**: Four visually distinctly different scatter plots (one linear, one curved, one with an outlier, etc.) can produce the exact same Pearson correlation coefficient. Always visualize your data rather than relying solely on a number.

## 2. Simple Linear Regression (SLR)

SLR deals with a single independent variable $X$ forecasting a continuous dependent variable $Y$.
*   **Dependent Variable ($Y$)**: The response, predicted, or output variable.
*   **Independent Variable ($X$)**: The predictor, regressor, or input variable.
*   **Simple vs. Multiple Regression**: Simple uses one $X$, Multiple uses several independent variables ($X_1, X_2...$).
*   **Soft Sensors**: Engineering application estimating difficult-to-measure properties (e.g., strength) via easier variables (e.g., temperature).

### 2.1 The Simple Linear Regression Model
The theoretical population model is defined as:
$$ Y_i = \beta_0 + \beta_1 X_i + \epsilon_i $$
*   **$\beta_0$ (Intercept)**: The expected value of $Y$ when $X = 0$.
*   **$\beta_1$ (Slope)**: Indicates how much $Y$ changes for every one-unit change in $X$.
*   **$\epsilon_i$ (Error/Residual)**: Difference between actual observation and prediction, covering measurement errors or unquantified factors.

### 2.2 Ordinary Least Squares (OLS) Method
We estimate the theoretical model using OLS: $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$. 
OLS identifies the "best fit" line that minimizes the sum of squared vertical distances (Sum of Squared Errors or SSE) between data points and the line.
$$ \text{Minimize } \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 $$

### 2.3 Verifying Linear Regression Assumptions (LINE)
1.  **L - Linearity**: The relationship between $X$ and the mean of $Y$ is linear.
2.  **I - Independence**: Observations are independent of each other.
3.  **N - Normality**: The residuals ($e_i$) must be normally distributed around a mean of zero.
4.  **E - Equal Variance (Homoscedasticity)**: The variance of the residuals remains constant across all values of $X$.

## 3. Model Assessment & Hypothesis Testing

Once a model is fitted, we must determine if it is adequate, if the coefficients are significant, and if assumptions are met.

### 3.1 Properties of Least Squares Estimates
The estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ derived from a sample have certain properties:
*   **Unbiasedness**: The expected value of the estimates is the true population value ($E[\hat{\beta}] = \beta$).
*   **Variability**: The variance of the slope depends on error variance ($\sigma^2$) and independent variable variance ($s_{xx}$).
*   **Estimating Error**: Estimated from data using SSE divided by $n-2$ degrees of freedom.

### 3.2 Hypothesis Testing for Coefficients
We test the Null Hypothesis ($H_0$): $\beta_1 = 0$ (X has no effect on Y).
*   We use the **T-Distribution** since error variance is estimated from data.
*   **Confidence Intervals**: If the 95% confidence interval for a coefficient includes zero, the coefficient is insignificant.
*   **P-Value**: A low p-value allows us to reject the null hypothesis, concluding the coefficient is significant.

### 3.3 The F-Test for Model Adequacy
Compares a Full Model (with slope) against a Reduced Model (just a constant mean).
*   **SST (Total Sum of Squares)**: Total variance in $Y$ around its mean.
*   **SSE (Sum of Squared Errors)**: Unexplained variance.
*   **SSR (Regression Sum of Squares)**: Variance explained by the model ($SST - SSE$).
If the **F-Statistic** indicates a large error reduction compared to random error, the model is deemed useful.

### 3.4 Coefficient of Determination ($R^2$) and Adjusted $R^2$
*   **$R^2 = SSR / SST$**: A value between 0 and 1 indicating the proportion of variance in $Y$ explained by $X$.
*   **Adjusted $R^2$**: Standard $R^2$ always increases when adding predictors, risking overfitting. Adjusted $R^2$ incorporates a penalty for extra, unnecessary variables.

### 3.5 Practical Example & Software
*   **Service Agent Performance**: Using a dataset (units repaired vs. minutes spent). If a generated 95% confidence interval for the intercept includes zero, the intercept is insignificant (makes sense: 0 units repaired = 0 time taken). If the slope's interval avoids zero, it is highly significant.
*   **Using R**: The `lm()` command creates a linear model (e.g., `model <- lm(minutes ~ units, data=dataset)`), providing coefficients, residuals range, and $R^2$.

## 4. Multivariate Linear Regression
Extends SLR to handle $k$ predictors.
$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_kX_k + \epsilon $$

### The Curse of Multicollinearity
This is a critical issue where two or more independent variables are highly correlated with each other.
*   **Consequence**: The model cannot accurately mathematically distinguish the individual effect of each variable on $Y$. Coefficient estimates become wildly unstable and standard errors artificially inflate.
*   **Detection**: Calculate the **Variance Inflation Factor (VIF)**. If VIF $> 10$ for a variable, severe multicollinearity exists, and that variable should likely be dropped or combined via PCA.

## 5. Diagnostics: Outliers and Leverage

Regression relies on means and variances, which are sensitive to extreme values.
1.  **Outliers**: Observations where the actual $Y$ value is very far from the predicted $\hat{Y}$ value (large residual).
2.  **Leverage Points**: Observations that have an extreme or unusual value for an $X$ predictor. They have the *potential* to drag the regression line heavily towards them.
3.  **Influential Points**: A data point that fundamentally changes the slope/intercept of the regression line if it is removed. A point is typically influential if it is *both* an outlier and has high leverage.
    *   *Metric*: Measured using **Cook’s Distance**. Points with high Cook's Distance warrant deep investigation.

## 6. Subset Selection
Finding the "best" combination of variables from a large pool of potential predictors.
1.  **Best Subset Selection**: Bruteforce fits every possible model ($2^k$ models). Computationally impossible if $k$ is large.
2.  **Stepwise Selection (Heuristic approach)**:
    *   **Forward Stepwise**: Start with a strictly intercept-only model. Add the predictor that gives the lowest p-value / highest adj-$R^2$. Repeat until no remaining variables statistically improve the model.
    *   **Backward Stepwise**: Start with all variables. Remove the variable with the highest p-value (least significant). Repeat until all remaining variables are statistically significant.