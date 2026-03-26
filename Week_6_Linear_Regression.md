# Week 6: Linear Regression & Model Assessment

Linear regression is the quintessential statistical method for modeling the mathematical relationship between a dependent (response) variable and one or more independent (predictor) variables. It is the building block for neural networks and predictive analytics.

## 1. Predictive Modelling Foundations: Measuring Relationships

Before building complex models, it is essential to measure the relationship between variables.

### 1.1 Basic Statistical Measures
*   **Sample Mean ($\bar{X}$)**: The average value of a set of observations.
*   **Sample Variance ($s^2$ or $\sigma^2_X$)**: A measure of how much the values deviate from the mean.
*   **Sample Covariance ($s_{XY}$ or $Cov(X,Y)$)**: Measures whether two variables move together. 
    *   **Formula**: $Cov(X,Y) = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{n-1}$
    *   *Positive* $\to$ move in the same direction. *Negative* $\to$ move in opposite directions.

### 1.2 Visualization via Scatter Plots
Before calculating numbers, a scatter plot provides a qualitative check:
*   **Positive Trend**: Points move up and to the right.
*   **Negative Trend**: Points move down and to the right.
*   **No Correlation**: Points are scattered randomly with no clear pattern.

### 1.3 Correlation Coefficients
1.  **Pearson’s Correlation Coefficient ($r$)**: Quantifies linear dependency between two variables (ranges from -1 to +1). It is effectively a standardized version of covariance.
    *   **Formula (using Covariance)**: $r = \frac{Cov(X,Y)}{\sigma_X \cdot \sigma_Y}$
    *   **Formula (expanded algebraic)**: $r = \frac{n\sum xy - \sum x \sum y}{\sqrt{[n\sum x^2 - (\sum x)^2][n\sum y^2 - (\sum y)^2]}}$
    *   $+1$ is a perfect positive linear relationship, $-1$ is a perfect negative linear relationship, and $0$ indicates no linear relationship. *Limitations*: Not robust against outliers and cannot be used for ordinal (ranked) variables.
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

### 2.2 Ordinary Least Squares (OLS) Method Requirements
We estimate the theoretical model using OLS: $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$. 
OLS identifies the "best fit" line that minimizes the sum of squared vertical distances (Sum of Squared Errors or SSE) between data points and the line.
$$ \text{Minimize } \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 $$

**Calculating the Coefficients**:
*   **Slope ($\beta_1$ or $b$)**: $b = \frac{SS_{XY}}{SS_{XX}}$ (How much X and Y vary *together* divided by how much X varies *alone*).
    *   Algebraic formula: $b = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - (\sum x_i)^2}$
*   **Intercept ($\beta_0$ or $a$)**: $a = \bar{y} - b\bar{x}$
    *   *Note: Finding the intercept forces the regression line through the mean point $(\bar{x}, \bar{y})$. The regression line ALWAYS passes through the mean point!*

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

### 3.3 The F-Test for Model Adequacy and Sum of Squares
Compares a Full Model (with slope) against a Reduced Model (just a constant mean). The foundation of this lies in the Key Relationship of variance:
**$$ \text{SST} = \text{SSR} + \text{SSE} $$**
*   **SST (Sum of Squares Total)**: Total variation in actual Y values ($\sum (y_i - \bar{y})^2$).
*   **SSR (Sum of Squares Regression)**: Variation explained by the model ($\sum (\hat{y}_i - \bar{y})^2$).
*   **SSE (Sum of Squares Error)**: Variation unexplained / residuals ($\sum (y_i - \hat{y}_i)^2$).

If the **F-Statistic** indicates a large error reduction compared to random error, the model is deemed useful.

### 3.4 Coefficient of Determination ($R^2$) and Adjusted $R^2$
*   **$R^2 = SSR / SST$**: A value between 0 and 1 indicating the proportion of variance in $Y$ explained by $X$.
*   **Adjusted $R^2$**: Standard $R^2$ always increases when adding predictors, risking overfitting. Adjusted $R^2$ incorporates a penalty for extra, unnecessary variables.
    *   **Formula**: $\bar{R}^2 = 1 - (1 - R^2) \times \frac{n-1}{n-k-1}$
    *   where $n$ = number of observations, and $k$ = number of predictor variables.

### 3.5 Practical Example & Software: Model Building in R
Building and analyzing a linear regression model in R follows a systematic workflow. Consider a dataset measuring Bond "Coupon Rates" (independent variable $X$) against their "Bid Prices" (dependent variable $Y$).

1.  **Data Loading & Visualization**:
    *   Load data: `bonds <- read.delim("bonds.txt", row.names=1)`
    *   Always plot first: `plot(bonds$CouponRate, bonds$BidPrice)`. A visual check might immediately reveal a linear trend alongside potential outliers pulling the expected line away from the bulk of the data.
2.  **Building the Linear Model**: 
    *   The `lm()` command creates a linear model. The tilde (`~`) tells R to regress $Y$ against $X$.
    *   Syntax: `model <- lm(BidPrice ~ CouponRate, data=bonds)`
    *   Visualizing the fit: `abline(model)` overlays the calculated "Best Fit" line onto the scatter plot.
3.  **Interpreting the `summary(model)` Output**:
    *   **Residuals**: Provides the 5-number summary (Min, 1Q, Median, 3Q, Max). A median close to zero indicates balanced errors.
    *   **Coefficients (Estimates)**: Gives values for $\hat{\beta}_0$ (Intercept) and $\hat{\beta}_1$ (Slope). e.g., A slope of 3.06 means for every 1% increase in the coupon rate, the bid price rises by 3.06 units.
    *   **Statistical Significance**: Look at the P-values. R assigns stars (`***`) to heavily significant variables. A P-value $< 0.05$ means you can reject the null hypothesis.
    *   **Model Fit**: Review the Multiple $R^2$ (explained variance) and F-Statistic (overall model viability).

## 4. Multivariate Linear Regression
Extends SLR to handle $k$ predictors.
$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_kX_k + \epsilon $$

### The Curse of Multicollinearity
This is a critical issue where two or more independent variables are highly correlated with each other.
*   **Consequence**: The model cannot accurately mathematically distinguish the individual effect of each variable on $Y$. Coefficient estimates become wildly unstable and standard errors artificially inflate.
*   **Detection**: Calculate the **Variance Inflation Factor (VIF)**. If VIF $> 10$ for a variable, severe multicollinearity exists, and that variable should likely be dropped or combined via PCA.

## 5. Diagnostics to Improve Linear Model Fit: Residuals

Traditional metrics like $R^2$ and F-Tests are not enough. As shown by Anscombe’s Quartet, vastly different datasets can have identical statistical summaries but require completely different models. Residual analysis serves as the ultimate "truth-teller."

### 5.1 Understanding Residuals Math
1.  **The Predicted Value ($\hat{y}_i$)**: $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$
2.  **The Boxed Residual Formula ($e_i$)**: 
    $$ \text{Residual} = \text{Actual Value} - \text{Predicted Value} $$ 
    $$ e_i = y_i - \hat{y}_i $$
    *(Note: For an OLS regression, the sum of all residuals must always equal zero: $\sum e_i = 0$)*.
    *   **Positive Residual**: Actual > Predicted $\to$ The model *underestimates* the reality.
    *   **Negative Residual**: Actual < Predicted $\to$ The model *overestimates* the reality.
3.  **Estimating Error Variance ($\hat{\sigma}^2$)**: Estimated using the Sum of Squared Errors: $\hat{\sigma}^2 = \frac{\sum(y_i - \hat{y}_i)^2}{n-k}$ (where $k$ is the number of parameters).
4.  **Standardized Residuals ($d_i$)**: Raw residuals depend on the data's scale. We standardize them: $d_i = \frac{e_i}{\hat{\sigma} \sqrt{1-h_{ii}}}$
    *   *$h_{ii}$ is the Leverage ("Hat Value"), measuring how far an $X$ value is from the mean of all $X$ values.*

### 5.2 Worked Example: Calculating a Single Residual Error
Understanding how to extract data from a table to find a single residual is a core concept. Let's use the Bond data where $\hat{y} = 74.7865 + 3.066x$.

1.  **Find the Predicted Value**: Want to check Case 13 where the Coupon Rate ($x$) is $3$.
    *   $\hat{y} = 74.7865 + 3.066(3) = 83.9845$
2.  **Find the Actual Value**: Look up Case 13 in the raw data table. The actual Bid Price ($y$) is $94.50$.
3.  **Calculate Residual**: $94.50 - 83.9845 = +10.5155$.
    *   Because it is a positive residual, the model under-predicted by $10.5155$.
    *   *Visual Intuition*: The actual point ($94.50$) sits high above the regression line drawn at $83.98$. A large residual immediately flags this point as a potential outlier.
We typically plot Standardized Residuals (Y-axis) against Predicted Values (X-axis).
*   **The "Random Cloud" (Good Sign)**: Points scatter randomly around the zero-line. $E[e_i] = 0$ and $Var(e_i) = \sigma^2$. The linear model is adequate.
*   **The "Curved Pattern" (Nonlinearity)**: Residuals form a U-shape or inverted U-shape. This mathematically means $Y$ is a function of $X^2$ or $X^3$. *Fix: Add a polynomial/quadratic term to the model.*
*   **The "Funnel Shape" (Heteroscedasticity)**: Spread gets wider/narrower as predictions increase. Means $Var(e_i)$ is not constant. *Fix: Use Weighted Least Squares instead of OLS.*

### 5.3 Normality Check: The Q-Q Plot
To verify if errors are Normally Distributed, we use a Quantile-Quantile plot. It graphs "theoretical quantiles" against "sample quantiles." If points fall roughly on a 45-degree straight line, the normality assumption is satisfied.

### 5.4 Outliers and Outlier Detection Strategy
Points lying far from the horizontal zero-line (Standardized residual $|d_i| > 2$ or $3$) are **outliers**.

**One-at-a-time removal strategy**:
1. Identify the point with the largest standardized residual magnitude.
2. Remove *only* that one point.
3. Re-run the regression and re-check the residual plot.
*Why?* A single massive outlier can "smear" results, making perfectly good points look like outliers. Deleting them all at once leads to data loss. (e.g., removing a few outliers can jump an $R^2$ from 0.75 to 0.99).

## 6. Definitions: Outliers, Leverage, and Influence

Regression relies on means and variances, which are highly sensitive to extreme values.
1.  **Outliers**: Observations where the actual $Y$ value is very far from the predicted $\hat{Y}$ value (large residual).
2.  **Leverage Points (High $h_{ii}$)**: Observations that have an extreme or unusual value for an $X$ predictor. They have the *potential* to drag the regression line heavily towards them.
3.  **Influential Points**: A data point that fundamentally changes the slope/intercept of the regression line if it is removed. A point is typically influential if it is *both* an outlier and has high leverage.
    *   *Metric*: Measured using **Cook’s Distance**. Points with high Cook's Distance warrant deep investigation.

## 7. Subset Selection
Finding the "best" combination of variables from a large pool of potential predictors.
1.  **Best Subset Selection**: Bruteforce fits every possible model ($2^k$ models). Computationally impossible if $k$ is large.
2.  **Stepwise Selection (Heuristic approach)**:
    *   **Forward Stepwise**: Start with a strictly intercept-only model. Add the predictor that gives the lowest p-value / highest adj-$R^2$. Repeat until no remaining variables statistically improve the model.
    *   **Backward Stepwise**: Start with all variables. Remove the variable with the highest p-value (least significant). Repeat until all remaining variables are statistically significant.