# Week 3: Statistics and Probability

Probability and statistics form the mathematical framework for handling uncertainty, making it possible to quantify the confidence in our machine learning models and infer population traits from small datasets.

## 1. Descriptive Statistics

Descriptive statistics summarize and organize characteristics of a data set. They do not draw conclusions beyond the data given.

### 1.1 Central Tendency
Where is the "middle" or "typical" value of the data?
*   **Mean ($\mu$ for population, $\bar{x}$ for sample)**: The arithmetic average. Highly sensitive to outliers.
    $$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$
*   **Median**: The middle value when data is sorted. If $n$ is even, it's the average of the two middle values. Robust to outliers (e.g., median income is preferred over mean income).
*   **Mode**: The value that appears most frequently. Used mostly for categorical data.

### 1.2 Dispersion (Spread)
How stretched or squeezed is the data?
*   **Range**: Difference between the maximum and minimum values.
*   **Variance ($\sigma^2$ for population, $s^2$ for sample)**: The average of the squared differences from the mean.
    $$ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$
    *(Note: Dividing by $n-1$ instead of $n$ for a sample is called Bessel's correction, giving an unbiased estimator of the population variance).*
*   **Standard Deviation ($\sigma$ or $s$)**: The square root of variance. Useful because it is in the same units as the original data.

### 1.3 Relationship Between Variables
*   **Covariance ($Cov(X,Y)$)**: Measures the directional relationship between two variables. Positive means they move together; negative means they move inversely. However, the magnitude is hard to interpret because it depends on the units of $X$ and $Y$.
    $$ Cov(X,Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) $$
*   **Correlation Coefficient ($r$ or $\rho$)**: Standardized covariance. It scales the value to be strictly between $-1$ (perfect negative linear relationship) and $+1$ (perfect positive linear relationship). An $r$ of $0$ means no *linear* relationship.
    $$ r = \frac{Cov(X,Y)}{s_x s_y} $$

## 2. Probability Theory

Probability quantifies the likelihood of events.
*   **Independent Events**: The probability of two independent events $A$ AND $B$ occurring simultaneously is the mathematically calculated product of their distinct probabilities: $P(A \cap B) = P(A) \times P(B)$.
*   **Union of Events**: The mathematical probability that strictly at least one of two particular events occurs is strictly formulated as: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

### 2.1 Random Variables
A random variable mathematically represents the outcome of a random phenomenon.
*   **Discrete Random Variable**: Takes on countable values (e.g., number of defective items in a batch).
*   **Continuous Random Variable**: Takes on any value within an interval (e.g., exact time it takes for a machine to degrade).

### 2.2 Probability mass and Density Functions
These functions describe the probability distribution of a random variable.
*   **Probability Mass Function (PMF)**: $P(X=x)$. Used for *discrete* variables. It gives the exact probability that the random variable $X$ is exactly equal to a value $x$. The sum of all probabilities must equal $1$.
*   **Probability Density Function (PDF)**: $f(x)$. Used for *continuous* variables. Because there are infinite possible points, the probability of $X$ being exactly a single point is zero. Instead, probabilities are given by the *area under the curve* over an interval $[a, b]$:
    $$ P(a \leq X \leq b) = \int_{a}^{b} f(x) dx $$
    The total area under the PDF curve is always exactly $1$.

## 3. Distributions

Distributions mathematically describe how probabilities are spread over the values of the random variable.

### 3.1 Univariate Normal (Gaussian) Distribution
The most important distribution in statistics. It is fully defined by its mean ($\mu$) and variance ($\sigma^2$) and is denoted as $N(\mu, \sigma^2)$.
*   **Shape**: Symmetric, bell-shaped curve centered at $\mu$. Mean = Median = Mode.
*   **Empirical Rule (68-95-99.7)**:
    *   $\sim 68\%$ of data falls within $\mu \pm 1\sigma$
    *   $\sim 95\%$ of data falls within $\mu \pm 2\sigma$
    *   $\sim 99.7\%$ of data falls within $\mu \pm 3\sigma$
*   **Standard Normal Distribution**: A normal distribution that has been standardized to have $\mu=0$ and $\sigma=1$. We convert any normal distribution to this using the Z-score: $Z = \frac{x - \mu}{\sigma}$.

### 3.2 Multivariate Normal Distribution
A generalization of the univariate normal distribution to two or more variables. Instead of a single mean, it has a mean vector $\boldsymbol{\mu}$, and instead of a single variance, it uses a Covariance Matrix $\boldsymbol{\Sigma}$.
*   The shape is a hyper-ellipsoid (like an oval in 2D or an egg in 3D).
*   Highly used in clustering algorithms like Gaussian Mixture Models (GMMs).

## 4. Inferential Statistics

Inferential statistics uses a random sample of data taken from a population to describe and make inferences about the population.

### 4.1 Hypothesis Testing
A formal mathematical procedure to decide whether to accept or reject beliefs about a population.
1.  **State the Hypotheses**:
    *   **Null Hypothesis ($H_0$)**: The status quo assertion. It usually assumes no effect, no difference, or no relationship. Example: "The new drug has no better effect than the old one."
    *   **Alternative Hypothesis ($H_a$ or $H_1$)**: What you are trying to prove. Example: "The new drug is more effective."
2.  **Choose a Significance Level ($\alpha$)**: The probability of rejecting the null hypothesis when it is actually true (Type I error). Typically set at $0.05$ (5%).
3.  **Calculate the Test Statistic and p-value**:
    *   **Test Statistic**: A standardized value calculated from sample data during a hypothesis test (e.g., t-statistic, z-statistic).
    *   **p-value**: The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct.
4.  **Make a Decision**:
    *   If **p-value $\leq \alpha$**: **Reject $H_0$**. The results are statistically significant.
    *   If **p-value $> \alpha$**: **Fail to reject $H_0$**. Insufficient evidence to suggest the alternative is true.

### 4.2 Confidence Intervals (CI)
Instead of a single point estimate (like saying the population mean is exactly $50$), a confidence interval provides a range of values describing the uncertainty surrounding the estimate.
*   **Format**: $\text{Estimate} \pm \text{Margin of Error}$
*   **Example**: "We are $95\%$ confident that the true population average height is between $165 \text{ cm}$ and $175 \text{ cm}$."
*   If we were to take 100 different samples and compute a 95% CI for each sample, approximately 95 of those intervals would contain the true population parameter.
*   The width of the CI is affected by the confidence level (higher confidence = wider interval) and sample size (larger sample = narrower, more precise interval).