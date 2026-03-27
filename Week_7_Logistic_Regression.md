# Week 7: Classification via Logistic Regression

While Linear Regression estimates continuous values, Logistic Regression is designed to predict discrete categorical outcomes—most commonly, binary outcomes (e.g., Fraud/Not Fraud, Survived/Died, 1/0).

## 1. Logistic Regression Theory

### 1.1 The Failure of Linear Regression for Classification
If you attempt to apply standard linear regression ($Y = \beta_0 + \beta_1X$) to a binary classification problem (where Y is simply 0 or 1), the regression line will inevitably predict values less than 0 and greater than 1. This violates the axioms of probability which must be strictly bounded between 0 and 1. While a simple linear function (like a hyperplane) can separate data into two halves, its unbounded nature (from $-\infty$ to $+\infty$) is a major drawback.

### 1.2 The Sigmoid (Logistic) Link Function
To elegantly solve this bounding issue, Logistic regression doesn't predict the class directly. Instead, it predicts the **probability** that an observation belongs to class 1, given the data $X$: $P(Y=1 | X)$.

To ensure the output mathematically stays strictly between 0 and 1, we wrap the linear regression equation inside a **Sigmoid Function**:
$$ p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} $$
This function effectively maps the linear decision boundary (the hyperplane) into a probabilistic value.
* If the exponent is $-\infty$, the probability becomes 0.
* If the exponent is $+\infty$, the probability becomes 1.

### 1.3 Odds and Log-Odds (The Logit)
How do we interpret the coefficients ($\beta$) now that we've added the sigmoid? Through the concept of odds.

*   **Odds**: The ratio of the probability of success to the probability of failure.
    $$ \text{Odds} = \frac{p(X)}{1 - p(X)} $$
*   **The Logit Transform**: If we take the natural logarithm of the odds, we magically recover our linear equation. This is also called the "log-odds ratio":
    $$ \ln\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1X_1 + \ldots + \beta_kX_k $$

### 1.4 Probabilistic Interpretation & Decision Boundaries
Using probabilities provides a more nuanced view than just "yes/no". 
A logistic regression model defines a decision boundary (hyperplane equation: $\beta_0 + \beta_1X = 0$) where the probability of a data point belonging to either class is exactly 0.5.
* Points far from the decision boundary have high confidence (probabilities near 0 or 1).
* Points near the boundary (where $p(X) \approx 0.5$) represent high uncertainty.

### 1.5 Optimization and Parameter Estimation
To find the best values for the parameters ($\beta$), the algorithm solves a nonlinear optimization problem using Maximum Likelihood Estimation.
* **Objective Function (Likelihood):** The goal is to maximize the product of probabilities for the observed data.
* For points in Class 1 ($y=1$), we maximize $p(X)$.
* For points in Class 0 ($y=0$), we maximize $1 - p(X)$.
* **Log-Likelihood:** The algorithm usually maximizes the log of this product to simplify calculations, turning products into sums.

### 1.6 Regularization: Avoiding Overfitting
When a model has many variables, it can become overly complex and "overfit" the training data. Regularization penalizes coefficients that don't significantly improve accuracy. A penalty term ($\lambda$) is added to the objective function:
* **L2 Regularization (Ridge):** Penalizes the square of the coefficients (e.g., $\beta_0^2 + \beta_1^2$). Results in less complex models that generalize better.
* **L1 Regularization (Lasso):** Penalizes the absolute values of coefficients.
* *Rule of Thumb:* A larger $\lambda$ means stronger regularization, trading off between fitting training data and keeping the model simple.

## 2. Performance Measures

Once we predict probabilities (e.g., $0.72$), we apply a classification threshold (usually $0.5$). If $p \geq 0.5$, we classify as Class 1 (Positive); if $p < 0.5$, Class 0 (Negative). 

### 2.1 The Confusion Matrix
The Confusion Matrix is the foundational table used to compare a classifier's predictions against actual results.

|                   | Actual Positive (1) | Actual Negative (0) |
|-------------------|---------------------|---------------------|
| **Predicted Positive (1)** | True Positive (TP)  | False Positive (FP) (Type 1 Error) |
| **Predicted Negative (0)** | False Negative (FN) (Type 2 Error) | True Negative (TN)  |

### 2.2 Key Performance Metrics
Based on the matrix, several metrics evaluate effectiveness:

1.  **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ (Total). 
    * Measures overall percentage of correct predictions. Can be misleading for heavily imbalanced data.
2.  **Sensitivity (Recall / True Positive Rate)**: $\frac{TP}{TP + FN}$
    * Measures how effectively the classifier identifies actual positive cases.
3.  **Specificity**: $\frac{TN}{TN + FP}$
    * Measures how effectively the classifier identifies actual negative cases.
4.  **Positive Predictive Value (Precision)**: $\frac{TP}{TP + FP}$
    * Of all predicted positives, how many are actually true? mitigating False Positives.
5.  **Negative Predictive Value**: $\frac{TN}{TN + FN}$
    * Of all predicted negatives, how many are actually true?
6.  **Balanced Accuracy**: The average of sensitivity and specificity.
7. **Kappa Statistic**: Benchmarks accuracy against expecting by random chance.
8. **Prevalence**: Proportion of positive cases in the total sample.

### 2.3 ROC Curve and Area Under Curve (AUC)
* **ROC Curve**: A plot of Sensitivity vs. (1 - Specificity). Increasing sensitivity often decreases specificity. A good classifier maintains high sensitivity without giving up too much specificity.
* **AUC**: A single number representing performance across all thresholds:
  * 0.9–1.0: Excellent
  * 0.8–0.9: Good
  * 0.7–0.8: Fair
  * 0.6–0.7: Poor

## 3. Cross-Validation

Cross-validation is a critical technique for selecting the optimal number of hyperparameters (or meta-parameters) in a model to ensure it generalizes well to new data.

### 3.1 Balancing Overfitting and Underfitting
The primary goal of cross-validation is to find the "sweet spot" in model complexity.
* **Overfitting:** Occurs when a model is too complex (too many parameters). It fits the training data perfectly (mean squared error near zero) but fails to predict new data accurately.
* **Underfitting:** Occurs when a model is too simple (too few parameters) to capture the underlying patterns in the data.
* **The Trade-off:** Increasing parameters reduces bias but increases variance. Cross-validation identifies the point where the Mean Squared Error (MSE) on a validation set is minimized, indicating the best trade-off.

### 3.2 Key Cross-Validation Techniques

#### 1. Validation Set Approach (Hold-out Method)
Used when you have a large dataset.
* **Process:** Divide data into a Training Set (e.g., 70%) to build the model and a Validation Set (e.g., 30%) to test it.
* **Selection:** Build models of varying complexity (e.g., linear, quadratic, cubic polynomials) and plot their MSE against the validation set. The complexity level with the lowest validation MSE is chosen as the optimal model.

#### 2. Leave-One-Out Cross-Validation (LOOCV)
Used when the dataset is small and you cannot afford to set aside a large validation block.
* **Process:** For *n* samples, you leave out exactly one sample for testing and use the remaining *n-1* for training. This is repeated *n* times so that every sample is used as the validation set exactly once.
* **Pros/Cons:** It is less biased than the hold-out method but is computationally expensive because the model must be rebuilt *n* times for every parameter choice.

#### 3. K-Fold Cross-Validation
A more computationally efficient alternative to LOOCV.
* **Process:** The dataset is divided into *K* equal-sized groups (folds). In each iteration, one fold is used as the validation set while the other *K-1* folds are used for training. This is repeated *K* times.
* **Common Practice:** Typically, *K* is set to 5 or 10. It provides a good balance between computational speed and error estimation accuracy.

### 3.3 Practical Applications
Cross-validation is not just for regression but is essential in various fields of data analytics:
* **Principal Component Analysis (PCA):** Selecting the number of relevant components.
* **Clustering (e.g., K-Means):** Determining the optimal number of clusters.
* **Polynomial Regression:** Deciding the highest degree of the polynomial to fit the data. (e.g. For automobile dataset plotting horsepower vs. mileage, a quadratic model is often optimal compared to higher-order models).

## 4. Implementation in R (Case Study: Automotive Crash Testing)

This example demonstrates classifying vehicles as either hatchbacks or SUVs based on crash test parameters.

### 4.1 Data Preparation

```r
# Setting working directory & loading datasets
# dataset contains predictors like head impact, body impact, etc.
crashTest_1 <- read.csv("crashTest_1.csv") # Training (80 obs)
crashTest_1_TEST <- read.csv("crashTest_1_TEST.csv") # Testing (20 obs)

# Verify structure and data types
str(crashTest_1)
summary(crashTest_1)
```

### 4.2 Building the Logistic Regression Model
Logistic regression falls under Generalized Linear Models (GLMs).

```r
# Fitting the model
# family = "binomial" specifically instructs R to use logistic regression
logisfit <- glm(CarType ~ ., 
                family = "binomial", 
                data = crashTest_1)

# Reviewing coefficients, deviance, and AIC
summary(logisfit)
```

### 4.3 Making Predictions & Probability Analysis

```r
# Predict probabilities on the test set
# type = "response" ensures outputs are probabilities
prob_pred <- predict(logisfit, type = 'response', newdata = crashTest_1_TEST)

# Analyze probabilities
# e.g. using tapply() to find mean probability per class
# lower prob -> Hatchbacks; higher prob -> SUVs

# Converting probabilities to strictly binary classifications based on a 0.5 threshold
y_pred <- ifelse(prob_pred > 0.5, "SUV", "Hatchback")
```

### 4.4 Evaluation Output

```r
# Using caret package to generate confusion matrix
library(caret)
# Generate confusion matrix and metrics (Accuracy, Sensitivity, Specificity, etc)
confusionMatrix(factor(y_pred), factor(crashTest_1_TEST$CarType))
```

## 5. Introduction to Classification Algorithms

Classification is a supervised learning task where an algorithm is trained on labeled data to distinguish between two or more classes. When a new, unlabeled data point is introduced, the algorithm predicts which of the predefined groups it belongs to.

### 5.1 Types of Classification Problems
* **Binary Classification:** The simplest form, involving only two classes (e.g., "Yes/No"). Examples include Equipment Monitoring ("Normal" vs "Abnormal") and Medical Diagnosis ("Cancerous" vs "Non-cancerous").
* **Multi-class Classification:** Involves three or more classes. For example, in equipment monitoring, "Abnormal" behavior could be further resolved into specific fault types like "Fault 1," "Fault 2," or "Fault 3".

### 5.2 Linearly Separable vs. Non-separable Problems
* **Linearly Separable:** Data can be perfectly split into groups using a straight line or a hyperplane.
* **Linearly Non-separable:** Data is organized such that no straight line can separate the classes without significant error. These require curved surfaces or non-linear decision boundaries.
* **Kernel Methods:** Used to solve non-linear problems by mapping them into a space where they can be solved using linear methods (the "kernel trick").

### 5.3 Classification Logic in Multi-class Problems
In multi-class scenarios, multiple hyperplanes can be used to create several regions. Each class is then identified based on which combination of these spaces it occupies (e.g. positive/negative half-spaces).

## 6. Multiple Linear Regression (MLR) in R

While classification predicts categories, **Multiple Linear Regression (MLR)** predicts continuous values using multiple variables. Here is a brief guide on MLR implementation.

### 6.1 Data Preparation and Exploration
Before building the model, it is crucial to check for inter-dependencies among variables.
* Use pairwise scatterplots `plot(dataset)` and a correlation matrix `cor(dataset)`.
* *Example:* If predicting a restaurant's dinner price based on Food rating, Decor rating, and Service rating, you might find a high correlation between Food and Service, suggesting redundancy.

### 6.2 Building the Linear Model in R
The `lm()` function is used for multiple linear regression.
```r
# Syntax: lm(DependentVariable ~ Indep1 + Indep2, data = dataset)
model <- lm(Price ~ Food + Decor + Service + East, data = NYC)

# Shorthand to regress against all other variables:
# model <- lm(Price ~ ., data = NYC)
```

### 6.3 Interpreting the Model
Use `summary(model)` to evaluate the significance of variables:
* **P-values:** Variables with a p-value < 0.05 are considered statistically significant predictors.
* **Insignificant Variables:** Variables with a high p-value (e.g., 0.99 for Service) do not significantly explain the price and can often be dropped.
* **R-squared ($R^2$):** Measures the proportion of variance in the target variable explained by the model.

### 6.4 Model Selection and Refinement
Refine the model by dropping insignificant variables:
1. Try building a new model without the insignificant variable (e.g., Service). If the $R^2$ remains almost identical, it confirms the variable wasn't contributing useful information.
2. For highly correlated variables (like Food vs. Service), experiment with dropping one or the other to see which retains a higher $R^2$.
3. Conclude with a final model containing only the most significant predictors, and optionally perform residual analysis to verify assumptions (constant variance and normality of errors).

## 7. Useful R Functions: The `apply` Family

When summarizing data in R (e.g., finding the mean of specific columns like PetalLength and PetalWidth in the `iris` dataset), you can use the `apply` family.

*   `apply(data, MARGIN, FUN)`: Best for matrices and dataframes where you specify the direction.
    *   `MARGIN = 1`: Applies the function row-wise (e.g., mean per observation).
    *   `MARGIN = 2`: Applies the function column-wise (e.g., mean per feature).
    *   *Example*: `apply(irisdata[, 3:4], 2, mean)` returns a vector of column means.
*   `lapply(data, FUN)`: Automatically applies to each column/list item and always returns a **list** (results have `$` signs).
    *   *Example*: `lapply(irisdata[, 3:4], mean)` works without any margin argument.
*   `sapply(data, FUN)`: Similar to `lapply` but returns a **simplified** output like a clean named vector.
    *   *Example*: `sapply(irisdata[, 3:4], mean)` provides a clean numeric vector.

*Note for exams: If you see options like `sapply(..., 2, mean)` or `lapply(..., 2, mean)`, the syntax is incorrect because only `apply()` requires and accepts the `MARGIN` (1 or 2) argument.*

## 8. Key Takeaways from Practice Assignment

1.  **Odds Ratio**: Defined formally as the ratio of the probability of an event occurring to the probability of the event not occurring.
2.  **Misclassification Rate**: Calculated from the confusion matrix as $\frac{\text{False Negative} + \text{False Positive}}{\text{Total number of samples}}$.
3.  **Sensitivity and Specificity Range**: The mathematical boundaries for both sensitivity and specificity lie strictly between **0 and 1**.
4.  **Logistic Regression Assumptions**: Logistic Regression assumes a linear relationship between the independent variables and the **log-odds (logit)** of the dependent variable, *not* the dependent variable itself, its log, or its sigmoid.
5.  **Confusion Matrix Outputs**: The standard confusion matrix for a binary classifier provides four distinct outputs: True Positives, False Positives, True Negatives, and False Negatives.