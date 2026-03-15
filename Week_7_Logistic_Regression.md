# Week 7: Classification via Logistic Regression

While Linear Regression estimates continuous values, Logistic Regression is designed to predict discrete categorical outcomes—most commonly, binary outcomes (e.g., Fraud/Not Fraud, Survived/Died, 1/0).

## 1. Logistic Regression Theory

### 1.1 The Failure of Linear Regression for Classification
If you attempt to apply standard linear regression ($Y = \beta_0 + \beta_1X$) to a binary classification problem (where Y is simply 0 or 1), the regression line will inevitably predict values less than 0 and greater than 1. This violates the axioms of probability.

### 1.2 The Sigmoid (Logistic) Link Function
To elegantly solve this, Logistic regression doesn't predict the class directly. Instead, it predicts the **probability** that an observation belongs to class 1, given the data $X$: $P(Y=1 | X)$.

To ensure the output mathematically stays strictly between 0 and 1, we wrap the linear regression equation inside a **Sigmoid Function**:
$$ p(X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} $$
This function forms a classic S-shaped curve perfectly bounded by the horizontal asymptotes $y=0$ and $y=1$.

### 1.3 Odds and Log-Odds (The Logit)
How do we interpret the coefficients ($\beta$) now that we've added the sigmoid? Through the concept of odds.

*   **Odds**: The ratio of the probability of success to the probability of failure.
    $$ \text{Odds} = \frac{p(X)}{1 - p(X)} $$
    *(If $p=0.8$, the odds are $0.8/0.2 = 4$. We say the odds are 4 to 1).*
*   **The Logit Transform**: If we take the natural logarithm of the odds, we magically recover our linear equation:
    $$ \ln\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1X_1 + \ldots + \beta_kX_k $$
*   **Interpretation**: $\beta_1$ represents the change in the *log-odds* of the outcome for a 1-unit increase in $X_1$, holding all other variables constant.

*(Note: Because no closed-form calculus solution exists for logistic regression coefficients, it utilizes advanced optimization—namely, Maximum Likelihood Estimation via Gradient Descent—to find the betas).*

## 2. Performance Measures (The Confusion Matrix)

Once we predict probabilities (e.g., $0.72$), we apply a classification threshold (usually $0.5$). If $p \geq 0.5$, we classify as Class 1 (Positive); if $p < 0.5$, Class 0 (Negative). We then build a **Confusion Matrix** to see where the model made errors.

|                   | Actual Positive (1) | Actual Negative (0) |
|-------------------|---------------------|---------------------|
| **Predicted Positive (1)** | True Positive (TP)  | False Positive (FP) |
| **Predicted Negative (0)** | False Negative (FN) | True Negative (TN)  |

*   **FP (Type I Error)**: Boy cries wolf.
*   **FN (Type II Error)**: Doctor misses a cancer diagnosis (usually the most dangerous error).

### 2.1 Deriving Business Metrics
Depending on your business problem, you optimize for different metrics derived strictly from this matrix:

1.  **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
    *   Proportion of completely correct predictions. Extremely misleading if the data is heavily imbalanced (e.g., 99% of transactions are non-fraud; predicting "non-fraud" every time yields 99% accuracy but fails the business goal).
2.  **Precision**: $\frac{TP}{TP + FP}$
    *   *Question*: Out of all the observations the model declared as positive, how many were *actually* positive?
    *   Focuses on mitigating False Positives. Crucial when the cost of a false alarm is high (e.g., spam filter deleting an important email).
3.  **Recall (Sensitivity / True Positive Rate)**: $\frac{TP}{TP + FN}$
    *   *Question*: Out of all the actual positive observations in the real world, how many did the model *successfully find*?
    *   Focuses on mitigating False Negatives. Crucial when missing a positive is catastrophic (e.g., cancer screening, fraud detection).
4.  **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
    *   The Harmonic Mean of Precision and Recall. Used when you need a single metric to balance the precision/recall trade-off, especially on imbalanced datasets.

## 3. Cross-Validation

If you train your model on Dataset A and evaluate it on Dataset A, you will get an artificially inflated accuracy score. The model has "memorized" the data.

**k-Fold Cross-Validation** represents the gold standard for reliably estimating model performance on unseen data.
1.  The complete dataset is randomly shuffled and divided into $k$ mutually exclusive, identically sized subsets (folds). Common values are $k=5$ or $k=10$.
2.  The model is structurally trained $k$ distinct times.
3.  In iteration 1, fold 1 is withheld as the "Test" set, and the model trains on folds 2 through $k$. The accuracy is calculated on fold 1.
4.  In iteration 2, fold 2 is withheld, and it trains on folds 1, 3...$k$.
5.  This repeats $k$ times. The final generalized accuracy is the calculated average of the $k$ resulting metrics.

## 4. Implementation in R

Logistic regression falls under the umbrella of Generalized Linear Models (GLMs).

```r
# Split logic (using caTools)
library(caTools)
set.seed(123)
split <- sample.split(dataset$TargetClass, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Fitting the Logistic Regression Model
# family = binomial explicitly dictates the use of the sigmoid link function
classifier <- glm(formula = TargetClass ~ ., 
                 family = binomial, 
                 data = training_set)

# Predicting probabilities on the test set
prob_pred <- predict(classifier, type = 'response', newdata = test_set[-ncol(test_set)])

# Converting probabilities to strict 1/0 classifications based on a 0.5 threshold
y_pred <- ifelse(prob_pred > 0.5, 1, 0)

# Generating the Confusion Matrix
cm <- table(test_set[, ncol(test_set)], y_pred)
```