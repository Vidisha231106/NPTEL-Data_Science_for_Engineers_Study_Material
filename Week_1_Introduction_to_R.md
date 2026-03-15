# Week 1: Introduction to R Programming

## 1. Course Philosophy and the Data Science Workflow
Data Science is the practice of extracting insight and knowledge from structured and unstructured data. In the context of engineering, this involves a rigorous workflow:
1. **Problem Definition**: Translating a business or engineering problem into a data science problem.
2. **Data Ingestion**: Collecting data from databases, APIs, or flat files (like CSVs).
3. **Data Cleaning & Munging (Wrangling)**: Handling missing values, smoothing noisy data, identifying or removing outliers, and resolving inconsistencies.
4. **Exploratory Data Analysis (EDA)**: Visualizing data, finding patterns, and understanding distributions to formulate hypotheses.
5. **Feature Engineering**: Creating new variables from existing ones that better represent the underlying problem to the predictive models.
6. **Predictive Modeling**: Applying statistical and machine learning algorithms (e.g., Regression, Classification, Clustering).
7. **Model Assessment & Selection**: Using metrics (R-squared, Accuracy, etc.) and validation techniques (Cross-Validation) to choose the best model.
8. **Deployment & Communication**: Implementing the model in a production environment and communicating the results to stakeholders.

R is a language built *by* statisticians *for* statisticians, making it incredibly powerful for EDA, statistical modeling, and visualization.

## 2. Basics of R: Variables and Data Types
In R, you don't need to declare the data type of a variable before using it; R dynamically types variables based on the value assigned to them. The assignment operator is `<-`.

### Fundamental Data Types (Atomic Classes)
*   **Numeric**: The default type for numbers. Includes floating-point (decimals).
    ```r
    x <- 10.5
    class(x) # Returns "numeric"
    ```
*   **Integer**: A whole number. Must explicitly append an `L` to force it to be an integer.
    ```r
    y <- 42L
    class(y) # Returns "integer"
    ```
*   **Character**: Text or string values. Enclosed in single or double quotes.
    ```r
    name <- "Data Science"
    class(name) # Returns "character"
    ```
*   **Logical**: Boolean values, returning `TRUE` or `FALSE` (can be abbreviated as `T` or `F`, but full words are preferred).
    ```r
    is_valid <- TRUE
    class(is_valid) # Returns "logical"
    ```
*   **Complex**: Used for complex numbers with a real and imaginary part.
    ```r
    z <- 3 + 2i
    class(z) # Returns "complex"
    ```

## 3. Data Structures in R
Data structures are used to store multiple values.

### 3.1 Vectors
A vector is a one-dimensional array that can hold data of only **one specific atomic type** (e.g., all numeric, or all character). If you mix types, R will *coerce* them into a single type (e.g., character > numeric > logical).
*   **Creation**: Using the combine function `c()`.
    ```r
    num_vec <- c(1, 2, 3, 4, 5)
    char_vec <- c("A", "B", "C")
    ```
*   **Vectorized Operations**: Operations on vectors happen element-by-element.
    ```r
    v1 <- c(1, 2, 3)
    v2 <- c(4, 5, 6)
    v1 + v2 # Returns c(5, 7, 9)
    v1 * 2  # Returns c(2, 4, 6)
    ```

### 3.2 Matrices
A two-dimensional, rectangular arrangement of data of the **same basic type**.
*   **Creation**: Using the `matrix()` function.
    ```r
    # Create a 3x3 matrix from values 1 to 9, filling by column (default)
    mat <- matrix(1:9, nrow = 3, ncol = 3)
    # Fill by row
    mat_row <- matrix(1:9, nrow=3, byrow=TRUE)
    ```
*   **Accessing Elements**: `mat[row, col]`. e.g., `mat[1, 2]` gets the element in row 1, column 2. `mat[1, ]` gets the entire first row.

### 3.3 Lists
A one-dimensional, heterogeneous data structure. A list can contain elements of **different types**, including other vectors, matrices, data frames, or even other lists.
*   **Creation**: Using the `list()` function.
    ```r
    my_list <- list(name="Alice", age=25, scores=c(90, 85, 95))
    ```
*   **Accessing Elements**: Use `[[ ]]` or `$` to extract the element itself.
    ```r
    my_list$name      # Returns "Alice"
    my_list[[3]][2]   # Returns 85 (2nd element of the 3rd list item)
    ```

### 3.4 Data Frames
The most critical data structure in R for data science. It is a two-dimensional, tabular data structure where each column can contain a **different data type** (e.g., column 1 is numeric, column 2 is character), but within a single column, all elements must be of the same type. It is essentially a list of equal-length vectors.
*   **Creation**:
    ```r
    df <- data.frame(
      ID = 1:3,
      Name = c("John", "Jane", "Bob"),
      Passed = c(TRUE, TRUE, FALSE)
    )
    ```
*   **Accessing**: `df$Name` or `df[ , "Name"]`.

## 4. Data Manipulation
Real-world data is rarely ready for analysis out of the box.

### 4.1 Reshaping Data (The `reshape2` package)
Data can be in "wide" format or "long" format. Tools like `reshape2` or newer packages like `tidyr` help switch between these.
*   **`melt()`**: Converts data from wide to long format. It takes multiple columns and collapses them into key-value pairs (e.g., Variable and Value columns).
*   **`dcast()`**: Converts data from long to wide format (casting a molten data frame).

### 4.2 Joining Data Frames
Combining data from multiple tables based on a common key (like SQL joins). Base R uses `merge()`. The `dplyr` package is highly recommended for this.
*   **Inner Join**: Returns rows where the key exists in *both* tables. `merge(df1, df2, by="ID")`
*   **Left Join**: Returns all rows from the left table, and matching rows from the right table. `merge(df1, df2, by="ID", all.x=TRUE)`

## 5. Advanced Programming in R

### 5.1 User-Defined Functions
Functions encapsulate code to perform specific tasks, making code modular, readable, and reusable.
*   **Syntax**:
    ```r
    calculate_mean <- function(x) {
      # Check if x is numeric
      if(!is.numeric(x)) {
        stop("Input must be numeric")
      }
      total <- sum(x)
      count <- length(x)
      return(total / count)
    }
    ```

### 5.2 Control Structures
Allow you to control the flow of execution in your scripts.
*   **`if-else` Statements**: Execute code conditionally.
    ```r
    score <- 85
    if (score >= 90) {
      print("Grade A")
    } else if (score >= 80) {
      print("Grade B")
    } else {
      print("Grade C")
    }
    ```
*   **`for` Loops**: Iterate over a sequence (like a vector or list).
    ```r
    for (i in 1:5) {
      print(i^2)
    }
    ```
*   **`while` Loops**: Execute code as long as a condition evaluates to TRUE. Be careful to avoid infinite loops.
    ```r
    count <- 1
    while (count <= 3) {
      print(count)
      count <- count + 1
    }
    ```

## 6. Data Visualization in R
Visualizing data helps identify trends, outliers, and patterns.

### Base R Graphics
R has built-in functions for quick plotting.
*   **Scatter Plot**: `plot(x, y)` - visualizes the relationship between two continuous variables.
*   **Histogram**: `hist(x)` - shows the frequency distribution of a single continuous variable.
*   **Boxplot**: `boxplot(x ~ group)` - displays the five-number summary (minimum, first quartile, median, third quartile, and maximum) and helps identify outliers.
*   **Barplot**: `barplot(heights)` - visualizes categorical data.

### Introduction to `ggplot2` (Bonus context for modern R)
While base R is great, `ggplot2` is the industry standard for production-quality graphics in R. It uses a "grammar of graphics" approach:
```r
# library(ggplot2)
# ggplot(data = df, aes(x = Weight, y = Height, color = Gender)) + 
#   geom_point() + 
#   labs(title = "Height vs Weight by Gender")
```
This builds a plot layer by layer: data, aesthetics (mappings), and geometries (points, lines, etc.).