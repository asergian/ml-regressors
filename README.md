# Regressors

A custom implementation of linear and logistic regression models with gradient descent optimization.

## Features

### Common Features
- Gradient descent optimization with adaptive learning rate
- Early stopping based on patience
- Convergence checking
- Cost history tracking
- L1 and L2 regularization
- Grid search for hyperparameter tuning

### Model-Specific Features
- Linear Regression
  - Mean squared error loss
  - Regularized coefficients
  - Compatible with scikit-learn's preprocessing tools
- Logistic Regression
  - Binary cross-entropy loss for binary classification
  - Multi-class classification with softmax activation
  - Probability predictions
  - Custom decision thresholds (for binary classification)
  - Multiple evaluation metrics (accuracy, precision, recall, F1 for binary classification)

## Package Structure

The package is organized into three main components:

1. **Base Classes**
   - `BaseRegression`: Abstract base class for all regression models
   - `BaseOptimizer`: Abstract base class for optimization algorithms

2. **Models**
   - `LinearRegression`: Linear regression model
   - `LogisticRegression`: Logistic regression model for binary and multi-class classification
   - Each model handles its own:
     - Parameter initialization
     - Gradient calculation
     - Prediction logic
     - Cost function

3. **Optimizers**
   - `GradientDescentOptimizer`: Gradient descent implementation
   - Handles:
     - Learning rate adaptation
     - Convergence checking
     - Early stopping
     - Cost history tracking

## Usage

### Linear Regression

```python
from regressors.models.linear_regression import LinearRegression
from sklearn.preprocessing import StandardScaler

# Scale the features (recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = LinearRegression(
    learning_rate=0.01,
    max_iter=1000,
    tol=0.0001,
    regularization='l2',
    lambda_param=0.1
)

# Fit model
model.fit(X_scaled, y)

# Make predictions
predictions = model.predict(scaler.transform(X_test))

# Get model parameters
coefficients = model.coef_
intercept = model.intercept_

# Get optimization history
n_iterations = model.n_iter_
cost_history = model.cost_history
```

### Binary Logistic Regression

```python
from regressors.models.logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features (recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = LogisticRegression(
    learning_rate=0.01,
    max_iter=1000,
    tol=0.0001,
    regularization='l2',
    lambda_param=0.1
)

# Fit model for binary classification
model.fit(X_scaled, y)  # y should contain two classes (e.g., 0 and 1)

# Make probability predictions
probabilities = model.predict_proba(scaler.transform(X_test))

# Make class predictions (with custom threshold)
predictions = model.predict(scaler.transform(X_test), threshold=0.5)

# Evaluate model
accuracy = model.score(X_test, y_test, metric='accuracy')
precision = model.score(X_test, y_test, metric='precision')
recall = model.score(X_test, y_test, metric='recall')
f1 = model.score(X_test, y_test, metric='f1')
```

### Multi-class Logistic Regression

```python
from regressors.models.logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features (recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = LogisticRegression(
    learning_rate=0.01,
    max_iter=1000,
    tol=0.0001,
    regularization='l2',
    lambda_param=0.1
)

# Fit model for multi-class classification
model.fit(X_scaled, y)  # y can contain any number of classes (e.g., 0, 1, 2, ...)

# Make probability predictions (returns probabilities for each class)
probabilities = model.predict_proba(scaler.transform(X_test))
# Shape: (n_samples, n_classes)

# Make class predictions (returns the most likely class for each sample)
predictions = model.predict(scaler.transform(X_test))

# Evaluate model (only accuracy is supported for multi-class)
accuracy = model.score(X_test, y_test, metric='accuracy')
```

### Grid Search

```python
# Define parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'lambda_param': [0.1, 1.0],
    'regularization': ['l1', 'l2']
}

# Perform grid search
results = model.grid_search(
    X_scaled, y,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'  # For logistic regression: 'accuracy', 'precision', 'recall', 'f1'
)

# Get best parameters
best_params = results['best_params']
best_score = results['best_score']
```

## Parameters

### Common Parameters
- `learning_rate`: float, default=0.01
  - Initial step size for gradient descent
- `max_iter`: int, default=1000
  - Maximum number of iterations
- `tol`: float, default=0.0001
  - Convergence tolerance
- `regularization`: str, default='l2'
  - Type of regularization ('l1', 'l2', or None)
- `lambda_param`: float, default=0.1
  - Regularization strength
- `patience`: int, default=5
  - Number of iterations with no improvement before early stopping
- `adaptive_learning`: bool, default=True
  - Whether to use adaptive learning rate
- `min_lr`: float, default=1e-6
  - Minimum learning rate when using adaptive learning
- `lr_decay_rate`: float, default=0.1
  - Decay rate for learning rate

### Grid Search Parameters
- `cv`: int, default=5
  - Number of cross-validation folds
- `scoring`: str
  - Linear Regression: 'rmse' or 'mse'
  - Logistic Regression: 
    - Binary classification: 'accuracy', 'precision', 'recall', or 'f1'
    - Multi-class classification: 'accuracy'

## How It Works

### Binary vs Multi-class Mode

The `LogisticRegression` class automatically detects whether it's being used for binary or multi-class classification:

- **Binary mode**: When the target variable contains exactly 2 unique classes
  - Uses sigmoid activation function
  - Supports precision, recall, and F1 evaluation metrics
  - Allows custom decision threshold

- **Multi-class mode**: When the target variable contains 3 or more unique classes
  - Uses softmax activation function
  - Supports only accuracy as evaluation metric
  - Returns probability distribution across all classes

## Installation

```bash
pip install regressors
```

## Requirements

- numpy>=1.20.0
- scikit-learn>=0.24.0 (for data preprocessing)

## License

MIT