import marimo

# Linear Regression - Implementation and Examples

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    mo.md("# Linear Regression - Implementation and Examples")
    return mo


@app.cell
def _():
    # ! pip install numpy
    return


@app.cell
def _():
    import numpy as np
    return np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## LinearRegression Implementation
        Implementation of Linear Regression with gradient descent optimization.

        This class implements a linear regression model trained using gradient descent, with options for L1 or L2 regularization.

        Attributes:
        - weights: ndarray, coefficients for each feature.
        - bias: float, intercept term.
        - learning_rate: float, step size for gradient descent.
        - iterations: int, number of training iterations.
        - regularization: str, regularization type ('l1', 'l2', or None).
        - lambda_param: float, regularization strength.
        - cost_history: list, stores cost at each iteration.

        Functions:
        - init(self, learning_rate=0.01, iterations=1000, regularization='l2', lambda_param=0.1)
        - fit(self, X, y) -> self
        - predict(self, X) -> y_predictions (n_features,)
        - calculate_cost(self, X, y, predictions=None) -> mse
        """
    )
    return


@app.cell
def _():
    # class LinearRegression:
    #     """Implementation of Linear Regression with gradient descent optimization.

    #     This class implements a linear regression model trained using gradient descent,
    #     with options for L1 or L2 regularization and adaptive learning rate.

    #     Attributes:
    #         weights: ndarray, coefficients for each feature.
    #         bias: float, intercept term.
    #         learning_rate: float, initial step size for gradient descent.
    #         max_iter: int, maximum number of iterations.
    #         tol: float, convergence tolerance.
    #         patience: int, number of iterations with no improvement before early stopping.
    #         regularization: str, regularization type ('l1', 'l2', or None).
    #         lambda_param: float, regularization strength.
    #         cost_history: list, stores cost at each iteration.
    #         n_iter_: int, number of iterations run during fitting.
    #         center_data: bool, whether to center the data before fitting.
    #         X_mean_: ndarray, mean of the training features (if centering).
    #         y_mean_: float, mean of the training targets (if centering).
    #         adaptive_learning: bool, whether to use adaptive learning rate.
    #         min_lr: float, minimum learning rate when using adaptive learning.
    #         lr_decay_rate: float, decay rate for learning rate.
    #     """

    #     def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001, 
    #                  regularization='l2', lambda_param=0.1, center_data=True,
    #                  patience=5, adaptive_learning=True, min_lr=1e-6, 
    #                  lr_decay_rate=0.1):
    #         """Initializes the LinearRegression model.

    #         Args:
    #             learning_rate: float, initial step size for gradient descent.
    #             max_iter: int, maximum number of iterations.
    #             tol: float, convergence tolerance.
    #             regularization: str, regularization type ('l1', 'l2', or None).
    #             lambda_param: float, regularization strength.
    #             center_data: bool, whether to center the data before fitting.
    #             patience: int, number of iterations with minimal improvement before stopping.
    #             adaptive_learning: bool, whether to decrease learning rate when progress slows.
    #             min_lr: float, minimum learning rate (prevents too small steps).
    #             lr_decay_rate: float, decay rate for learning rate.
    #         """
    #         self.weights = None
    #         self.bias = None
    #         self.learning_rate = learning_rate
    #         self.max_iter = max_iter
    #         self.tol = tol
    #         self.patience = patience
    #         self.regularization = regularization
    #         self.lambda_param = lambda_param
    #         self.cost_history = []
    #         self.n_iter_ = None
    #         self.center_data = center_data
    #         self.X_mean_ = None
    #         self.y_mean_ = None
    #         self.adaptive_learning = adaptive_learning
    #         self.min_lr = min_lr
    #         self.lr_decay_rate = lr_decay_rate

    #     def _center_data(self, X, y):
    #         """Centers the data by subtracting the mean.

    #         This improves numerical stability during gradient descent.

    #         Args:
    #             X: ndarray of shape (n_samples, n_features), input samples.
    #             y: ndarray of shape (n_samples,), target values.

    #         Returns:
    #             tuple: (X_centered, y_centered) - centered data
    #         """
    #         # Store means for later prediction adjustments
    #         self.X_mean_ = np.mean(X, axis=0)
    #         self.y_mean_ = np.mean(y)

    #         # Center data
    #         X_centered = X - self.X_mean_
    #         y_centered = y - self.y_mean_

    #         return X_centered, y_centered

    #     def _calculate_learning_rate(self, initial_lr, iteration):
    #         """
    #         Calculate decaying learning rate using inverse time decay.

    #         Args:
    #             initial_lr (float): Initial learning rate
    #             iteration (int): Current iteration

    #         Returns:
    #             float: Adjusted learning rate
    #         """
    #         return initial_lr / (1 + self.lr_decay_rate * iteration)

    #     def _check_convergence(self, prev_params, current_params, prev_cost, current_cost):
    #         """
    #         Check whether the optimization algorithm has converged based on parameter 
    #         and cost changes.

    #         Args:
    #             prev_params (np.ndarray): Parameters from the previous iteration.
    #             current_params (np.ndarray): Parameters from the current iteration.
    #             prev_cost (float): Cost from the previous iteration.
    #             current_cost (float): Cost from the current iteration.

    #         Returns:
    #             bool: True if the change in parameters or cost is below the tolerance threshold, 
    #                   indicating convergence; False otherwise.
    #         """
    #         # Check parameter change
    #         param_change = np.max(np.abs(current_params - prev_params))

    #         # Check cost change
    #         cost_change = np.abs(current_cost - prev_cost)

    #         return (param_change < self.tol or 
    #                 cost_change < self.tol)

    #     def fit(self, X, y):
    #         """Trains the model using gradient descent.

    #         Initializes weights and bias, then updates them using gradient descent
    #         to minimize the cost function. If center_data=True, the data is centered
    #         first for improved numerical stability.

    #         The method uses improved convergence criteria with a patience parameter
    #         to handle oscillating solutions, and can adapt the learning rate when
    #         progress slows.

    #         Args:
    #             X: ndarray of shape (n_samples, n_features), training input samples.
    #             y: ndarray of shape (n_samples,), target values.

    #         Returns:
    #             self: returns an instance of self.
    #         """
    #         # Center data if requested for better numerical stability
    #         if self.center_data:
    #             X_train, y_train = self._center_data(X, y)
    #         else:
    #             X_train, y_train = X, y

    #         n_samples, n_features = X_train.shape

    #         self.weights = np.zeros(n_features)
    #         self.bias = 0

    #         # Calculate initial cost
    #         predictions = self.predict(X_train)
    #         prev_cost = self.calculate_cost(X_train, y_train, predictions)
    #         self.cost_history.append(prev_cost)

    #         # Initialize parameters
    #         prev_params = np.concatenate([self.weights, [self.bias]])

    #         # Initialize variables for improved convergence criteria
    #         best_cost = float('inf')
    #         patience_counter = 0

    #         for i in range(self.max_iter):
    #             # Calculate current learning rate (if adaptive learning is on)
    #             current_lr = (self.learning_rate if not self.adaptive_learning 
    #                           else self._calculate_learning_rate(self.learning_rate, i))

    #             # calculate predictions
    #             predictions = self.predict(X_train)

    #             # calculate gradients
    #             dw = (1 / n_samples) * np.dot(X_train.T, predictions - y_train)
    #             db = (1 / n_samples) * np.sum(predictions - y_train)

    #             # Add regularization
    #             if self.regularization == 'l2':
    #                 dw += (self.lambda_param / n_samples) * self.weights
    #             if self.regularization == 'l1':
    #                 dw += (self.lambda_param / n_samples) * np.sign(self.weights)

    #             # update params
    #             self.weights -= current_lr * dw
    #             self.bias -= current_lr * db

    #             # calculate cost    
    #             current_cost = self.calculate_cost(X, y, predictions)
    #             self.cost_history.append(current_cost)

    #             # Prepare parameters for convergence check
    #             current_params = np.concatenate([self.weights, [self.bias]])

    #             # Check for early stopping based on patience
    #             if current_cost < best_cost:
    #                 best_cost = current_cost
    #                 patience_counter = 0
    #             else:
    #                 patience_counter += 1

    #             if patience_counter >= self.patience:
    #                 print(f"Early stopping: No improvement for {self.patience} iterations")
    #                 break

    #             # Check for convergence
    #             if self._check_convergence(prev_params, current_params, prev_cost, current_cost):
    #                 print("Converged based on tolerance")
    #                 break

    #             prev_params = current_params
    #             prev_cost = current_cost

    #         # Store the actual number of iterations
    #         self.n_iter_ = i + 1

    #         # If we completed all iterations without converging
    #         if i == self.max_iter - 1:
    #             print(f"Maximum iterations ({self.max_iter}) reached without convergence.")

    #         # If data was centered, adjust bias to account for centering
    #         if self.center_data:
    #             self.bias = self.y_mean_ - np.dot(self.X_mean_, self.weights)

    #         return self

    #     def predict(self, X):
    #         """Makes predictions using the trained model.

    #         This method works correctly regardless of whether data was centered
    #         during training, thanks to bias adjustment performed at the end of training.

    #         Args:
    #             X: ndarray of shape (n_samples, n_features), input samples.

    #         Returns:
    #             ndarray of shape (n_samples,), predicted values.
    #         """
    #         return np.dot(X, self.weights) + self.bias

    #     def calculate_cost(self, X, y, predictions=None):
    #         """Calculates the cost function with regularization.

    #         Args:
    #             X: ndarray of shape (n_samples, n_features), input samples.
    #             y: ndarray of shape (n_samples,), target values.
    #             predictions: ndarray of shape (n_samples,), predicted values (optional).

    #         Returns:
    #             float: Mean squared error with regularization.
    #         """
    #         if predictions is None:
    #             predictions = self.predict(X)

    #         n_samples = X.shape[0]

    #         mse = np.sum((predictions-y)**2) / n_samples

    #         if self.regularization == 'l1':
    #             mse += (self.lambda_param / n_samples) * np.sum(np.abs(self.weights))
    #         if self.regularization == 'l2':
    #             mse += (self.lambda_param / (2*n_samples)) * np.sum(self.weights**2)

    #         return mse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Test LinearRegression with California Dataset""")
    return


@app.cell
def _():
    # ! pip install pandas sklearn matplotlib
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    # Import the LinearRegression class
    from regressors import LinearRegression
    return (
        LinearRegression,
        StandardScaler,
        fetch_california_housing,
        mean_squared_error,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(fetch_california_housing):
    # Load the California Housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    return X, california, y


@app.cell
def _(X, california, y):
    # Print information about the dataset
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {california.feature_names}")
    print(f"Target: Median house value (in $100,000)")
    return


@app.cell
def _(StandardScaler, X, train_test_split, y):
    # Scale the features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled, X_test, X_train, scaler, y_test, y_train


@app.cell
def _(LinearRegression, X_test, X_train, np, y_test, y_train):
    # Define your parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_iter': [100, 1000],
        'tol': [0.0001, 1e-6],
        'lambda_param': [0.001, 0.01, 0.1]
    }

    # Create a base model
    model = LinearRegression()

    # Run grid search
    results = model.grid_search(X_train, y_train, param_grid, cv=5, scoring='rmse')


    # Print results
    print("Best parameters found:")
    print(results['best_params'])
    print(f"\nBest RMSE score: {results['best_score']:.4f}")

    # Create and fit final model with best parameters
    final_model = LinearRegression(**results['best_params'])
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"\nTest RMSE: {test_rmse:.4f}")
    return final_model, model, param_grid, results, test_rmse, y_pred


@app.cell
def _(LinearRegression, X_test, X_train, plt, y_pred, y_test, y_train):
    model_1 = LinearRegression(learning_rate=0.1, max_iter=2000, tol=1e-06, regularization='l2', lambda_param=0.001, patience=5, adaptive_learning=True)
    model_1.fit(X_train, y_train)
    mse = model_1.score(X_test, y_test, metric='mse')
    rmse = model_1.score(X_test, y_test, metric='rmse')
    r2 = model_1.score(X_test, y_test, metric='r2')
    print(f'\nModel Evaluation:')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    plt.figure(figsize=(5, 3))
    plt.plot(model_1.cost_history)
    plt.title('Cost History During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(5, 3))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    plt.show()
    return model_1, mse, r2, rmse


@app.cell
def _(
    X_test,
    X_train,
    california,
    mean_squared_error,
    model_1,
    mse,
    np,
    r2,
    r2_score,
    rmse,
    y_test,
    y_train,
):
    from sklearn.linear_model import Ridge as SklearnRidge
    sklearn_ridge = SklearnRidge(alpha=model_1.lambda_param)
    sklearn_ridge.fit(X_train, y_train)
    ridge_pred = sklearn_ridge.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(ridge_mse)
    ridge_r2 = r2_score(y_test, ridge_pred)
    print('\nYour Model:')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    print('\nScikit-learn Ridge Regression:')
    print(f'Mean Squared Error: {ridge_mse:.4f}')
    print(f'Root Mean Squared Error: {ridge_rmse:.4f}')
    print(f'R² Score: {ridge_r2:.4f}')
    print('\nModel Coefficients Comparison:')
    print('Feature\t\tYour Model\t\tRidge Reg')
    print('-' * 70)
    for (i, feature) in enumerate(california.feature_names):
        print(f'{feature}\t\t{model_1.coef_[i]:.4f}\t\t{sklearn_ridge.coef_[i]:.4f}')
    print(f'Bias\t\t\t{model_1.intercept_:.4f}\t\t{sklearn_ridge.intercept_:.4f}')
    print(f'\nYour model iterations: {model_1.n_iter_}')
    return (
        SklearnRidge,
        feature,
        i,
        ridge_mse,
        ridge_pred,
        ridge_r2,
        ridge_rmse,
        sklearn_ridge,
    )


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
