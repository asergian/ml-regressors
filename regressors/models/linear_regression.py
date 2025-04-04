import numpy as np
from itertools import product
from ..base.base_model import BaseRegression
from ..optimizers.gradient_descent import GradientDescentOptimizer

class LinearRegression(BaseRegression):
    """Linear regression model with gradient descent optimization.
    
    This class implements a linear regression model trained using gradient descent,
    with options for L1 or L2 regularization and adaptive learning rate.
    
    Attributes:
        coef_: ndarray, coefficients for each feature.
        intercept_: float, intercept term.
        regularization: str, regularization type ('l1', 'l2', or None).
        lambda_param: float, regularization strength.
        n_iter_: int, number of iterations run during optimization.
        cost_history: list, stores cost at each iteration.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001,
                 regularization='l2', lambda_param=0.1,
                 patience=5, adaptive_learning=True, min_lr=1e-6,
                 lr_decay_rate=0.1):
        """Initialize the linear regression model.
        
        Args:
            learning_rate: float, initial step size for gradient descent.
            max_iter: int, maximum number of iterations.
            tol: float, convergence tolerance.
            regularization: str, regularization type ('l1', 'l2', or None).
            lambda_param: float, regularization strength.
            patience: int, number of iterations with no improvement before early stopping.
            adaptive_learning: bool, whether to use adaptive learning rate.
            min_lr: float, minimum learning rate when using adaptive learning.
            lr_decay_rate: float, decay rate for learning rate.
        """
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            regularization=regularization,
            lambda_param=lambda_param,
            patience=patience,
            adaptive_learning=adaptive_learning,
            min_lr=min_lr,
            lr_decay_rate=lr_decay_rate
        )
        
        # Initialize optimizer
        self._optimizer = GradientDescentOptimizer(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            patience=patience,
            adaptive_learning=adaptive_learning,
            min_lr=min_lr,
            lr_decay_rate=lr_decay_rate
        )

    @property
    def n_iter_(self):
        """Get the number of iterations run during optimization."""
        return self._optimizer.n_iter_
        
    @property
    def cost_history(self):
        """Get the cost history from optimization."""
        return self._optimizer.cost_history

    def _calculate_gradients(self, X, y, n_samples):
        """Calculate gradients for model parameters.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Training data.
            y: array-like, shape (n_samples,)
                Target values.
            n_samples: int
                Number of training samples.
                
        Returns:
            dict: Dictionary of gradients for each parameter.
        """
        # Calculate predictions
        predictions = np.dot(X, self.coef_) + self.intercept_
        error = predictions - y
        
        # Calculate gradients
        dw = (2 / n_samples) * np.dot(X.T, error)
        db = (2 / n_samples) * np.sum(error)
        
        # Add regularization to gradient
        dw += self._calculate_regularization_gradient(n_samples)
            
        return {
            'coef_': dw,
            'intercept_': db
        }

    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
                
        Returns:
            array-like, shape (n_samples,)
                Predicted values.
        """
        if self.coef_ is None:
            return np.zeros(len(X))
            
        return np.dot(X, self.coef_) + self.intercept_

    def calculate_cost(self, X, y, predictions=None):
        """Calculate the mean squared error with regularization.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            y: array-like, shape (n_samples,)
                Target values.
            predictions: array-like, shape (n_samples,), optional
                Pre-computed predictions.
                
        Returns:
            float: Mean squared error with regularization.
        """
        if predictions is None:
            predictions = self.predict(X)
        
        n_samples = len(X)
        
        # Calculate mean squared error
        mse = np.mean((predictions - y) ** 2)
        
        # Add regularization term
        mse += self._calculate_regularization_cost(n_samples)
        
        return mse

    def fit(self, X, y):
        """Fit the model to the training data.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Training data.
            y: array-like, shape (n_samples,)
                Target values.
                
        Returns:
            self: Returns the instance itself.
        
        Raises:
            ValueError: If X or y is empty, contains NaN/inf values, or have incompatible shapes.
        """
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty dataset provided")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input contains infinite values")
        
        # Initialize parameters and optimize
        self._initialize_parameters(X.shape[1])
        self._optimizer.optimize(self, X, y)
        
        return self

    def score(self, X, y, metric='mse'):
        """Calculate specified metric between predictions and true values.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            y: array-like, shape (n_samples,)
                Target values.
            metric: str, optional (default='mse')
                Metric to calculate: 'mse', 'rmse', or 'r2'.
                
        Returns:
            float: Calculated metric value.
        """
        predictions = self.predict(X)
        if metric == 'mse':
            return np.mean((predictions - y) ** 2)
        elif metric == 'rmse':
            return np.sqrt(np.mean((predictions - y) ** 2))
        elif metric == 'r2':
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - predictions) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError("metric must be 'mse', 'rmse', or 'r2'")

    def grid_search(self, X, y, param_grid, cv=5, scoring='rmse'):
        """Perform grid search over specified parameter values.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Training data.
            y: array-like, shape (n_samples,)
                Target values.
            param_grid: dict
                Parameter names mapped to lists of values to try.
            cv: int, optional (default=5)
                Number of cross-validation folds.
            scoring: str, optional (default='rmse')
                Scoring metric to use ('rmse' or 'mse').
                
        Returns:
            dict: Dictionary containing:
                - 'best_params': dict, best parameter combination found
                - 'best_score': float, best score achieved
        """
        # Split data into folds
        n_samples = len(X)
        fold_size = n_samples // cv
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Get default parameters from current instance
        default_params = {
            'learning_rate': self.learning_rate,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'regularization': self.regularization,
            'lambda_param': self.lambda_param,
            'patience': self.patience,
            'adaptive_learning': self.adaptive_learning,
            'min_lr': self.min_lr,
            'lr_decay_rate': self.lr_decay_rate
        }
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        best_score = float('inf')
        best_params = None
        
        # Evaluate each parameter combination
        for params in param_combinations:
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))
            
            # Update default parameters with current combination
            current_params = default_params.copy()
            current_params.update(param_dict)
            
            # Initialize model with current parameters
            model = LinearRegression(**current_params)
            
            # Perform cross-validation
            scores = []
            for i in range(cv):
                # Split data into training and validation sets
                val_start = i * fold_size
                val_end = (i + 1) * fold_size
                val_indices = indices[val_start:val_end]
                train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
                
                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]
                
                # Fit model and calculate score
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                
                if scoring == 'rmse':
                    score = np.sqrt(np.mean((predictions - y_val) ** 2))
                else:  # mse
                    score = np.mean((predictions - y_val) ** 2)
                    
                scores.append(score)
            
            # Calculate average score
            avg_score = np.mean(scores)
            
            # Update best parameters if current combination is better
            if avg_score < best_score:
                best_score = avg_score
                best_params = current_params
        
        return {'best_params': best_params, 'best_score': best_score} 