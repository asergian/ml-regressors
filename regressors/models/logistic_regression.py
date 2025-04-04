import numpy as np
from ..base.base_model import BaseRegression
from ..optimizers.gradient_descent import GradientDescentOptimizer
from itertools import product

class LogisticRegression(BaseRegression):
    """Logistic regression model with gradient descent optimization.
    
    This class implements logistic regression for both binary and multi-class (softmax)
    classification, with options for L1 or L2 regularization and adaptive learning rate.
    
    Attributes:
        coef_: ndarray, coefficients for each feature (and each class for multi-class).
        intercept_: float or ndarray, intercept term(s).
        classes_: ndarray, unique class labels.
        n_classes_: int, number of classes.
        regularization: str, regularization type ('l1', 'l2', or None).
        lambda_param: float, regularization strength.
        n_iter_: int, number of iterations run during optimization.
        cost_history: list, stores cost at each iteration.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001,
                 regularization='l2', lambda_param=0.1,
                 patience=5, adaptive_learning=True, min_lr=1e-6,
                 lr_decay_rate=0.1):
        """Initialize the logistic regression model.
        
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
        
        # Class attributes
        self.classes_ = None
        self.n_classes_ = None
        self._binary_mode = None

    @property
    def n_iter_(self):
        """Get the number of iterations run during optimization."""
        return self._optimizer.n_iter_
        
    @property
    def cost_history(self):
        """Get the cost history from optimization."""
        return self._optimizer.cost_history

    def _sigmoid(self, z):
        """Apply the sigmoid function with improved numerical stability.
        
        This implementation handles large positive and negative values separately
        to maintain numerical stability and avoid overflow.
        
        Args:
            z: array-like
                Input values.
                
        Returns:
            array-like: Sigmoid of input values.
        """
        # Clip values for numerical stability
        z = np.clip(z, -250, 250)
        
        # Handle positive and negative values separately for numerical stability
        mask = z < 0
        result = np.zeros_like(z, dtype=np.float64)
        
        # For negative values: exp(x) / (1 + exp(x))
        result[mask] = np.exp(z[mask]) / (1 + np.exp(z[mask]))
        # For positive values: 1 / (1 + exp(-x))
        result[~mask] = 1 / (1 + np.exp(-z[~mask]))
        
        return result
    
    def _softmax(self, z):
        """Apply softmax function with improved numerical stability.
        
        Args:
            z: array-like, shape (n_samples, n_classes)
                Input values.
                
        Returns:
            array-like, shape (n_samples, n_classes): Softmax probabilities.
        """
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _initialize_parameters(self, n_features):
        """Initialize model parameters based on the number of classes.
        
        Args:
            n_features: int
                Number of features in the input data.
        """

        # Initialize with small random values instead of zeros
        # Using He initialization scaled down to avoid saturation
        if self._binary_mode:
            # For binary classification, we only need one vector of coefficients
            self.coef_ = np.random.randn(n_features) * np.sqrt(2.0 / n_features) * 0.01
            self.intercept_ = np.float64(0.0)
        else:
            # For multi-class classification, we need one vector of coefficients per class
            self.coef_ = np.random.randn(n_features, self.n_classes_) * np.sqrt(2.0 / n_features) * 0.01
            self.intercept_ = np.random.randn(self.n_classes_) * 0.01

    def _calculate_gradients(self, X, y, n_samples):
        """Calculate gradients for model parameters with improved numerical stability.
        
        This implementation includes:
        - Better handling of numerical precision
        - Proper scaling of regularization
        - Protection against extreme probability values
        
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
        z = np.dot(X, self.coef_) + self.intercept_
        
        # Convert y to class indices
        y_indices = np.searchsorted(self.classes_, y)

        # Calculate error
        if self._binary_mode:
            predictions = self._sigmoid(z)
            # Add small epsilon to avoid taking log of 0
            eps = 1e-15
            predictions = np.clip(predictions, eps, 1 - eps)
            error = predictions - y_indices
        else:
            predictions = self._softmax(z)
            # Convert y to one-hot encoding
            y_one_hot = np.zeros((n_samples, self.n_classes_))
            y_one_hot[np.arange(n_samples), y_indices] = 1
            error = predictions - y_one_hot

        # Calculate gradients with proper scaling
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error, axis=0)
        
        # Add properly scaled regularization to gradient
        if self.regularization == 'l2':
            dw += (self.lambda_param / n_samples) * self.coef_
        elif self.regularization == 'l1':
            dw += (self.lambda_param / n_samples) * np.sign(self.coef_)
            
        return {
            'coef_': dw,
            'intercept_': db
        }

    def predict_proba(self, X):
        """Predict class probabilities.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
                
        Returns:
            array-like, shape (n_samples, n_classes)
                Predicted probabilities for each class.
        """
        if self.coef_ is None:
            return np.zeros((len(X), self.n_classes_ if self.n_classes_ else 2))
            
        z = np.dot(X, self.coef_) + self.intercept_

        if self._binary_mode:
            prob_class_1 = self._sigmoid(z)
            return np.column_stack((1 - prob_class_1, prob_class_1))
        else:
            return self._softmax(z)

    def predict(self, X, threshold=0.5):
        """Predict class labels.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            threshold: float, optional (default=0.5)
                Decision threshold for binary classification.
                
        Returns:
            array-like, shape (n_samples,)
                Predicted class labels.
        """
        probas = self.predict_proba(X)
        
        if self._binary_mode:
            # For binary classification, use threshold on class 1 probability
            # Get indices (0 or 1) based on threshold
            indices = (probas[:, 1] >= threshold).astype(int)
            # Map indices back to original class labels
            return self.classes_[indices]
        else:
            # For multi-class, get index of highest probability for each sample
            indices = np.argmax(probas, axis=1)
            # Map indices back to original class labels
            return self.classes_[indices]

    def calculate_cost(self, X, y, predictions=None):
        """Calculate the binary cross-entropy loss with regularization.
        
        This implementation includes:
        - Protection against taking log of 0
        - Proper scaling of the regularization term
        - Improved numerical stability
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            y: array-like, shape (n_samples,)
                Target values.
            predictions: array-like, optional
                Pre-computed predictions.
                
        Returns:
            float: Cost value with regularization.
        """
        if predictions is None:
            predictions = self.predict_proba(X)
        
        n_samples = len(X)

        # Convert y to class indices
        y_indices = np.searchsorted(self.classes_, y)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)

        if self._binary_mode:
            # Calculate binary cross-entropy with improved numerical stability
            cost = -(1 / n_samples) * np.sum(
                y_indices * np.log(predictions[:, 1]) + 
                (1 - y_indices) * np.log(predictions[:, 0])
            )
        else:
            # Calculate multi-class cross-entropy with improved numerical stability
            y_one_hot = np.zeros((n_samples, self.n_classes_))
            y_one_hot[np.arange(n_samples), y_indices] = 1
            cost = -(1 / n_samples) * np.sum(y_one_hot * np.log(predictions))
        
        # Add properly scaled regularization term
        cost += self._calculate_regularization_cost(n_samples)
        
        return cost

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
        
        # Check for NaN or infinite values in X
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input features contain NaN or infinite values")
        
        # Check for NaN only in numeric y values
        if np.issubdtype(y.dtype, np.number) and (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            raise ValueError("Target values contain NaN or infinite values")
        
        # Determine number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Determine if this is a binary or multi-class problem
        self._binary_mode = (self.n_classes_ == 2)
        
        # Initialize parameters and optimize
        self._initialize_parameters(X.shape[1])
        self._optimizer.optimize(self, X, y)
        
        return self 

    def score(self, X, y, metric='accuracy'):
        """Calculate specified metric between predictions and true values.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            y: array-like, shape (n_samples,)
                Target values.
            metric: str, optional (default='accuracy')
                Metric to calculate: 'accuracy', 'precision', 'recall', 'f1'.
                
        Returns:
            float: Calculated metric value.
            
        Raises:
            ValueError: If metric is not supported.
        """
        predictions = self.predict(X)
        
        if metric == 'accuracy':
            return np.mean(predictions == y)
        
        if not self._binary_mode and metric in ['precision', 'recall', 'f1']:
            raise ValueError(f"Metric '{metric}' is only supported for binary classification")
        
        # Calculate confusion matrix elements for binary case
        true_pos = np.sum((predictions == self.classes_[1]) & (y == self.classes_[1]))
        false_pos = np.sum((predictions == self.classes_[1]) & (y == self.classes_[0]))
        false_neg = np.sum((predictions == self.classes_[0]) & (y == self.classes_[1]))
        
        if metric == 'precision':
            # Handle division by zero
            if true_pos + false_pos == 0:
                return 0.0
            return true_pos / (true_pos + false_pos)
            
        elif metric == 'recall':
            # Handle division by zero
            if true_pos + false_neg == 0:
                return 0.0
            return true_pos / (true_pos + false_neg)
            
        elif metric == 'f1':
            # Calculate precision and recall
            if true_pos + false_pos == 0:
                precision = 0.0
            else:
                precision = true_pos / (true_pos + false_pos)
                
            if true_pos + false_neg == 0:
                recall = 0.0
            else:
                recall = true_pos / (true_pos + false_neg)
            
            # Handle division by zero
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
            
        else:
            raise ValueError("metric must be 'accuracy', 'precision', 'recall', or 'f1'") 

    def grid_search(self, X, y, param_grid, cv=5, scoring='accuracy'):
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
            scoring: str, optional (default='accuracy')
                Scoring metric to use ('accuracy', 'precision', 'recall', or 'f1').
                
        Returns:
            dict: Dictionary containing:
                - 'best_params': dict, best parameter combination found
                - 'best_score': float, best score achieved
        """
        # Check if scoring metric is compatible with problem type
        if not self._binary_mode and scoring in ['precision', 'recall', 'f1']:
            raise ValueError(f"Scoring metric '{scoring}' is only supported for binary classification")
        
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
        
        # Calculate total number of combinations
        total_combinations = len(param_combinations)
        
        best_score = float('-inf')
        best_params = None
        
        # Try each parameter combination
        for i, params in enumerate(param_combinations):
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))
            current_params = default_params.copy()
            current_params.update(param_dict)
            
            # Print progress
            progress = (i + 1) / total_combinations * 100
            print(f"\rGrid search progress: {progress:.1f}% ({i+1}/{total_combinations})", end="")
            
            # Perform cross-validation
            fold_scores = []
            for j in range(cv):
                # Get validation indices
                val_start = j * fold_size
                val_end = (j + 1) * fold_size if j < cv - 1 else n_samples
                val_indices = indices[val_start:val_end]
                train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
                
                # Split data
                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]
                
                # Train model
                model = LogisticRegression(**current_params)
                model.fit(X_train, y_train)
                
                # Calculate score
                score = model.score(X_val, y_val, metric=scoring)
                fold_scores.append(score)
            
            # Calculate mean score across folds
            mean_score = np.mean(fold_scores)
            
            # Update best parameters if better score found
            if mean_score > best_score:
                best_score = mean_score
                best_params = current_params
        
        print()  # New line after progress bar
        return {
            'best_params': best_params,
            'best_score': best_score
        } 