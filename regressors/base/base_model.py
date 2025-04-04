from abc import ABC, abstractmethod
import numpy as np

class BaseRegression(ABC):
    """Base class for regression models.
    
    This class provides the interface and common functionality for regression models.
    It handles parameter initialization and gradient calculation.
    
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
        """Initialize the regression model.
        
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
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.patience = patience
        self.adaptive_learning = adaptive_learning
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay_rate
        
        # Initialize model parameters
        self.coef_ = None
        self.intercept_ = None

    def _initialize_parameters(self, n_features):
        """Initialize model parameters.
        
        Args:
            n_features: int
                Number of features in the input data.
        """
        # Initialize with small random values instead of zeros
        # Using He initialization scaled down to avoid saturation
        self.coef_ = np.random.randn(n_features) * np.sqrt(2.0 / n_features) * 0.01
        self.intercept_ = np.float64(0.0)  # Keep intercept at 0 initially

    @abstractmethod
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
        pass

    @abstractmethod
    def calculate_cost(self, X, y, predictions=None):
        """Calculate the cost function with regularization.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
            y: array-like, shape (n_samples,)
                Target values.
            predictions: array-like, shape (n_samples,), optional
                Pre-computed predictions.
                
        Returns:
            float: Cost value with regularization.
        """
        pass

    def _calculate_regularization_cost(self, n_samples):
        """Calculate the regularization term of the cost function.
        
        Args:
            n_samples: int
                Number of samples in the dataset.
                
        Returns:
            float: Regularization cost.
        """
        if self.coef_ is None:
            return 0.0
            
        if self.regularization == 'l1':
            return (self.lambda_param / n_samples) * np.sum(np.abs(self.coef_))
        elif self.regularization == 'l2':
            return (0.5 * self.lambda_param / n_samples) * np.sum(self.coef_ ** 2)
        return 0.0

    def _calculate_regularization_gradient(self, n_samples):
        """Calculate the regularization term of the gradient.
        
        Args:
            n_samples: int
                Number of samples in the dataset.
                
        Returns:
            ndarray: Regularization gradient for coefficients.
        """
        if self.coef_ is None:
            return 0.0
            
        if self.regularization == 'l2':
            return (self.lambda_param / n_samples) * self.coef_
        elif self.regularization == 'l1':
            return (self.lambda_param / n_samples) * np.sign(self.coef_)
        return 0.0

    @abstractmethod
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Input data.
                
        Returns:
            array-like, shape (n_samples,)
                Predicted values.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the training data.
        
        Args:
            X: array-like, shape (n_samples, n_features)
                Training data.
            y: array-like, shape (n_samples,)
                Target values.
                
        Returns:
            self: Returns the instance itself.
        """
        pass 