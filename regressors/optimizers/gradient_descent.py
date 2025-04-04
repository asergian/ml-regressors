import numpy as np
from ..base.base_optimizer import BaseOptimizer

class GradientDescentOptimizer(BaseOptimizer):
    """Gradient descent optimizer with adaptive learning rate and early stopping.
    
    This class implements gradient descent optimization with features like:
    - Adaptive learning rate
    - Early stopping
    - Convergence checking
    - Cost history tracking
    
    Attributes:
        learning_rate: float, initial step size.
        max_iter: int, maximum number of iterations.
        tol: float, convergence tolerance.
        patience: int, number of iterations with no improvement before early stopping.
        adaptive_learning: bool, whether to use adaptive learning rate.
        min_lr: float, minimum learning rate when using adaptive learning.
        lr_decay_rate: float, decay rate for learning rate.
        n_iter_: int, number of iterations run.
        cost_history: list, stores cost at each iteration.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001,
                 patience=5, adaptive_learning=True, min_lr=1e-6,
                 lr_decay_rate=0.1):
        """Initialize the gradient descent optimizer.
        
        Args:
            learning_rate: float, initial step size.
            max_iter: int, maximum number of iterations.
            tol: float, convergence tolerance.
            patience: int, number of iterations with no improvement before early stopping.
            adaptive_learning: bool, whether to use adaptive learning rate.
            min_lr: float, minimum learning rate when using adaptive learning.
            lr_decay_rate: float, decay rate for learning rate.
        """
        super().__init__(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            patience=patience,
            adaptive_learning=adaptive_learning,
            min_lr=min_lr,
            lr_decay_rate=lr_decay_rate
        )
        self._current_lr = learning_rate

    def optimize(self, model, X, y):
        """Run gradient descent optimization.
        
        Args:
            model: BaseRegression instance
                The model to optimize.
            X: array-like, shape (n_samples, n_features)
                Training data.
            y: array-like, shape (n_samples,)
                Target values.
                
        Returns:
            self: Returns the instance itself.
        """
        n_samples = len(X)
        
        # Initialize model parameters if not already initialized
        if model.coef_ is None:
            model._initialize_parameters(X.shape[1])
        
        # Get initial parameters and cost
        prev_params = {
            'coef_': model.coef_.copy(),
            'intercept_': model.intercept_
        }
        prev_cost = model.calculate_cost(X, y)
        self.cost_history.append(prev_cost)
        
        # Run optimization
        for self.n_iter_ in range(self.max_iter):
            # Calculate gradients and update parameters
            gradients = model._calculate_gradients(X, y, n_samples)
            self._update_parameters(model, gradients, self._current_lr)
            
            # Get current parameters and cost
            current_params = {
                'coef_': model.coef_.copy(),
                'intercept_': model.intercept_
            }
            current_cost = model.calculate_cost(X, y)
            self.cost_history.append(current_cost)
            
            # Check convergence
            if self._check_convergence(prev_params, current_params, prev_cost, current_cost):
                break
                
            # Update learning rate if using adaptive learning
            if self.adaptive_learning:
                self._current_lr = self._update_learning_rate(self._current_lr, current_cost)
            
            # Update previous values
            prev_params = current_params
            prev_cost = current_cost
        
        return self