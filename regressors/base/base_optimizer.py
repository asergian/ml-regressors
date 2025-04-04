from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    """Abstract base class for all optimizers.
    
    This class defines the common interface and shared methods for all optimizers.
    Concrete implementations must implement the abstract methods.
    
    Attributes:
        learning_rate: float, initial step size for gradient descent.
        max_iter: int, maximum number of iterations.
        tol: float, convergence tolerance.
        patience: int, number of iterations with no improvement before early stopping.
        adaptive_learning: bool, whether to use adaptive learning rate.
        min_lr: float, minimum learning rate when using adaptive learning.
        lr_decay_rate: float, decay rate for learning rate.
        cost_history: list, stores cost at each iteration.
        n_iter_: int, number of iterations run during optimization.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=0.0001,
                 patience=5, adaptive_learning=True, min_lr=1e-6,
                 lr_decay_rate=0.1):
        """Initialize the base optimizer.
        
        Args:
            learning_rate: float, initial step size for gradient descent.
            max_iter: int, maximum number of iterations.
            tol: float, convergence tolerance.
            patience: int, number of iterations with no improvement before early stopping.
            adaptive_learning: bool, whether to use adaptive learning rate.
            min_lr: float, minimum learning rate when using adaptive learning.
            lr_decay_rate: float, decay rate for learning rate.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.adaptive_learning = adaptive_learning
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay_rate
        self.cost_history = []
        self.n_iter_ = None

    @abstractmethod
    def optimize(self, model, X, y):
        """Optimize the model parameters.
        
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
        pass

    def _check_convergence(self, prev_params, current_params, prev_cost, current_cost):
        """Check if optimization has converged.
        
        Convergence is determined by either:
        1. No cost improvement over the last 'patience' iterations
        2. Parameter changes below tolerance threshold
        
        Args:
            prev_params: dict
                Previous model parameters.
            current_params: dict
                Current model parameters.
            prev_cost: float
                Previous cost value.
            current_cost: float
                Current cost value.
                
        Returns:
            bool: True if converged, False otherwise.
        """
        # Need enough history for patience check
        if len(self.cost_history) < self.patience:
            return False
        
        # Check if cost has improved in the last 'patience' iterations
        recent_costs = self.cost_history[-self.patience:]
        cost_converged = all(cost >= recent_costs[0] for cost in recent_costs[1:])
        
        # Check parameter convergence
        param_diff = np.sum([np.sum(np.abs(current_params[k] - prev_params[k]))
                            for k in current_params.keys()])
        param_converged = param_diff < self.tol
        
        return cost_converged or param_converged

    def _update_parameters(self, model, gradients, current_lr):
        """Update model parameters using calculated gradients.
        
        Args:
            model: BaseRegression instance
                The model to update.
            gradients: dict
                Dictionary of gradients for each parameter.
            current_lr: float
                Current learning rate.
        """
        for param_name, gradient in gradients.items():
            current_value = getattr(model, param_name)
            setattr(model, param_name, current_value - current_lr * gradient)

    def _update_learning_rate(self, current_lr, cost):
        """Update learning rate based on cost changes.
        
        The learning rate is reduced when the cost increases, indicating
        that we might be overshooting the minimum.
        
        Args:
            current_lr: float
                Current learning rate.
            cost: float
                Current cost value.
                
        Returns:
            float: Updated learning rate.
        """
        if not self.adaptive_learning or len(self.cost_history) < 2:
            return current_lr
        
        # Check if cost increased
        if cost > self.cost_history[-2]:
            # Decay learning rate
            current_lr *= self.lr_decay_rate
            
            # Ensure learning rate doesn't go below minimum
            current_lr = max(current_lr, self.min_lr)
        
        return current_lr 