import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from regressors import LinearRegression

def test_linear_regression_fit_predict():
    """Test basic fitting and prediction functionality."""
    # Generate simple linear data
    X = np.array([[1], [2], [3], [4]])
    y = 2 * X.ravel() + 1  # y = 2x + 1
    
    # Create and fit model
    model = LinearRegression(
        learning_rate=0.01,  # Smaller learning rate for stability
        max_iter=5000,  # More iterations
        tol=1e-6,  # Tighter tolerance
        adaptive_learning=False,  # Disable adaptive learning for stability
        regularization=None  # No regularization needed for this simple case
    )
    model.fit(X, y)
    
    # Test predictions
    X_test = np.array([[5], [6]])
    predictions = model.predict(X_test)
    
    # Check if predictions are close to expected values
    expected = np.array([11, 13])  # 2*5+1=11, 2*6+1=13
    assert np.allclose(predictions, expected, rtol=0.1)

def test_regularization():
    """Test that L2 regularization reduces coefficient magnitudes."""
    # Generate data with high noise to better demonstrate regularization effect
    np.random.seed(42)  # For reproducibility
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    # True relationship: y = 2x + 1, but with high noise
    noise = np.random.normal(0, 2, size=10)  # High noise
    y = 2 * X.ravel() + 1 + noise
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit with and without regularization
    model_no_reg = LinearRegression(
        regularization=None,
        learning_rate=0.01,
        max_iter=2000,
        adaptive_learning=False  # Disable adaptive learning for this test
    )
    model_l2 = LinearRegression(
        regularization='l2',
        lambda_param=10.0,  # Strong regularization
        learning_rate=0.01,
        max_iter=2000,
        adaptive_learning=False  # Disable adaptive learning for this test
    )
    
    model_no_reg.fit(X_scaled, y)
    model_l2.fit(X_scaled, y)
    
    # The L2 regularized model should have smaller weights
    assert np.abs(model_l2.coef_[0]) < np.abs(model_no_reg.coef_[0]), \
        f"L2 weights ({model_l2.coef_[0]}) should be smaller than unregularized weights ({model_no_reg.coef_[0]})"

def test_early_stopping():
    """Test that early stopping triggers before max_iter."""
    # Generate data
    X = np.array([[1], [2], [3], [4]])
    y = 2 * X.ravel() + 1
    
    # Create model with small patience
    model = LinearRegression(
        learning_rate=0.01,
        patience=2,
        tol=1e-4,
        adaptive_learning=False  # Disable adaptive learning for stability
    )
    model.fit(X, y)
    
    # Check if early stopping was triggered
    assert model.n_iter_ < 1000  # Should stop before max_iter

def test_adaptive_learning():
    """Test that adaptive learning rate reduces cost."""
    # Generate data
    X = np.array([[1], [2], [3], [4]])
    y = 2 * X.ravel() + 1
    
    # Create model with adaptive learning
    model = LinearRegression(
        adaptive_learning=True,
        learning_rate=0.01,
        tol=1e-4
    )
    model.fit(X, y)
    
    # Check if cost decreased
    assert len(model.cost_history) > 0, "Cost history should not be empty"
    assert model.cost_history[-1] < model.cost_history[0]

def test_score_metrics():
    """Test different scoring metrics."""
    # Generate regression dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = LinearRegression(
        learning_rate=0.01,
        max_iter=1000,
        regularization=None
    )
    model.fit(X_scaled, y)
    
    # Test MSE
    mse = model.score(X_scaled, y, metric='mse')
    assert mse > 0, "MSE should be positive"
    
    # Test RMSE
    rmse = model.score(X_scaled, y, metric='rmse')
    assert rmse > 0, "RMSE should be positive"
    assert np.isclose(rmse, np.sqrt(mse)), "RMSE should be square root of MSE"
    
    # Test R2
    r2 = model.score(X_scaled, y, metric='r2')
    assert r2 <= 1.0, "R2 should be <= 1"
    assert r2 > 0.9, "R2 should be high for this simple dataset"

def test_sklearn_preprocessing_compatibility():
    """Test compatibility with scikit-learn preprocessing."""
    # Generate data with different scales
    X = np.array([[100, 0.1], [200, 0.2], [300, 0.3], [400, 0.4]])
    y = X[:, 0] * 2 + X[:, 1] * 10 + 1
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model
    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
        tol=1e-6
    )
    model.fit(X_scaled, y)
    
    # Make predictions on scaled data
    X_test = np.array([[500, 0.5]])
    X_test_scaled = scaler.transform(X_test)
    prediction = model.predict(X_test_scaled)
    
    # Expected value: 500*2 + 0.5*10 + 1 = 1006
    assert np.isclose(prediction, 1006, rtol=0.1)

def test_invalid_score_metric():
    """Test that invalid score metric raises ValueError."""
    X = np.array([[1], [2], [3], [4]])
    y = 2 * X.ravel() + 1
    
    model = LinearRegression()
    model.fit(X, y)
    
    with pytest.raises(ValueError):
        model.score(X, y, metric='invalid_metric')

def test_edge_cases():
    """Test model behavior with edge cases."""
    # Test with empty feature matrix
    X = np.array([]).reshape(0, 1)
    y = np.array([])
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)
    
    # Test with NaN values
    X = np.array([[1], [2], [np.nan], [4]])
    y = np.array([1, 2, 3, 4])
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)
    
    # Test with infinite values
    X = np.array([[1], [2], [np.inf], [4]])
    y = np.array([1, 2, 3, 4])
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)
    
    # Test with very large/small values
    X = np.array([[1e10], [2e10], [3e10], [4e10]])
    y = 2 * X.ravel() + 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    X_test = np.array([[5e10]])
    X_test_scaled = scaler.transform(X_test)
    prediction = model.predict(X_test_scaled)
    assert not np.isnan(prediction), "Prediction should not be NaN for large values"

def test_convergence_behavior():
    """Test model convergence under different conditions."""
    # Generate data
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test convergence with different tolerances
    tols = [1e-2, 1e-4, 1e-6]
    n_iters = []
    for tol in tols:
        model = LinearRegression(tol=tol, adaptive_learning=False)
        model.fit(X_scaled, y)
        n_iters.append(model.n_iter_)
    assert n_iters[0] < n_iters[1] < n_iters[2], "More iterations needed for tighter tolerance"
    
    # Test convergence with different learning rates
    lrs = [0.1, 0.01, 0.001]
    costs = []
    for lr in lrs:
        model = LinearRegression(learning_rate=lr, adaptive_learning=False)
        model.fit(X_scaled, y)
        costs.append(model.cost_history[-1])
    assert all(cost > 0 for cost in costs), "Final costs should be positive"
    
    # Test convergence with different regularization strengths
    lambdas = [0.1, 1.0, 10.0]
    coef_norms = []
    for lambda_param in lambdas:
        model = LinearRegression(regularization='l2', lambda_param=lambda_param)
        model.fit(X_scaled, y)
        coef_norms.append(np.linalg.norm(model.coef_))
    assert coef_norms[0] > coef_norms[1] > coef_norms[2], "Stronger regularization should lead to smaller coefficients"

def test_multifeature_handling():
    """Test model behavior with multiple features."""
    # Generate multi-feature dataset with known coefficients
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    true_coef = np.array([10.0, 1.0, 1.0, 1.0, 1.0])  # First feature is 10x more important
    y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)  # Small noise
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test with many features
    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
        tol=1e-6,
        regularization=None  # No regularization for this test
    )
    model.fit(X_scaled, y)
    assert len(model.coef_) == 5, "Should have coefficient for each feature"
    
    # Test with correlated features
    X_corr = np.copy(X_scaled)
    X_corr[:, 1] = X_corr[:, 0] * 0.9 + np.random.normal(0, 0.1, size=n_samples)  # Create correlation
    model = LinearRegression(regularization='l2')  # L2 regularization helps with correlated features
    model.fit(X_corr, y)
    predictions = model.predict(X_corr)
    assert not np.any(np.isnan(predictions)), "Predictions should not be NaN with correlated features"
    
    # Test feature importance by creating a new dataset where first feature is clearly more important
    X_imp = np.random.randn(n_samples, 2)  # Only 2 features for clarity
    y_imp = 10.0 * X_imp[:, 0] + 1.0 * X_imp[:, 1] + np.random.normal(0, 0.1, n_samples)
    X_imp_scaled = scaler.fit_transform(X_imp)
    
    model = LinearRegression(
        learning_rate=0.01,
        max_iter=5000,
        tol=1e-6,
        regularization=None
    )
    model.fit(X_imp_scaled, y_imp)
    assert abs(model.coef_[0]) > abs(model.coef_[1]), \
        "First coefficient should be larger than second coefficient"

def test_grid_search():
    """Test grid search functionality."""
    # Generate dataset
    X, y = make_regression(n_samples=50, n_features=1, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test parameter grid validation
    model = LinearRegression()
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'lambda_param': [0.1, 1.0],
        'regularization': ['l1', 'l2']
    }
    results = model.grid_search(X_scaled, y, param_grid, cv=3)
    assert 'best_params' in results, "Grid search should return best parameters"
    assert 'best_score' in results, "Grid search should return best score"
    
    # Test cross-validation splits
    cv_sizes = [2, 3, 5]
    scores = []
    for cv in cv_sizes:
        results = model.grid_search(X_scaled, y, param_grid, cv=cv)
        scores.append(results['best_score'])
    assert len(set(scores)) > 1, "Different CV splits should give different scores"
    
    # Test best parameter selection
    strong_reg = {
        'learning_rate': [0.01],
        'lambda_param': [100.0],  # Very strong regularization
        'regularization': ['l2']
    }
    weak_reg = {
        'learning_rate': [0.01],
        'lambda_param': [0.0001],  # Very weak regularization
        'regularization': ['l2']
    }
    strong_results = model.grid_search(X_scaled, y, strong_reg, cv=3)
    weak_results = model.grid_search(X_scaled, y, weak_reg, cv=3)
    assert strong_results['best_score'] > weak_results['best_score'], \
        "Strong regularization should perform better on noisy data" 