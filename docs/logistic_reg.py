import marimo

# Logistic Regression - Implementation and Examples

__generated_with = "0.12.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    mo.md("# Logistic Regression - Implementation and Examples")
    return mo


@app.cell
def _():
    # ! pip install numpy seaborn
    return


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
    from regressors.models.logistic_regression import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_curve,
        auc,
        confusion_matrix
    )
    import matplotlib.pyplot as plt

    from regressors import LogisticRegression
    return (
        GridSearchCV,
        LogisticRegression,
        SKLogisticRegression,
        StandardScaler,
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        load_breast_cancer,
        np,
        plt,
        precision_score,
        recall_score,
        roc_curve,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## LogisticRegression Implementation
        """
    )
    return


@app.cell
def _(load_breast_cancer):
    # Load the breast cancer dataset
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    return X, breast_cancer, y


@app.cell
def _(X, breast_cancer, y):
    # Print information about the dataset
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {breast_cancer.feature_names}")
    print(f"Target: Binary (0/1)")
    return


@app.cell
def _(StandardScaler, X, train_test_split, y):
    # Scale the features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled, X_test, X_train, scaler, y_test, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    print("\nPerforming grid search with custom logistic regression...")
    # Initialize model
    custom_model = LogisticRegression()

    # Define parameter grid with more conservative values
    param_grid = {
        'max_iter': [100, 1000, 2000],
        'tol': [1e-4, 1e-6],
        'learning_rate': [0.001, 0.01, 0.1],  # Added smaller learning rate
        'lambda_param': [0.0001, 0.001, 0.01],  # Added smaller regularization values
        'regularization': ['l2']  # Start with just L2 to simplify
    }

    # Perform grid search
    results = custom_model.grid_search(
        X_train, y_train,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    print("Best parameters:", results['best_params'])
    print("Best cross-validation score:", results['best_score'])
    return custom_model, param_grid, results


@app.cell
def _(
    LogisticRegression,
    X_test,
    X_train,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    results,
    y_test,
    y_train,
):
    print("\nTraining best custom model...")
    # Get default parameters
    default_params = {
        'max_iter': 1000,
        'tol': 1e-4,
        'patience': 5,
        'adaptive_learning': True,
        'min_lr': 1e-4,
        'lr_decay_rate': 0.5
    }

    # Combine with best parameters from grid search
    model_params = default_params.copy()
    model_params.update(results['best_params'])

    # Train model with combined parameters
    best_custom_model = LogisticRegression(**model_params)
    best_custom_model.fit(X_train, y_train)

    # Get predictions and probabilities
    custom_predictions = best_custom_model.predict(X_test)
    custom_probas = best_custom_model.predict_proba(X_test)

    # Calculate metrics for custom model
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    custom_precision = precision_score(y_test, custom_predictions)
    custom_recall = recall_score(y_test, custom_predictions)
    custom_f1 = f1_score(y_test, custom_predictions)

    print("\nCustom Logistic Regression Performance:")
    print(f"Accuracy: {custom_accuracy:.4f}")
    print(f"Precision: {custom_precision:.4f}")
    print(f"Recall: {custom_recall:.4f}")
    print(f"F1 Score: {custom_f1:.4f}")
    return (
        best_custom_model,
        custom_accuracy,
        custom_f1,
        custom_precision,
        custom_predictions,
        custom_probas,
        custom_recall,
        default_params,
        model_params,
    )


@app.cell
def _(
    GridSearchCV,
    SKLogisticRegression,
    X_test,
    X_train,
    accuracy_score,
    custom_accuracy,
    custom_f1,
    custom_precision,
    custom_recall,
    f1_score,
    precision_score,
    recall_score,
    y_test,
    y_train,
):
    print("\nTraining sklearn model with grid search...")
    # Define sklearn parameter grid
    sklearn_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'max_iter': [2000],  # Increased from 1000
        'tol': [1e-4],
        'solver': ['lbfgs', 'newton-cg', 'liblinear']  # Removed sag, added liblinear
    }

    # Perform grid search with sklearn
    sklearn_grid = GridSearchCV(
        SKLogisticRegression(random_state=42),
        sklearn_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    sklearn_grid.fit(X_train, y_train)

    print("Best sklearn parameters:", sklearn_grid.best_params_)
    print("Best sklearn cross-validation score:", sklearn_grid.best_score_)

    # Get best sklearn model
    sklearn_model = sklearn_grid.best_estimator_
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_probas = sklearn_model.predict_proba(X_test)

    # Calculate metrics for sklearn model
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_precision = precision_score(y_test, sklearn_predictions)
    sklearn_recall = recall_score(y_test, sklearn_predictions)
    sklearn_f1 = f1_score(y_test, sklearn_predictions)

    # Print performance comparison
    print("\nMetric Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Custom':>15} {'Sklearn':>15}")
    print("-" * 50)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    custom_scores = [custom_accuracy, custom_precision, custom_recall, custom_f1]
    sklearn_scores = [sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1]
    for metric, custom, sklearn in zip(metrics, custom_scores, sklearn_scores):
        print(f"{metric:<15} {custom:>15.4f} {sklearn:>15.4f}")
    print("-" * 50)
    return (
        custom,
        custom_scores,
        metric,
        metrics,
        sklearn,
        sklearn_accuracy,
        sklearn_f1,
        sklearn_grid,
        sklearn_model,
        sklearn_param_grid,
        sklearn_precision,
        sklearn_predictions,
        sklearn_probas,
        sklearn_recall,
        sklearn_scores,
    )


@app.cell
def _(
    auc,
    best_custom_model,
    confusion_matrix,
    custom_predictions,
    custom_probas,
    plt,
    roc_curve,
    sklearn_predictions,
    sklearn_probas,
    y_test,
):
    # Visualizations
    plt.style.use('default')  # Use default matplotlib style instead of seaborn

    # Set up better default parameters for plots
    plt.rcParams.update({
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.titlesize': 12,
        'axes.labelsize': 10
    })

    # 1. Cost History and ROC Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Cost History
    ax1.plot(best_custom_model.cost_history)
    ax1.set_title('Cost History During Training')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost')
    ax1.grid(True)

    # ROC Curves
    # Calculate ROC curve for custom model
    custom_fpr, custom_tpr, _ = roc_curve(y_test, custom_probas[:,1])
    custom_auc = auc(custom_fpr, custom_tpr)

    # Calculate ROC curve for sklearn model
    sklearn_fpr, sklearn_tpr, _ = roc_curve(y_test, sklearn_probas[:, 1])
    sklearn_auc = auc(sklearn_fpr, sklearn_tpr)

    ax2.plot(custom_fpr, custom_tpr, label=f'Custom (AUC = {custom_auc:.3f})')
    ax2.plot(sklearn_fpr, sklearn_tpr, label=f'Sklearn (AUC = {sklearn_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Custom model confusion matrix
    custom_cm = confusion_matrix(y_test, custom_predictions)
    im1 = ax1.imshow(custom_cm, cmap='Blues')
    ax1.set_title('Custom Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, custom_cm[i, j], ha='center', va='center')

    # Sklearn model confusion matrix
    sklearn_cm = confusion_matrix(y_test, sklearn_predictions)
    im2 = ax2.imshow(sklearn_cm, cmap='Blues')
    ax2.set_title('Sklearn Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, sklearn_cm[i, j], ha='center', va='center')

    plt.tight_layout()
    plt.show()

    # 4. Probability Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Custom model probabilities
    ax1.hist(custom_probas[:,1][y_test == 0], bins=20, alpha=0.5, label='Negative class', density=True)
    ax1.hist(custom_probas[:,1][y_test == 1], bins=20, alpha=0.5, label='Positive class', density=True)
    ax1.set_title('Custom Model\nProbability Distribution')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True)

    # Sklearn model probabilities
    ax2.hist(sklearn_probas[:, 1][y_test == 0], bins=20, alpha=0.5, label='Negative class', density=True)
    ax2.hist(sklearn_probas[:, 1][y_test == 1], bins=20, alpha=0.5, label='Positive class', density=True)
    ax2.set_title('Sklearn Model\nProbability Distribution')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return (
        ax1,
        ax2,
        custom_auc,
        custom_cm,
        custom_fpr,
        custom_tpr,
        fig,
        i,
        im1,
        im2,
        j,
        sklearn_auc,
        sklearn_cm,
        sklearn_fpr,
        sklearn_tpr,
    )


if __name__ == "__main__":
    app.run()
