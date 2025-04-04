import marimo

# Softmax Regression - Implementation and Examples

__generated_with = "0.12.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    mo.md("# Softmax Regression - Implementation and Examples")
    return mo

@app.cell
def _():
    # ! pip install numpy
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

    from regressors import LogisticRegression
    from sklearn.linear_model import LogisticRegression as SKLogisticRegression
    return (
        GridSearchCV,
        LogisticRegression,
        SKLogisticRegression,
        StandardScaler,
        accuracy_score,
        auc,
        confusion_matrix,
        f1_score,
        label_binarize,
        load_digits,
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
        ## SoftmaxRegression Implementation
        """
    )
    return


@app.cell
def _(load_digits):
    wine = load_digits()
    X, y = wine.data, wine.target
    return X, wine, y


@app.cell
def _(StandardScaler, X, train_test_split, y):
    # Scale the features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_scaled, X_test, X_train, scaler, y_test, y_train


@app.cell
def _(X_test, X_train, np, y):
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    return


@app.cell
def _(LogisticRegression, X_train, y_train):
    # Now perform grid search with your custom implementation
    print("\nPerforming grid search with custom softmax logistic regression...")
    # Initialize your softmax model (assuming your class is named SoftmaxRegression)
    custom_model = LogisticRegression()

    # Define parameter grid for multi-class classification
    param_grid = {
        'max_iter': [3000],
        'tol': [1e-4, 1e-6],
        'learning_rate': [0.01, 0.1],
        'lambda_param': [0.0001, 0.001, 0.01],
        'regularization': ['l1', 'l2']
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
    print('\nTraining best custom model...')
    default_params = {'max_iter': 2000, 'tol': 0.0001, 'patience': 5, 'adaptive_learning': True, 'min_lr': 0.0001, 'lr_decay_rate': 0.5}
    model_params = default_params.copy()
    model_params.update(results['best_params'])
    best_custom_model = LogisticRegression(**model_params)
    best_custom_model.fit(X_train, y_train)
    custom_predictions = best_custom_model.predict(X_test)
    _custom_probas = best_custom_model.predict_proba(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    custom_precision = precision_score(y_test, custom_predictions, average='weighted')
    custom_recall = recall_score(y_test, custom_predictions, average='weighted')
    custom_f1 = f1_score(y_test, custom_predictions, average='weighted')
    print('\nCustom Logistic Regression Performance:')
    print(f'Accuracy: {custom_accuracy:.4f}')
    print(f'Precision: {custom_precision:.4f}')
    print(f'Recall: {custom_recall:.4f}')
    print(f'F1: {custom_f1:.4f}')
    return (
        best_custom_model,
        custom_accuracy,
        custom_f1,
        custom_precision,
        custom_predictions,
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
    print('\nTraining sklearn model with grid search...')
    sklearn_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [2000], 'tol': [0.0001], 'solver': ['lbfgs', 'newton-cg', 'liblinear']}
    sklearn_grid = GridSearchCV(SKLogisticRegression(random_state=42), sklearn_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    sklearn_grid.fit(X_train, y_train)
    print('Best sklearn parameters:', sklearn_grid.best_params_)
    print('Best sklearn cross-validation score:', sklearn_grid.best_score_)
    sklearn_model = sklearn_grid.best_estimator_
    sklearn_predictions = sklearn_model.predict(X_test)
    _sklearn_probas = sklearn_model.predict_proba(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_precision = precision_score(y_test, sklearn_predictions, average='weighted')
    sklearn_recall = recall_score(y_test, sklearn_predictions, average='weighted')
    sklearn_f1 = f1_score(y_test, sklearn_predictions, average='weighted')
    print('\nMetric Comparison:')
    print('-' * 50)
    print(f"{'Metric':<15} {'Custom':>15} {'Sklearn':>15}")
    print('-' * 50)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    custom_scores = [custom_accuracy, custom_precision, custom_recall, custom_f1]
    sklearn_scores = [sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1]
    for (metric, custom, sklearn) in zip(metrics, custom_scores, sklearn_scores):
        print(f'{metric:<15} {custom:>15.4f} {sklearn:>15.4f}')
    print('-' * 50)
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
        sklearn_recall,
        sklearn_scores,
    )


@app.cell
def _(
    X_test,
    auc,
    best_custom_model,
    confusion_matrix,
    custom_predictions,
    label_binarize,
    np,
    plt,
    roc_curve,
    sklearn_grid,
    sklearn_predictions,
    y,
    y_test,
):
    plt.style.use('default')
    plt.rcParams.update({'figure.autolayout': True, 'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2, 'axes.titlesize': 12, 'axes.labelsize': 10})
    _custom_probas = best_custom_model.predict_proba(X_test)
    _sklearn_probas = sklearn_grid.best_estimator_.predict_proba(X_test)
    plt.figure(figsize=(10, 4))
    plt.plot(best_custom_model.cost_history)
    plt.title('Cost History During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 6))
    custom_cm = confusion_matrix(y_test, custom_predictions)
    im1 = ax1.imshow(custom_cm, cmap='Blues')
    ax1.set_title('Custom Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    fig.colorbar(im1, ax=ax1)
    if len(custom_cm) <= 10:
        for i in range(len(custom_cm)):
            for j in range(len(custom_cm)):
                ax1.text(j, i, custom_cm[i, j], ha='center', va='center', color='white' if custom_cm[i, j] > custom_cm.max() / 2 else 'black')
    sklearn_cm = confusion_matrix(y_test, sklearn_predictions)
    im2 = ax2.imshow(sklearn_cm, cmap='Blues')
    ax2.set_title('Sklearn Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    fig.colorbar(im2, ax=ax2)
    if len(sklearn_cm) <= 10:
        for i in range(len(sklearn_cm)):
            for j in range(len(sklearn_cm)):
                ax2.text(j, i, sklearn_cm[i, j], ha='center', va='center', color='white' if sklearn_cm[i, j] > sklearn_cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()
    n_classes = len(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(10, 8))
    classes_to_plot = range(min(5, n_classes))
    for i in classes_to_plot:
        (custom_fpr, custom_tpr, _) = roc_curve(y_test_bin[:, i], _custom_probas[:, i])
        custom_auc_val = auc(custom_fpr, custom_tpr)
        plt.plot(custom_fpr, custom_tpr, lw=2, label=f'Class {i} - Custom (AUC = {custom_auc_val:.2f})')
        (sklearn_fpr, sklearn_tpr, _) = roc_curve(y_test_bin[:, i], _sklearn_probas[:, i])
        sklearn_auc_val = auc(sklearn_fpr, sklearn_tpr)
        plt.plot(sklearn_fpr, sklearn_tpr, lw=2, linestyle='--', label=f'Class {i} - Sklearn (AUC = {sklearn_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    boxplot_data = []
    labels = []
    for digit in range(10):
        true_samples = y_test == digit
        correct_probs = _custom_probas[true_samples, digit]
        boxplot_data.append(correct_probs)
        labels.append(f'{digit}-Custom')
        correct_probs = _sklearn_probas[true_samples, digit]
        boxplot_data.append(correct_probs)
        labels.append(f'{digit}-Sklearn')
    plt.boxplot(boxplot_data, labels=labels, showfliers=False)
    plt.xticks(rotation=45)
    plt.title('Probability Distribution by Class')
    plt.ylabel('Probability assigned to correct class')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (
        ax1,
        ax2,
        boxplot_data,
        classes_to_plot,
        correct_probs,
        custom_auc_val,
        custom_cm,
        custom_fpr,
        custom_tpr,
        digit,
        fig,
        i,
        im1,
        im2,
        j,
        labels,
        n_classes,
        sklearn_auc_val,
        sklearn_cm,
        sklearn_fpr,
        sklearn_tpr,
        true_samples,
        y_test_bin,
    )


if __name__ == "__main__":
    app.run()
