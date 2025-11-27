import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch

# Local imports for dataset loaders
try:
    from models.gnn import build_train_test_loaders
except Exception:
    # Fallback when this module is executed in different contexts
    from .gnn import build_train_test_loaders


# Function to train Random Forest on data
def random_forest(data, labels, test_size, verbose, estimators, criterion, max_features):
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    rfr = RandomForestRegressor(n_estimators=estimators, criterion=criterion, max_features=max_features, random_state=42)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    # Calculate metrics for test set
    
    mean_squared_error_test = mean_squared_error(y_test, y_pred)
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    r_squared_test = r2_score(y_test, y_pred)

    # Calculate metrics for training set
    y_train_pred = rfr.predict(X_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_train_pred)
    mean_squared_error_train = mean_squared_error(y_train, y_train_pred)
    r_squared_train = r2_score(y_train, y_train_pred)

    if verbose:
        print("\nTraining Set:")
        print(f"Mean Absolute Error: {mean_absolute_error_train:.4f}")
        print(f"Mean Squared Error: {mean_squared_error_train:.4f}")
        print(f"R-squared Score: {r_squared_train:.4f}")

        print("Test Set:")
        print(f"Mean Absolute Error: {mean_absolute_error_test:.4f}")
        print(f"Mean Squared Error: {mean_squared_error_test:.4f}")
        print(f"R-squared Score: {r_squared_test:.4f}")
    return (mean_absolute_error_test, mean_squared_error_test, r_squared_test,
            mean_absolute_error_train, mean_squared_error_train, r_squared_train, rfr)



def hyperparameter_search_rf(X_train, y_train):
    model = RandomForestRegressor(random_state=1)
    hp_grid = {
        'n_estimators': [300],  
        'max_features': ['sqrt', 0.3,0.5],#, 'log2'], 
        'criterion': ['squared_error'],#,'friedman_mse', 'absolute_error'],
        'max_depth': [None, 10, 30],
        'min_samples_split': [2, 5],
        #'min_samples_leaf': [1, 2, 4]
    }
    GSCV = GridSearchCV(estimator=model, param_grid=hp_grid, cv=3)
    GSCV.fit(X_train, y_train)
    print("Best params:", GSCV.best_params_)
    return GSCV.best_params_

def svm(data, labels, test_size, verbose, kernel='linear', C=1.0, epsilon=0.1):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    # Calculate metrics for test set
    mean_squared_error_test = mean_squared_error(y_test, y_pred)
    mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
    r_squared_test = r2_score(y_test, y_pred)

    # Calculate metrics for training set
    y_train_predictions = svr.predict(X_train)
    mean_absolute_error_train = mean_absolute_error(y_train, y_train_predictions)
    mean_squared_error_train = mean_squared_error(y_train, y_train_predictions)
    r_squared_train = r2_score(y_train, y_train_predictions)

    if verbose:
        print("\nSVM Training Set:")
        print(f"Mean Absolute Error: {mean_absolute_error_train:.4f}")
        print(f"Mean Squared Error: {mean_squared_error_train:.4f}")
        print(f"R-squared Score: {r_squared_train:.4f}")

        print("SVM Test Set:")
        print(f"Mean Absolute Error: {mean_absolute_error_test:.4f}")
        print(f"Mean Squared Error: {mean_squared_error_test:.4f}")
        print(f"R-squared Score: {r_squared_test:.4f}")
    return (
        mean_absolute_error_test, mean_squared_error_test, r_squared_test,
        mean_absolute_error_train, mean_squared_error_train, r_squared_train, svr)

def hyperparameter_search_svm(X_train, y_train):
    model = SVR()
    hp_grid = {
            'kernel': ["rbf"], 
            'C': [1],  
            'epsilon': [0.1]}
    GSCV = GridSearchCV(estimator=model, param_grid=hp_grid, cv=5)
    GSCV.fit(X_train, y_train)
    print("Best params:", GSCV.best_params_)
    return GSCV.best_params_


def svm_classification(
    data,
    labels,
    test_size,
    verbose,
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight=None,
    probability=False,
):
    """
    Train an SVM classifier (SVC) on provided features and labels.

    Notes:
    - Intended for 152-dim "binned152" global features used in the GNNs.
    - Applies standardization before SVC via a pipeline.
    - If labels are floats not strictly in {0,1}, they will be thresholded at 0.5.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42, stratify=labels)

    # Binarize labels if they are float-like and not exactly 0/1
    def _to_binary(y):
        y = np.asarray(y)
        uniq = np.unique(y)
        if set(uniq.tolist()) <= {0, 1}:
            return y.astype(int)
        return (y >= 0.5).astype(int)

    y_train_bin = _to_binary(y_train)
    y_test_bin = _to_binary(y_test)

    svc = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, probability=probability)
    )
    svc.fit(X_train, y_train_bin)

    # Predictions
    y_pred_test = svc.predict(X_test)
    y_pred_train = svc.predict(X_train)

    # For AUC/AP if probabilities requested; otherwise use decision_function
    if probability:
        scores_test = svc.predict_proba(X_test)[:, 1]
        scores_train = svc.predict_proba(X_train)[:, 1]
    else:
        scores_test = svc.decision_function(X_test)
        scores_train = svc.decision_function(X_train)

    # Metrics - Test
    acc_test = accuracy_score(y_test_bin, y_pred_test)
    pr_test, rc_test, f1_test, _ = precision_recall_fscore_support(y_test_bin, y_pred_test, average='binary', zero_division=0)
    try:
        auc_test = roc_auc_score(y_test_bin, scores_test)
    except Exception:
        auc_test = float('nan')
    try:
        ap_test = average_precision_score(y_test_bin, scores_test)
    except Exception:
        ap_test = float('nan')
    cm_test = confusion_matrix(y_test_bin, y_pred_test, labels=[0, 1])

    # Metrics - Train
    acc_train = accuracy_score(y_train_bin, y_pred_train)
    pr_train, rc_train, f1_train, _ = precision_recall_fscore_support(y_train_bin, y_pred_train, average='binary', zero_division=0)
    try:
        auc_train = roc_auc_score(y_train_bin, scores_train)
    except Exception:
        auc_train = float('nan')
    try:
        ap_train = average_precision_score(y_train_bin, scores_train)
    except Exception:
        ap_train = float('nan')

    if verbose:
        print("\nSVM Classification Training Set:")
        print(f"Accuracy: {acc_train:.4f}")
        print(f"Precision: {pr_train:.4f}")
        print(f"Recall: {rc_train:.4f}")
        print(f"F1: {f1_train:.4f}")
        print(f"ROC-AUC: {auc_train:.4f}")
        print(f"Avg Precision: {ap_train:.4f}")

        print("SVM Classification Test Set:")
        print(f"Accuracy: {acc_test:.4f}")
        print(f"Precision: {pr_test:.4f}")
        print(f"Recall: {rc_test:.4f}")
        print(f"F1: {f1_test:.4f}")
        print(f"ROC-AUC: {auc_test:.4f}")
        print(f"Avg Precision: {ap_test:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm_test)

    return (
        acc_test, pr_test, rc_test, f1_test, auc_test, ap_test,
        acc_train, pr_train, rc_train, f1_train,
        svc
    )


def hyperparameter_search_svc(X_train, y_train):
    """Grid search for SVC with standardization."""
    # Ensure binary labels (0/1)
    y_train_bin = np.asarray(y_train)
    if set(np.unique(y_train_bin).tolist()) - {0, 1}:
        y_train_bin = (y_train_bin >= 0.5).astype(int)

    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), SVC())
    hp_grid = {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1.0, 3.0],
        'svc__gamma': ['scale', 0.1, 0.03],
        'svc__class_weight': [None, 'balanced'],
    }
    GSCV = GridSearchCV(estimator=pipe, param_grid=hp_grid, cv=5)
    GSCV.fit(X_train, y_train_bin)
    print("Best params:", GSCV.best_params_)
    return GSCV.best_params_


def svm_classification_from_pkls(
    pkl_paths,
    test_size=0.2,
    verbose=True,
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    global_feature_variant: str = 'binned152',
    batch_size: int = 256,
    seed: int = 42,
):
    """
    Build a dataset from PKL paths using binned152 global features and train SVC.

    Returns same tuple as svm_classification with the fitted pipeline as last element.
    """
    # Build loaders and collect features/labels
    train_loader, test_loader = build_train_test_loaders(
        pkl_paths,
        train_split=1.0 - float(test_size),
        batch_size=int(batch_size),
        seed=int(seed),
        global_feature_variant=global_feature_variant,
        node_feature_backend_variant=None,
    )

    def _collect(loader):
        xs, ys = [], []
        for batch in loader:
            g = batch.global_features
            y = batch.y.view(-1)
            if g.dim() == 1:
                num_graphs = getattr(batch, 'num_graphs', int(y.shape[0]))
                g = g.view(int(num_graphs), -1)
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            xs.append(g[mask].cpu().numpy().astype(np.float32))
            ys.append(y[mask].cpu().numpy().astype(np.float32))
        if not xs:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = _collect(train_loader)
    X_test, y_test = _collect(test_loader)
    if X_train.size == 0 or X_test.size == 0:
        raise RuntimeError('Empty train/test after loading PKLs; check dataset paths and format.')

    return svm_classification(
        data=np.concatenate([X_train, X_test], axis=0),
        labels=np.concatenate([y_train, y_test], axis=0),
        test_size=float(test_size),
        verbose=verbose,
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
    )

def perform_pca(dataset_name, data, num_qubit, num_components=None):
    """
    Perform PCA on the given data and plot the explained variance ratio.
    
    Parameters:
    - data: The input data for PCA.
    - num_qubits: number of qubits of the quantum circuits analysed
    - num_components: Number of principal components to keep. If None, use all features.
    
    Returns:
    - transformed_data: The data transformed into the principal component space.
    """
    if num_components is None:
        num_components = int(data.shape[1]/4)

    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(data)
    # features = pca.components_
    # print(features[0])
    
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_components + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, num_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xticks(ticks=np.arange(1, num_components + 1))  
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('PCA Explained Variance')
    save_dir = "/data/P70087789/GNN/models/results"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/pca_{dataset_name}_{num_qubit}.png")
    
    return transformed_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data and return regression metrics.
    Returns: (mae, mse, rmse, r2)
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2