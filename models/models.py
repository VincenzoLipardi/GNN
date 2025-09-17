import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR


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