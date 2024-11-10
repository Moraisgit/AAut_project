"""
This module implements Support Vector Machine related functions.
"""

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from colorama import Fore
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import common


def do_svm_grid_search(X_train, y_train, X_val, y_val):
    """
    Perform a grid search for tuning SVM hyperparameters on the training data and evaluate the best model on the validation set.

    Args:
        X_train (numpy.ndarray): The feature matrix for training data.
        y_train (numpy.ndarray): The labels for the training data.
        X_val (numpy.ndarray): The feature matrix for validation data.
        y_val (numpy.ndarray): The labels for the validation data.

    Returns:
        None: Prints the best SVM parameters, the best F1 score from cross-validation, the best F1 score on the validation set, 
              and plots the results of the grid search.
    """
    # Define the parameter grid for different kernels
    param_grid = [
        {'kernel': ['linear'], 
         'C': [0.00001, 0.0001, 0.001, 0.1, 1, 10]
        },  # Linear kernel doesn't need 'degree' or 'gamma'
        {'kernel': ['rbf', 'sigmoid'], 
         'C': [0.00001, 0.0001, 0.001, 0.1, 1, 10], 
         'gamma': ['scale']
         },  # RBF and sigmoid kernels use 'gamma'
        {'kernel': ['poly'], 
         'C': [0.00001, 0.0001, 0.001, 0.1, 1, 10], 
         'degree': range(1,7), 
         'gamma': ['scale']
         }  # Polynomial uses 'degree' and 'gamma'
    ]

    # Initialize GridSearchCV with a support vector machine, grid search parameters, and scoring as F1
    grid_search = GridSearchCV(SVC(random_state=42), param_grid=param_grid, refit=True, verbose=2, scoring='f1')

    # Fit the grid search to the training data
    grid_search.fit(X=X_train, y=y_train)

    # Output the best parameters and the best F1 score during cross-validation
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Support Vector Machine (SVM)" + Fore.RESET + ":")
    print(f"\tBest Parameters: {grid_search.best_params_}")
    print(f"\tBest F1 Score (Grid Search Cross-Validation): {grid_search.best_score_}")

    # Get the best estimator (SVM model) and make predictions on the validation set
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X=X_val)

    # Compute the F1 score for the best model on the validation set
    f1_best = f1_score(y_true=y_val, y_pred=y_pred, average='macro')
    print(f"\tBest F1 Score (Actual Validation Set): {f1_best}")

    return grid_search.cv_results_


def svm_model(X_train, y_train, X_val, y_val):
    """
    Train an SVM model using the best parameters from grid search and evaluate it on the validation set.

    Args:
        X_train (numpy.ndarray): Training input features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation input features.
        y_val (numpy.ndarray): Validation labels.

    Returns:
        None
    """
    best_params = {
        'kernel': 'rbf',
        'C': 10,
        'gamma': 'scale'
    }

    # Create the SVM model with the best parameters
    svm_model = SVC(
        kernel=best_params['kernel'], 
        C=best_params['C'], 
        gamma=best_params.get('gamma', 'scale'),
        verbose=True,
        probability=True,
        random_state=42
    )

    # Train the SVM model
    svm_model.fit(X=X_train, y=y_train)

    # Make predictions on the validation set
    y_pred = svm_model.predict(X=X_val)

    # Calculate the F1 score on the validation set
    f1 = f1_score(y_true=y_val, y_pred=y_pred, average='macro')

    # Output the best parameters and the best F1 score
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Support Vector Machine (SVM)" + Fore.RESET + ":")
    print("\tBest Parameters for SVM:", best_params)
    print(f"\tF1 Score on Validation Set: {f1}")

    return svm_model


def svm_predict_extra(model, X_train, y_train, X_train_extra):
    """
    Predicts on the extra data, filters samples with probabilities above the threshold,
    and appends the best (most confident) samples to the training set, ensuring class balance.

    Args:
        model (SVC): The trained SVM model.
        X_train_extra (numpy.ndarray): The extra input data.
        y_train (numpy.ndarray): The original training labels.
        threshold (float): The probability threshold to consider for filtering.

    Returns:
        X_train (numpy.ndarray): The updated training input features (shuffled).
        y_train (numpy.ndarray): The updated training labels (shuffled).
    """
    THRESHOLD = 0.9

    # Get the predicted probabilities
    y_proba = model.predict_proba(X_train_extra)
    
    # Extract probabilities for class 0 and class 1
    prob_class_0 = y_proba[:, 0]  # Probabilities for class 0
    prob_class_1 = y_proba[:, 1]  # Probabilities for class 1

    # Select samples where the probability is higher than the threshold
    selected_class_0_idx = np.where(prob_class_0 >= THRESHOLD)[0]
    selected_class_1_idx = np.where(prob_class_1 >= THRESHOLD)[0]

    # Corresponding samples and labels
    selected_class_0 = X_train_extra[selected_class_0_idx]
    selected_class_1 = X_train_extra[selected_class_1_idx]

    # Balance the samples: take the minimum count between class 0 and class 1
    min_count = min(len(selected_class_0), len(selected_class_1))

    # Sort the samples by the highest probability and select the top samples
    sorted_class_0_idx = np.argsort(prob_class_0[selected_class_0_idx])[::-1]
    sorted_class_1_idx = np.argsort(prob_class_1[selected_class_1_idx])[::-1]

    # Get the top `min_count` samples
    best_class_0 = selected_class_0[sorted_class_0_idx[:min_count]]
    best_class_1 = selected_class_1[sorted_class_1_idx[:min_count]]

    # Corresponding labels
    best_labels_0 = np.zeros(len(best_class_0))
    best_labels_1 = np.ones(len(best_class_1))

    # Combine the selected samples and labels
    X_combined = np.vstack([best_class_0, best_class_1])
    y_combined = np.concatenate([best_labels_0, best_labels_1])

    # Append to the original dataset
    X_train_updated = np.vstack([X_train, X_combined])
    y_train_updated = np.concatenate([y_train, y_combined])

    # Shuffle the data using sklearn's shuffle
    X_train_shuffled, y_train_shuffled = shuffle(X_train_updated, y_train_updated, random_state=42)

    print("After " + Fore.YELLOW + "Data Augmentation (extra dataset)" + Fore.RESET + ":")
    common.get_imbalance(y=y_train_shuffled)

    return X_train_shuffled, y_train_shuffled


def plot_svm_history(grid_search_results):
    """
    Plots the F1 score as a function of regularization parameter C for different SVM kernels.

    Args:
        grid_search_results (dict): The results from a grid search, typically the `cv_results_` from a 
                                    GridSearchCV object. It should include 'param_C', 'param_kernel', 
                                    and 'mean_test_score' among other fields.

    Returns:
        None: Displays the plot showing the relationship between C, F1 score, and kernel.
    """
    # Convert grid search results to a pandas DataFrame for easier manipulation
    df_results = pd.DataFrame(grid_search_results)

    # Filter the relevant columns: 'param_C' (regularization strength), 'param_kernel' (SVM kernel type),
    # and 'mean_test_score' (mean F1 score from cross-validation)
    df_filtered = df_results[['param_C', 'param_kernel', 'mean_test_score']]

    # Get the unique SVM kernels used in the grid search
    kernels = df_filtered['param_kernel'].unique()

    # Loop through each kernel and plot the F1 score against the C parameter
    for kernel in kernels:
        # Filter the DataFrame for the current kernel
        df_kernel = df_filtered[df_filtered['param_kernel'] == kernel]
        
        # Plot F1 score as a function of the regularization parameter C for the current kernel
        plt.plot(df_kernel['param_C'], df_kernel['mean_test_score'], label=kernel)

    # Customize the plot
    plt.xlabel('C (Regularization parameter)')
    plt.ylabel('F1 Score')
    plt.legend()  # Add a legend to distinguish between kernels
    plt.xscale('log')  # Use logarithmic scale for C, as SVM parameters often vary across orders of magnitude
    plt.grid(True)  # Add gridlines for better readability
    plt.tight_layout()
    plt.show()  # Display the plot