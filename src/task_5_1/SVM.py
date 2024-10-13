from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from colorama import Fore
import matplotlib.pyplot as plt


def do_svm_grid_search(X_train, y_train, X_val, y_val):
    # Define the parameter grid
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

    # Set up GridSearchCV
    grid_search = GridSearchCV(SVC(), param_grid=param_grid, refit=True, verbose=2, scoring='f1')

    # Fit to the training data
    grid_search.fit(X=X_train, y=y_train)

    # Best parameters and score
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Support Vector Machine (SVM)" + Fore.RESET + ":")
    print(f"\tBest Parameters: {grid_search.best_params_}")
    print(f"\tBest F1 Score (Grid Search Cross-Validation): {grid_search.best_score_}")

    # Make predictions with the best estimator
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_val)
    f1_best = f1_score(y_val, y_pred)
    print(f"\tBest F1 Score (Actual Validation Set): {f1_best}")

    plot_svm_history(grid_search_results=grid_search.cv_results_)


def plot_svm_history(grid_search_results):
    # Convert to DataFrame for easier handling
    df_results = pd.DataFrame(grid_search_results)

    # Filter out the relevant columns (mean F1 score, C, and kernel)
    df_filtered = df_results[['param_C', 'param_kernel', 'mean_test_score']]

    # Create separate plots for each kernel
    kernels = df_filtered['param_kernel'].unique()
    for kernel in kernels:
        # Filter data for the current kernel
        df_kernel = df_filtered[df_filtered['param_kernel'] == kernel]
        
        # Plot F1 score as a function of C
        plt.plot(df_kernel['param_C'], df_kernel['mean_test_score'], label=kernel)

    # Customize plot
    plt.xlabel('C')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.xscale('log')  # Since C values are often in a logarithmic scale
    plt.grid(True)
    plt.show()

