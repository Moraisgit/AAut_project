import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore
from typing import Tuple
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    Ridge,
    RANSACRegressor,
    LinearRegression,
)
from utils import get_absolute_path, load_data, get_plot_save_path, save_npy_to_output
from sklearn.model_selection import cross_validate


def plot_training_data(X_train: np.ndarray, individual_plots: bool = False) -> None:
    """
    Plot training data features as scatter plots.

    Parameters:
    X_train (np.ndarray): The training data with features to be plotted.
    individual_plots (bool): If True, save individual plots for each feature;
                             if False, create subplots for all features.

    Returns:
    None: This function saves the plots to files and does not return any value.
    """
    # Sample range corresponding to the number of rows in X_train
    samples = np.arange(start=0, stop=X_train.shape[0], step=1)

    # Titles for each feature plot
    titles_X = [
        "Daily Averages of Air Temperature (x1)",
        "Water Temperature (x2)",
        "Wind Speed (x3)",
        "Wind Direction (x4)",
        "Illumination (x5)",
    ]

    # Check if X_train is a DataFrame and convert to NumPy array if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values  # Convert to NumPy array for consistent indexing

    if individual_plots:
        # Save individual plots for each feature
        for i in range(5):
            plt.figure(figsize=(6, 4))  # Create a new figure for each plot
            plt.scatter(x=samples, y=X_train[:, i], color="blue", s=10)  # Scatter plot
            plt.title(label=titles_X[i])  # Set the title for the plot
            plt.xlabel(xlabel="Sample Index")  # X-axis label
            plt.ylabel(ylabel="Value")  # Y-axis label
            plt.grid(visible=True)  # Add gridlines for better readability

            # Save each plot separately using the image name
            plt.savefig(get_plot_save_path(image_name=f"plot_{i+1}.png"))
            plt.close()  # Close the plot after saving to free up memory

        print("Individual plots saved as plot_1.png, plot_2.png, ..., plot_5.png")
        print()
    else:
        # Create subplots for all features in a specified layout
        figure, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))  # 3 rows and 2 columns
        figure.suptitle(
            t="Scatter plots of the independent variables after IQR outlier removal",
            fontsize=16,
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between plots

        # Plot the first 4 figures in a 2x2 grid
        for i in range(4):
            row = i // 2
            col = i % 2
            axis[row, col].scatter(samples, X_train[:, i], color="blue", s=10)
            axis[row, col].set_title(titles_X[i])  # Set the title for the subplot
            axis[row, col].set_xlabel("Sample Index")  # X-axis label
            axis[row, col].set_ylabel("Value")  # Y-axis label
            axis[row, col].grid(True)  # Add gridlines for better readability

        # Remove the unused subplot (axis[2,1])
        figure.delaxes(ax=axis[2, 1])

        # Plot the last figure (centered in the last row)
        axis[2, 0].scatter(samples, X_train[:, 4], color="blue", s=10)
        axis[2, 0].set_title(titles_X[4])  # Set title for the last subplot
        axis[2, 0].set_xlabel("Sample Index")  # X-axis label
        axis[2, 0].set_ylabel("Value")  # Y-axis label
        axis[2, 0].grid(True)  # Add gridlines for better readability

        # Save the entire figure to a file
        plt.savefig(get_plot_save_path(image_name="Cleaned_data.png"))
        print("Figure saved as Cleaned_data.png")
        print()


def remove_outliers_with_ransac(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use RANSAC to fit a model and remove outliers from the dataset.

    Parameters:
    X_train (np.ndarray): The independent variables (features).
    y_train (np.ndarray): The dependent variable (target).

    Returns:
    Tuple[np.ndarray, np.ndarray]: 
        - X_inliers: The filtered independent variables without outliers.
        - y_inliers: The filtered dependent variable without outliers.
        - X_outliers: The outlier independent variables.
        - y_outliers: The outlier dependent variables.
    """
    # Create the RANSAC model with a base estimator (e.g., LinearRegression)
    ransac = RANSACRegressor(estimator=LinearRegression())

    # Fit the RANSAC model to the training data
    ransac.fit(X=X_train, y=y_train)

    # Get a mask of inliers and outliers
    inlier_mask = ransac.inlier_mask_  # True for inliers, False for outliers

    # Separate inliers and outliers
    X_inliers = X_train[inlier_mask]
    y_inliers = y_train[inlier_mask]
    X_outliers = X_train[~inlier_mask]
    # y_outliers = y_train[~inlier_mask]

    # Print statistics about inliers and outliers
    print(f"Number of inliers: {Fore.BLUE}{len(X_inliers)}{Fore.RESET}")
    print(f"Number of outliers: {Fore.BLUE}{len(X_outliers)}{Fore.RESET}")
    print()

    return X_inliers, y_inliers


def toxic_algae_model(
    X_test: np.ndarray, intercept: float, coefs: np.ndarray
) -> np.ndarray:
    """
    Predict the target values based on the linear regression model.

    Parameters:
    X_test (np.ndarray): New data points (n_samples, n_features).
    intercept (float): The intercept term from the model.
    coefs (np.ndarray): The coefficients (slopes) for each feature from the model.

    Returns:
    np.ndarray: Predicted target values.
    """
    # Calculate the predicted values: ŷ = β0 + Σ (βi * Xi)
    y_pred = intercept + np.dot(X_test, coefs)

    return y_pred


def shuffle_data(X, y):
    """
    Shuffle the dataset X and corresponding labels y while maintaining their row correspondence.

    Parameters:
    X (np.ndarray): The input features for training.
    y (np.ndarray): The target labels corresponding to X_train.

    Returns:
    Tuple[np.ndarray, np.ndarray]: 
        X_shuffled: The shuffled input features, maintaining the same shape as X.
        y_shuffled: The shuffled labels, maintaining the same shape as y.
    """
    # Horizontally stack X and y into a single 2D array
    data = np.hstack([X, y.reshape(-1, 1)])  # Ensure y is a column vector
    
    # Shuffle the rows of the combined data to randomize the order of samples
    np.random.shuffle(data)
    
    # Separate the shuffled data back into X and y
    X_shuffled = data[:, :-1]   # All columns except the last one (original X)
    y_shuffled = data[:, -1:]    # Only the last column (original y)
    
    return X_shuffled, y_shuffled


def compare_models(X_clean, y_clean):
    NUM_FOLDS = 5
    lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
    l1_ratio = [0.1, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    X_clean_shuffled, y_clean_shuffled = shuffle_data(X=X_clean, y=y_clean)

    ########################
    # Test Linear Regression
    ########################
    linear_reg = LinearRegression().fit(X=X_clean_shuffled, y=y_clean_shuffled)
    linear_reg_scores = cross_validate(
        estimator=linear_reg, 
        X=X_clean_shuffled,
        y=y_clean_shuffled,
        cv=NUM_FOLDS
    )["test_score"]

    print("---------------------------")
    print("Using " + Fore.YELLOW + "Linear Regression" + Fore.RESET + ":")
    print(f"\tAverage score (R²) = {sum(linear_reg_scores)/NUM_FOLDS}")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {linear_reg.intercept_}\n\t\tCoefficients: {linear_reg.coef_}")

    #######################
    # Test Ridge Regression
    #######################
    ridge_reg_avg_scores = []
    for alpha in lambdas:
        ridge_reg = Ridge(alpha=alpha, fit_intercept=True).fit(X=X_clean_shuffled, y=y_clean_shuffled)
        ridge_reg_scores = cross_validate(
            estimator=ridge_reg,
            X=X_clean_shuffled,
            y=y_clean_shuffled,
            cv=NUM_FOLDS
        )["test_score"]
        ridge_reg_avg_scores.append(sum(ridge_reg_scores)/NUM_FOLDS)
    max_ridge_avg_scores = max(ridge_reg_avg_scores)
    max_ridge_lambda = lambdas[ridge_reg_avg_scores.index(max_ridge_avg_scores)]

    ridge_reg = Ridge(alpha=max_ridge_lambda, fit_intercept=True).fit(X=X_clean_shuffled, y=y_clean_shuffled)
    cross_validate(
        estimator=ridge_reg,
        X=X_clean_shuffled,
        y=y_clean_shuffled,
        cv=NUM_FOLDS
    )
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Ridge Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_ridge_avg_scores} (lambda = {max_ridge_lambda})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {ridge_reg.intercept_}\n\t\tCoefficients: {ridge_reg.coef_}")

    #######################
    # Test Lasso Regression
    #######################
    lasso_reg_avg_scores = []
    for alpha in lambdas:
        lasso_reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000).fit(X=X_clean_shuffled, y=y_clean_shuffled)
        lasso_reg_scores = cross_validate(
            estimator=lasso_reg,
            X=X_clean_shuffled,
            y=y_clean_shuffled,
            cv=NUM_FOLDS
        )["test_score"]
        lasso_reg_avg_scores.append(sum(lasso_reg_scores)/NUM_FOLDS)
    max_lasso_avg_scores = max(lasso_reg_avg_scores)
    max_lasso_lambda = lambdas[lasso_reg_avg_scores.index(max_lasso_avg_scores)]

    lasso_reg = Lasso(alpha=max_lasso_lambda, fit_intercept=True, max_iter=5000).fit(X=X_clean_shuffled, y=y_clean_shuffled)
    cross_validate(
        estimator=lasso_reg,
        X=X_clean_shuffled,
        y=y_clean_shuffled,
        cv=NUM_FOLDS
    )
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Lasso Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_lasso_avg_scores} (lambda = {max_lasso_lambda})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {lasso_reg.intercept_}\n\t\tCoefficients: {lasso_reg.coef_}")

    ############################
    # Test ElasticNet Regression
    ############################
    elastic_net_reg_avg_scores = []
    max_elastic_net_avg_scores = 0
    for alpha in lambdas:
        for ratio in l1_ratio:
            elastic_net_reg = ElasticNet(alpha=alpha, fit_intercept=True, l1_ratio=ratio, max_iter=6000).fit(X=X_clean_shuffled, y=y_clean_shuffled)
            elastic_net_reg_scores = cross_validate(
                estimator=elastic_net_reg,
                X=X_clean_shuffled,
                y=y_clean_shuffled,
                cv=NUM_FOLDS
            )["test_score"]
            elastic_net_reg_avg_scores.append(sum(elastic_net_reg_scores)/NUM_FOLDS)
            if max(elastic_net_reg_avg_scores) > max_elastic_net_avg_scores:
                max_elastic_net_lambda = alpha
                max_elastic_ratio = ratio

    elastic_net_reg = ElasticNet(
        alpha=max_elastic_net_lambda, 
        fit_intercept=True, 
        l1_ratio=max_elastic_ratio, 
        max_iter=6000
    ).fit(X=X_clean_shuffled, y=y_clean_shuffled)
    cross_validate(
        estimator=elastic_net_reg,
        X=X_clean_shuffled,
        y=y_clean_shuffled,
        cv=NUM_FOLDS
    )
    print("---------------------------")
    print("Using " + Fore.YELLOW + "ElasticNet Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max(elastic_net_reg_avg_scores)} (lambda = {max_elastic_net_lambda}, l1_ratio = {max_elastic_ratio})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {elastic_net_reg.intercept_}\n\t\tCoefficients: {elastic_net_reg.coef_}")

    ############################
    # Final comparison
    ############################
    r_squared = [sum(linear_reg_scores)/NUM_FOLDS, max_ridge_avg_scores, max_lasso_avg_scores, max_elastic_net_avg_scores]
    if max(r_squared) == sum(linear_reg_scores)/NUM_FOLDS:
        print("---------------------------")
        print(f"Best model is Linear Regression with R² = {sum(linear_reg_scores)/NUM_FOLDS}")
    elif max(r_squared) == max_ridge_avg_scores:
        print("---------------------------")
        print(f"Best model is Ridge Regression with R² = {max_ridge_avg_scores}")
    elif max(r_squared) == max_lasso_avg_scores:
        print("---------------------------")
        print(f"Best model is Lasso Regression with R² = {max_lasso_avg_scores}")
    elif max(r_squared) == max_elastic_net_avg_scores:
        print("---------------------------")
        print(f"Best model is ElasticNet Regression with R² = {max_elastic_net_avg_scores}")

    # plt.plot(lambdas, ridge_reg_avg_scores, label="Ridge regression")
    # plt.legend()
    # plt.figure()
    # plt.plot(lambdas, lasso_reg_avg_scores, label="Lasso regression")
    # plt.legend()
    # plt.show()


def chosen_model(X_clean, y_clean):
    NUM_FOLDS = 5
    ALPHA = 10
    X_clean_shuffled, y_clean_shuffled = shuffle_data(X=X_clean, y=y_clean)

    #######################
    # Chosen model is Ridge
    #######################
    ridge_reg = Ridge(alpha=ALPHA, fit_intercept=True).fit(X=X_clean_shuffled, y=y_clean_shuffled)
    ridge_reg_scores = cross_validate(
        estimator=ridge_reg,
        X=X_clean_shuffled,
        y=y_clean_shuffled,
        cv=NUM_FOLDS
    )    
    max_ridge_scores = max(ridge_reg_scores)
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Ridge Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_ridge_scores} (lambda = {ALPHA})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {ridge_reg.intercept_}\n\t\tCoefficients: {ridge_reg.coef_}")

    coefs = np.array(ridge_reg.coef_).flatten()
    intercept = ridge_reg.intercept_[0]

    return intercept, coefs


def main():
    """
    Main function to execute the workflow for training and evaluating the model.
    It handles loading data, removing outliers, splitting data, and fitting a regression model.
    """
    # Our output will be compared with the teachers output using SSE metric
    X_test = load_data(
        filename=get_absolute_path("X_test.npy")
    )  # Test data for the model
    y_train = load_data(
        filename=get_absolute_path("y_train.npy")
    )  # Expected output for the training data
    X_train = load_data(
        filename=get_absolute_path("X_train.npy")
    )  # Training data for the model

    # Remove outliers from training data
    X_clean, y_clean = remove_outliers_with_ransac(X_train=X_train, y_train=y_train)

    # # Plot the cleaned training data
    plot_training_data(X_train=X_clean, individual_plots=True)

    compare_models(X_clean=X_clean, y_clean=y_clean)

    # intercept, coefficients = chosen_model(X_clean=X_clean, y_clean=y_clean)

    # # Predict using the trained model
    # y_pred = toxic_algae_model(X_test=X_test, intercept=intercept, coefs=coefficients)

    # # Save the predictions to a file
    # save_npy_to_output(file_name="y_pred.npy", data=y_pred)


if __name__ == "__main__":
    main()
