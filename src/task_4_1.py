import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from colorama import init as colorama_init
from colorama import Fore, Style
from typing import Tuple
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, RANSACRegressor, LinearRegression
from utils import get_absolute_path, load_data, get_plot_save_path, save_npy_to_output
from sklearn.model_selection import train_test_split

def generate_train_validation_data(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Split the training data into training and validation sets.

    Parameters:
    X_train (np.ndarray): The input features for training.
    y_train (np.ndarray): The target labels corresponding to X_train.

    Returns:
    tuple: Four elements: 
        - X_train_split (np.ndarray): Training features after the split.
        - X_validation_split (np.ndarray): Validation features after the split.
        - y_train_split (np.ndarray): Training labels after the split.
        - y_validation_split (np.ndarray): Validation labels after the split.
    """
    # Split the data into training and validation sets
    X_train_split, X_validation_split, y_train_split, y_validation_split = (
        train_test_split(
            X_train,  # Input features
            y_train,  # Target labels corresponding to X_train
            test_size=0.2,  # 20% of the data will go to validation set
            shuffle=True,  # Shuffle the data before splitting for randomness
            random_state=42,  # Ensure reproducibility of the split
        )
    )

    # Print the shapes of the split data
    print("Split shapes:")
    print(f"\tX_train_split: {Fore.CYAN}{X_train_split.shape}{Fore.RESET}, y_train_split: {Fore.CYAN}{y_train_split.shape}{Fore.RESET}")
    print(f"\tX_validation_split: {Fore.MAGENTA}{X_validation_split.shape}{Fore.RESET}, y_validation_split: {Fore.MAGENTA}{y_validation_split.shape}{Fore.RESET}")
    # Return the split datasets
    return X_train_split, X_validation_split, y_train_split, y_validation_split


def remove_outliers_with_ransac(X_train: np.ndarray, y_train: np.ndarray):
    """
    Use RANSAC to fit a model and remove outliers from the dataset.

    Parameters:
    X (np.ndarray): The independent variables (features).
    y (np.ndarray): The dependent variable (target).

    Returns:
    X_inliers (np.ndarray): The filtered independent variables without outliers.
    y_inliers (np.ndarray): The filtered dependent variable without outliers.
    X_outliers (np.ndarray): The outlier independent variables.
    y_outliers (np.ndarray): The outlier dependent variables.
    """
    # Create the RANSAC model with a base estimator (e.g., LinearRegression)
    ransac = RANSACRegressor(LinearRegression(), random_state=0)

    # Fit the RANSAC model to the data
    ransac.fit(X_train, y_train)

    # Get a mask of inliers and outliers
    inlier_mask = ransac.inlier_mask_  # True for inliers, False for outliers

    # Separate inliers and outliers
    X_inliers = X_train[inlier_mask]
    y_inliers = y_train[inlier_mask]
    X_outliers = X_train[~inlier_mask]
    # y_outliers = y_train[~inlier_mask]

    # Print some stats
    print(f"Number of inliers: {Fore.GREEN}{len(X_inliers)}{Fore.RESET}")
    print(f"Number of outliers: {Fore.RED}{len(X_outliers)}{Fore.RESET}")
    print()

    return X_inliers, y_inliers


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
    samples = np.arange(0, X_train.shape[0], 1)
    
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
            plt.scatter(samples, X_train[:, i], color="blue", s=10)  # Scatter plot
            plt.title(titles_X[i])  # Set the title for the plot
            plt.xlabel("Sample Index")  # X-axis label
            plt.ylabel("Value")  # Y-axis label
            plt.grid(True)  # Add gridlines for better readability
            
            # Save each plot separately using the image name
            plt.savefig(get_plot_save_path(image_name=f"plot_{i+1}.png"))
            plt.close()  # Close the plot after saving to free up memory

        print("Individual plots saved as plot_1.png, plot_2.png, ..., plot_5.png")
        print()
    else:
        # Create subplots for all features in a specified layout
        figure, axis = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows and 2 columns
        figure.suptitle(
            "Scatter plots of the independent variables after IQR outlier removal",
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
        figure.delaxes(axis[2, 1])

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


def regression(
    X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray, regression_technique: str
) -> Tuple[float, np.ndarray]:
    """
    Perform Ridge regression using scikit-learn's Ridge model.

    Parameters:
    X_train (np.ndarray): The independent variables.
    y_train (np.ndarray): The dependent variable.
    alpha (float): The regularization parameter

    Returns:
    coefs (np.ndarray): Coefficients for each feature after Ridge regression.
    intercept (float): The intercept term for the Ridge regression.
    """
    if regression_technique == "ElasticNetCV":
        regression = ElasticNetCV(alphas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10], random_state=42, max_iter=3000, fit_intercept=True).fit(X=X_train, y=y_train)
    elif regression_technique == "RidgeCV":
        regression = RidgeCV(alphas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10], fit_intercept=True).fit(X=X_train, y=y_train)
    elif regression_technique == "LassoCV":
        regression = LassoCV(alphas = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10], random_state=42, max_iter=3000, fit_intercept=True).fit(X=X_train, y=y_train)
    
    train_score = regression.score(X=X_train, y=y_train)
    validation_score = regression.score(X=X_validation, y=y_validation)
    # Printing with colors
    print("\nUsing " + Fore.YELLOW + regression_technique + Fore.RESET + ":")
    print("\tThe train score is: {}{}{}".format(Fore.GREEN, train_score, Fore.RESET))
    print("\tThe validation score is: {}{}{}".format(Fore.BLUE, validation_score, Fore.RESET))

    return regression.intercept_, regression.coef_


def toxic_algae_model(X_new: np.ndarray, intercept: float, coefs: np.ndarray) -> np.ndarray:
    """
    Predict the target values based on the linear regression model.

    Parameters:
    X_new (np.ndarray): New data points (n_samples, n_features).
    intercept (float): The intercept term from the model.
    coefs (np.ndarray): The coefficients (slopes) for each feature from the model.

    Returns:
    y_pred (np.ndarray): Predicted target values.
    """
    # Calculate the predicted values: ŷ = β0 + Σ (βi * Xi)
    y_pred = intercept + np.dot(X_new, coefs)

    return y_pred


def main():
    # Our output will be compared with the teachers output using SSE metric
    X_test = load_data(filename=get_absolute_path("X_test.npy"))  # Test data for the model
    y_train = load_data(filename=get_absolute_path("y_train.npy"))  # Expected output for the training data
    X_train = load_data(filename=get_absolute_path("X_train.npy"))  # Training data for the model

    X_clean, y_clean = remove_outliers_with_ransac(X_train=X_train, y_train=y_train)
    
    plot_training_data(X_train=X_clean, individual_plots=True)

    X_train_split, X_validation_split, y_train_split, y_validation_split = generate_train_validation_data(
        X_train=X_clean,
        y_train=y_clean
    )

    regression_techniques = ["RidgeCV", "LassoCV", "ElasticNetCV"]

    # Estimate coefficients using Ridge regression
    intercept, coefficients = regression(X_train=X_train_split, 
        y_train=y_train_split, 
        X_validation=X_validation_split, 
        y_validation=y_validation_split,
        regression_technique=regression_techniques[0]
    )

    # Print the results
    print("\nModel parameters:")
    print(f"\tIntercept: {Fore.GREEN}{intercept}{Fore.RESET}")
    print(f"\tCoefficients: {Fore.GREEN}{coefficients}{Fore.RESET}\n")

    y_pred = toxic_algae_model(X_test, intercept, coefficients)
    save_npy_to_output("y_pred.npy", y_pred)


if __name__ == "__main__":
    main()