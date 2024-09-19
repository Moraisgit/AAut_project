import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.linear_model import Ridge
from utils import get_absolute_path, load_data, get_plot_save_path


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


def remove_outliers_iqr(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    print("Initial shapes: " + str(X.shape) + str(y.shape))

    X = pd.DataFrame(X)
    y = pd.Series(y)

    # INitialize the list to store the indices of the outlierrs
    outliers_indices = []

    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Detect outliers
        outliers = X[(X[column] < lower) | (X[column] > upper)].index
        outliers_indices.extend(outliers)

    # Remove duplicates in outliers indices
    outliers_indices = list(set(outliers_indices))

    # Drop the outlier rows from both X and y
    X_clean = X.drop(index=outliers_indices).reset_index(drop=True)
    y_clean = y.drop(index=outliers_indices).reset_index(drop=True)

    print("New shapes: " + str(X_clean.shape) + str(y_clean.shape) + "\n")

    return X_clean, y_clean


def ridge_regression(
    X_train: np.ndarray, y_train: np.ndarray, alpha: float
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
    # Create Ridge regression model with the specified regularization parameter (alpha)
    ridge_model = Ridge(alpha=alpha, fit_intercept=True)

    # Fit the model on the training data
    ridge_model.fit(X_train, y_train)

    # Get the coefficients (slopes) and intercept
    coefs = ridge_model.coef_
    intercept = ridge_model.intercept_

    return intercept, coefs


# Function to create predictions for new data points
def ridge_model(X_new: np.ndarray, intercept: float, coefs: np.ndarray) -> np.ndarray:
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

    X_clean, y_clean = remove_outliers_iqr(X_train, y_train)

    plot_training_data(X_train=X_clean, individual_plots=True)

    # Regularization parameter
    alpha = 1.0

    # Estimate coefficients using Ridge regression
    intercept, coefficients = ridge_regression(X_clean, y_clean, alpha)

    # Print the results
    print("Ridge Model parameters:")
    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coefficients}\n")

    y_pred = ridge_model(X_test, intercept, coefficients)
    print("Y predicted:")
    print(y_pred)


if __name__ == "__main__":
    main()
