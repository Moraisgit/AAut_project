import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_absolute_path, load_data, get_plot_save_path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler


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
    print(f"X_train_split: {X_train_split.shape}, y_train_split: {y_train_split.shape}")
    print(f"X_validation_split: {X_validation_split.shape}, y_validation_split: {y_validation_split.shape}")
    
    # Return the split datasets
    return X_train_split, X_validation_split, y_train_split, y_validation_split


def detect_outliers(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Detect and remove outliers from the training data based on Z-scores.

    Parameters:
    X_train (np.ndarray): The input features of the training data.
    y_train (np.ndarray): The target labels of the training data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Cleaned features and labels without outliers.
    """
    print("Initial shapes: " + str(X_train.shape) + str(y_train.shape))

    # Define the threshold for Z-score to identify outliers
    # Fine-tuned to lose the 25% of outliers - 200/4=50
    threshold = 1.87

    # Extract all columns from X_train
    cols = [X_train[:, i] for i in range(X_train.shape[1])]
    # Calculate the mean of each column
    means = [np.mean(col) for col in cols]
    # Calculate the standard deviation of each column
    stds = [np.std(col) for col in cols]
    # Calculate the Z-scores for each column
    z_scores = [(col - means[i]) / stds[i] for i, col in enumerate(cols)]

    # Identify rows with absolute Z-scores greater than the threshold
    rows_to_remove = set()  # Use a set to avoid duplicate indices
    for z_col in z_scores:
        for row_idx, z_score in enumerate(z_col):
            if np.abs(z_score) > threshold:
                rows_to_remove.add(row_idx)

    # Determine the indices of rows to keep (those not identified as outliers)
    rows_to_keep = list(set(range(X_train.shape[0])) - rows_to_remove)
    
    # Keep only the non-outlier rows in X_train
    X_clean = X_train[rows_to_keep, :]
    # Keep only the non-outlier rows in y_train
    y_clean = y_train[rows_to_keep]

    print("New shapes: " + str(X_clean.shape) + str(y_clean.shape) + "\n")
    
    return X_clean, y_clean


def lasso_regression(X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray):
    """
    Perform Lasso regression using scikit-learn's Lasso model.

    Parameters:
    X_train (np.ndarray): The independent variables.
    y_train (np.ndarray): The dependent variable.

    Returns:
    coefs (np.ndarray): Coefficients for each feature after Lasso regression.
    intercept (float): The intercept term for the Lasso regression.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X=X_train)
    X_validation = scaler.transform(X=X_validation)

    lasso = Lasso(alpha = 10)
    lasso.fit(X=X_train, y=y_train)
    train_score_ls =lasso.score(X=X_train, y=y_train)
    test_score_ls =lasso.score(X=X_validation, y=y_validation)

    print("The train score for ls model is {}".format(train_score_ls))
    print("The test score for ls model is {}".format(test_score_ls))

    # # Create Lasso regression model with the specified regularization parameter (alpha)
    # lasso_cv = LassoCV(alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10], random_state=0, max_iter=3000).fit(X=X_train, y=y_train)

    # train_score_ls = lasso_cv.score(X=X_train, y=y_train)
    # validation_score_ls = lasso_cv.score(X=X_validation, y=y_validation)

    # print("The train score for ls model is {}".format(train_score_ls))
    # print("The test score for ls model is {}".format(validation_score_ls))

    # Return the coefficients (slopes) and intercept
    return lasso.intercept_, lasso.coef_


def lasso_model(X_test: np.ndarray, intercept: float, coefs: np.ndarray) -> np.ndarray:
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
    y_pred = intercept + np.dot(a=X_test, b=coefs)

    return y_pred


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

def main():
    # Our output will be compared with the teachers output using SSE metric
    X_test = load_data(filename=get_absolute_path(file_name="X_test.npy"))  # Test data for the model
    y_train = load_data(filename=get_absolute_path(file_name="y_train.npy"))  # Expected output for the training data
    X_train = load_data(filename=get_absolute_path(file_name="X_train.npy"))  # Training data for the model

    X_clean, y_clean = detect_outliers(X_train=X_train, y_train=y_train)

    # plot_training_data(X_train=X_clean, individual_plots=True)

    X_train_split, X_validation_split, y_train_split, y_validation_split = generate_train_validation_data(
        X_train=X_clean,
        y_train=y_clean
    )

    # Estimate coefficients using Lasso regression
    intercept, coefficients = lasso_regression(
        X_train=X_train_split, 
        y_train=y_train_split, 
        X_validation=X_validation_split, 
        y_validation=y_validation_split
    )

    # Print the results
    print("Lasso Model parameters:")
    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coefficients}\n")

    y_pred = lasso_model(X_test=X_test, intercept=intercept, coefs=coefficients)
    print("Y predicted:")
    print(y_pred)

if __name__ == "__main__":
    main()
