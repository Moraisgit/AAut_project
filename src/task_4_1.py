import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

def generate_train_validation_data(X_train, y_train):
    X_train_split, X_validation_split, y_train_split, y_validation_split = (
        train_test_split(
            X_train,  # Input features
            y_train,  # Target labels corresponding to X_train
            test_size=0.2,  # 20% of the data will go to validation set
            shuffle=True,  # Shuffle the data before splitting
            random_state=42,  # Ensure reproducibility of the split (same result each time you run the code)
        )
    )
    return X_train_split, X_validation_split, y_train_split, y_validation_split


def load_data(filename: str) -> np.ndarray:
    return np.load(file=filename)

def detect_outliers(X_train: np.ndarray):
    threshold = 3
    
    cols = [X_train[:, i] for i in range(5)]  # Extract the first 5 columns
    means = [np.mean(col) for col in cols] # Calculate the mean of each column
    stds = [np.std(col) for col in cols] # Calculate the of each column

    # Calculate the Z-scores for each column
    z_scores = [(col - means[i]) / stds[i] for i, col in enumerate(cols)]
    
    
def plot_training_data(X_train: np.ndarray, individual_plots: bool = False) -> None:
    samples = np.arange(0, X_train.shape[0], 1)  # Sample range X_train.shape[0]: number of rows
    titles_X = [
        "Daily Averages of Air Temperature (x1)",
        "Water Temperature (x2)",
        "Wind Speed (x3)",
        "Wind Direction (x4)",
        "Illumination (x5)",
    ]
    
    # Check if X_train is a DataFrame or numpy array
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values  # Convert to NumPy array for consistent indexing

    if individual_plots:
        # Save individual plots for each feature
        for i in range(5):
            plt.figure(figsize=(6, 4))
            plt.scatter(samples, X_train[:, i], color="blue", s=10)
            plt.title(titles_X[i])
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)  # Add gridlines for better readability
            plt.savefig(f"../plots/plot_{i+1}.png")  # Save each plot separately
            plt.close()  # Close the plot after saving to free up memory
        print("Individual plots saved as plot_1.png, plot_2.png, ..., plot_5.png")
    else:
        # Create subplots with the desired layout (as in the original function)
        figure, axis = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows and 2 columns
        figure.suptitle("Scatter plots of the independent variables after IQR outlier removal", fontsize=16)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between plots
        
        # Plot the first 4 figures in a 2x2 grid
        for i in range(4):
            row = i // 2
            col = i % 2
            axis[row, col].scatter(samples, X_train[:, i], color="blue", s=10)
            axis[row, col].set_title(titles_X[i])
            axis[row, col].set_xlabel("Sample Index")
            axis[row, col].set_ylabel("Value")
            axis[row, col].grid(True)  # Add gridlines for better readability

        # Remove the unused subplot (axis[2,1])
        figure.delaxes(axis[2, 1])

        # Plot the last figure (centered in the last row)
        axis[2, 0].scatter(samples, X_train[:, 4], color="blue", s=10)
        axis[2, 0].set_title(titles_X[4])
        axis[2, 0].set_xlabel("Sample Index")
        axis[2, 0].set_ylabel("Value")
        axis[2, 0].grid(True)  # Add gridlines for better readability

        # Save the figure to a file
        plt.savefig("../plots/Cleaned_data.png")
        print("Figure saved as Cleaned_data.png")


def remove_outliers_iqr(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    print("Initial shapes: "+ str(X.shape) + str(y.shape))
    
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # INitialize the list to store the indices of the outlierrs
    outliers_indices = []

    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 +1.5*IQR
        
        # Detect outliers
        outliers = X[(X[column] < lower) | (X[column] > upper)].index
        outliers_indices.extend(outliers)
    
    # Remove duplicates in outliers indices
    outliers_indices = list(set(outliers_indices))

    # Drop the outlier rows from both X and y
    X_clean = X.drop(index=outliers_indices).reset_index(drop=True)
    y_clean = y.drop(index=outliers_indices).reset_index(drop=True)
    
    print("NEw shapes: " + str(X_clean.shape) + str(y_clean.shape))

    return X_clean, y_clean

def main():
    # Our output will be compared with the teachers output using SSE metric
    # X_test = load_data("../data/X_test.npy")  # Test data for the model

    y_train = load_data("../data/y_train.npy")  # Expected output for the training data

    X_train = load_data("../data/X_train.npy")  # Training data for the model

    # plot_training_data(X_train=X_train)

    detect_outliers(X_train=X_train)

    X_clean, y_clean = remove_outliers_iqr(X_train, y_train)
    
    plot_training_data(X_train=X_clean, individual_plots=True)

if __name__ == "__main__":
    main()
