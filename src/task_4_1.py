import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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


def plot_training_data(X_train: np.ndarray) -> None:
    samples = np.arange(
        0, 200, 1
    )  # Sample range (assuming X_train has at least 200 rows)
    # Create subplots with the desired layout
    figure, axis = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows and 2 columns
    figure.suptitle("Scatter plots of the independent variables", fontsize=16)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between plots

    # Define titles for the subplots
    titles_X = [
        "Daily Averages of Air Temperature (x1)",
        "Water Temperature (x2)",
        "Wind Speed (x3)",
        "Wind Direction (x4)",
        "Illumination (x5)",
    ]

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
    plt.savefig("../plots/scatter_plots.png")
    print("Figure saved as scatter_plots.png")


def main():
    # Our output will be compared with the teachers output using SSE metric
    # X_test = load_data("../data/X_test.npy")  # Test data for the model

    # y_train = load_data("../data/y_train.npy")  # Expected output for the training data

    X_train = load_data("../data/X_train.npy")  # Training data for the model

    plot_training_data(X_train=X_train)


if __name__ == "__main__":
    main()
