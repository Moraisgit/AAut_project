"""
This module includes commonly used functions across other files.
"""

import numpy as np
import os
from colorama import init, Fore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_imbalance(y, do_print=True):
    """
    Calculates the class distribution of the dataset and optionally prints it.

    Args:
        y (numpy.ndarray): Labels array where 0 indicates 'No crater' and 1 indicates 'Crater'.
        do_print (bool, optional): If True, prints the class imbalance. If False, returns the absolute difference between the two classes.

    Returns:
        int or None: The absolute difference between the number of 'Crater' and 'No crater' samples if `do_print` is False. Otherwise, None.
    """
    num_no_craters = len([0 for i in range(y.shape[0]) if y[i] == 0])
    num_craters = len([1 for i in range(y.shape[0]) if y[i] == 1])
    total = num_no_craters + num_craters

    if do_print:
        print("\tCheck imbalance:")
        print(f"\tClass 0 (No crater): {num_no_craters} ({num_no_craters / total * 100:.2f} %)")
        print(f"\tClass 1 (Crater): {num_craters} ({num_craters / total * 100:.2f} %)")
    else:
        return abs(num_craters - num_no_craters)


def plot_dataset(X, y, num_images=16):
    """
    Plots a grid of images with their corresponding labels.

    Args:
        X (numpy.ndarray): Input feature array of shape (n_samples, 2304) where each entry is an image.
        y (numpy.ndarray): Labels array corresponding to the input features.
        num_images (int, optional): The number of images to display. Default is 16.

    Returns:
        None: This function displays the plot and does not return any value.
    """
    num_images = min(num_images, len(y))  # Ensure we don't plot more than available
    subplot = int(np.ceil(np.sqrt(num_images)))  # Calculate grid size (rows and columns)
    
    # Automatically calculate the figure size based on the number of images
    plt.figure(figsize=(subplot * 2, subplot * 2))  # Each subplot will be roughly 2x2 inches

    for i in range(num_images):
        plt.subplot(subplot, subplot, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(
            X[i, :].reshape(48, 48),  # Reshape to (48, 48)
            origin="lower",
            cmap="gray",              # Use grayscale colormap for better visibility
            interpolation="nearest",
        )
        plt.title(f"Label: {y[i]}")  # Display the label for each image
    plt.tight_layout()
    plt.show()


def load_all_data():
    """
    Loads the training and test datasets from `.npy` files.

    Args:
        None
    
    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): Training input features.
            - Y_train (numpy.ndarray): Training labels.
            - X_train_extra (numpy.ndarray): Additional training input features.
            - X_test (numpy.ndarray): Test input features.
    """
    # Load training data
    X_train = np.load(file="Xtrain1.npy")
    
    # Load training labels
    Y_train = np.load(file="Ytrain1.npy")
    
    # Load extra training data
    X_train_extra = np.load(file="Xtrain1_extra.npy")
    
    # Load test data
    X_test = np.load(file="Xtest1.npy")
    
    return X_train, Y_train, X_train_extra, X_test


def save_npy_to_output(file_name: str, data: np.array) -> None:
    """
    Save a NumPy array to a .npy file in the 'output' directory of the project.

    Parameters:
    file_name (str): The name to save the file as, including the .npy extension.
    data (np.array): The NumPy array to be saved.
    
    Returns:
    None
    """
    # Initialize colorama to use colored output in the terminal
    init()

    # Get the project root directory by navigating one level up from the current script's directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Construct the path to the 'output' directory within the project root
    output_dir = os.path.join(project_root, "output")

    # If the 'output' directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full file path by combining the output directory and the provided file name
    file_path = os.path.join(output_dir, file_name)

    # Save the provided NumPy array to a .npy file at the constructed file path
    np.save(file_path, data)

    # Print the full path to the saved file using blue text for the file path
    print(f"File saved at: {Fore.BLUE}{file_path}{Fore.RESET}")


def get_plot_save_path(image_name: str) -> str:
    """
    Generate the absolute path for saving a plot in the 'plots' directory.
    Assumes that the 'plots' directory is at the same level as the 'src' directory.

    Parameters:
    image_name (str): The name of the image file to save.

    Returns:
    str: The absolute path to the specified image file in the 'plots' directory.
    """
    # Get the project root directory
    project_root = os.path.abspath(path=os.path.join(os.path.dirname(__file__), "../.."))

    # Construct the path to the 'plots' directory
    plots_dir = os.path.join(project_root, "plots")

    # Construct the full file path
    file_path = os.path.join(plots_dir, image_name)

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    return file_path


def split_data(X, y):
    """
    Splits the training data into training, validation, and test datasets.

    Args:
        X (np.ndarray): The original training data.
        y (np.ndarray): The original training data.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    # First, split off 80% for training and 20% for validation + test
    X_train_split, X_tmp_split, y_train_split, y_tmp_split = train_test_split(
        X, 
        y, 
        test_size=0.2,  # 20% for validation and test
        shuffle=True,   # Shuffle the data
        random_state=42, # Ensure reproducibility
        stratify=y # Stratify based on labels to ensure class balance
    )

    # Now, split the remaining 20% into 10% validation and 10% test
    X_val_split, X_test_split, y_val_split, y_test_split = train_test_split(
        X_tmp_split, 
        y_tmp_split, 
        test_size=0.5,
        shuffle=True,
        random_state=42
    )

    return X_train_split, y_train_split, X_val_split, y_val_split, X_test_split, y_test_split