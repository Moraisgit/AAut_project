"""
This module includes commonly used functions across other files.
"""

import numpy as np
import os
from colorama import init, Fore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_imbalance(y, data_format):
    """
    Calculates the class distribution of the dataset and optionally prints it.

    Args:
        y (numpy.ndarray): Labels array where 0 indicates 'No crater pixel' and 1 indicates 'Crater pixel'.
        do_print (bool, optional): If True, prints the class imbalance. If False, returns the absolute difference between the two classes.

    Returns:
        int or None: The absolute difference between the number of 'Crater pixel' and 'No crater pixel' samples if `do_print` is False. Otherwise, None.
    """
    if data_format == "a":
        num_crater_pixels = sum(y)
        num_no_crater_pixels = len(y) - num_crater_pixels
        total = num_no_crater_pixels + num_crater_pixels

        print("\tCheck imbalance (format a):")
        print(f"\tClass 0 (No crater pixel): {num_no_crater_pixels} ({num_no_crater_pixels / total * 100:.2f} %)")
        print(f"\tClass 1 (Crater pixel): {num_crater_pixels} ({num_crater_pixels / total * 100:.2f} %)")     
    
    elif data_format == "b":   
        num_crater_pixels = len([1 for pixel in y.flatten() if pixel == 1])
        num_no_crater_pixels = len([0 for pixel in y.flatten() if pixel == 0])
        total = num_no_crater_pixels + num_crater_pixels

        print("\tCheck imbalance (format b):")
        print(f"\tClass 0 (No crater pixel): {num_no_crater_pixels} ({num_no_crater_pixels / total * 100:.2f} %)")
        print(f"\tClass 1 (Crater pixel): {num_crater_pixels} ({num_crater_pixels / total * 100:.2f} %)")  
    else:
        raise ValueError("Provide a valid format of y_data: either data_format='a' or data_format='b'.")


def load_all_data():
    """
    Loads the training and test datasets from `.npy` files.

    Args:
        None
    
    Returns:
        tuple: A tuple containing:
    """
    # Load training data
    X_train2_a = np.load(file="/home/morais/AAut_project/data/Xtrain2_a.npy")
    X_train2_b = np.load(file="/home/morais/AAut_project/data/Xtrain2_b.npy")
    
    # Load training labels
    Y_train2_a = np.load(file="/home/morais/AAut_project/data/Ytrain2_a.npy")
    Y_train2_b = np.load(file="/home/morais/AAut_project/data/Ytrain2_b.npy")
    
    # Load test data
    X_test2_a = np.load(file="/home/morais/AAut_project/data/Xtest2_a.npy")
    X_test2_b = np.load(file="/home/morais/AAut_project/data/Xtest2_b.npy")
    
    return X_train2_a, X_train2_b, Y_train2_a, Y_train2_b, X_test2_a, X_test2_b


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