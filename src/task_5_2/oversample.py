"""
This module provides functions for oversampling datasets using SMOTE and ImageDataGenerator. 
It includes functions to apply each technique and plot synthetic images.
"""

import common
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from random import shuffle
from colorama import Fore


def oversample_dataset(X, y, data_format, smote=False, img_data_gen=False):
    """
    Oversamples the dataset using one of the following techniques: SMOTE, ImageDataGenerator, 
    manual horizontal/vertical flips, or random oversampling.

    Args:
        X (numpy.ndarray): Input feature array of shape (n_samples, 2304).
        y (numpy.ndarray): Labels array corresponding to the input features.
        smote (bool, optional): Whether to apply SMOTE for oversampling. Default is False.
        img_data_gen (bool, optional): Whether to apply image data augmentation. Default is False.

    Returns:
        tuple: A tuple containing the oversampled feature array and labels array.
    
    Raises:
        ValueError: If none of the oversampling methods are selected.
    """
    if smote and (data_format != None):
        # Apply SMOTE to balance the dataset
        return use_smote(X=X, y=y, data_format=data_format)

    # elif img_data_gen and format:
    #     # Apply ImageDataGenerator for augmentation
    #     return use_img_data_gen(X=X, y=y)
    
    else:
        # Raise an error if none of the methods are selected
        raise ValueError("At least one method ('smote' or 'img_data_gen') must be True to perform oversampling.")


def use_smote(X, y, data_format):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset by generating 
    synthetic samples for the minority class.

    Args:
    X (ndarray): Input feature matrix (n_samples, n_features).
    y (ndarray): Target labels (n_samples,), with 0 indicating no crater and 1 indicating a crater.

    Returns:
    tuple:
        X_oversampled (ndarray): The oversampled feature matrix including synthetic samples.
        y_oversampled (ndarray): The oversampled target labels including synthetic labels.
    """
    # Print class imbalance before applying SMOTE
    print("Before " + Fore.YELLOW + "SMOTE" + Fore.RESET + ":")
    common.get_imbalance(y=y, data_format=data_format)

    # Apply SMOTE to generate synthetic samples for the minority class
    smote = SMOTE(k_neighbors=3, sampling_strategy='minority', random_state=42)
    X_oversampled, y_oversampled = smote.fit_resample(X=X, y=y)

    # Print class imbalance after applying SMOTE
    print("After " + Fore.YELLOW + "SMOTE" + Fore.RESET + ":")
    common.get_imbalance(y=y_oversampled, data_format=data_format)

    return X_oversampled, y_oversampled