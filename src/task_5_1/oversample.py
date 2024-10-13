"""
This module provides functions for oversampling datasets using SMOTE, ImageDataGenerator, manual flips, 
and random oversampling techniques. It includes functions to apply each technique and plot synthetic images.
"""

from utils import get_imbalance, plot_dataset
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from random import shuffle
from imblearn.over_sampling import RandomOverSampler


def plot_oversample_images(X, y, len_prior_oversample, num_images=16):
    """
    Plots a grid of synthetic oversampled images along with their labels.

    Args:
        X (numpy.ndarray): Input feature array of shape (n_samples, 2304) where each entry is an image.
        y (numpy.ndarray): Labels array corresponding to the input features.
        len_prior_oversample (int): The number of original samples before oversampling.
        num_images (int, optional): The number of images to display. Default is 16.

    Returns:
        None: This function displays the plot and does not return any value.
    """
    # Get only synthetic images and their corresponding labels
    X_oversample = X[len_prior_oversample:]  
    y_oversample = y[len_prior_oversample:]  
    
    # Plot the synthetic images using the previously defined plot_dataset function
    plot_dataset(X=X_oversample, y=y_oversample, num_images=num_images)


def oversample_dataset(X, y, smote=False, img_data_gen=False, manual_flips=False, rand_over_samp=True):
    """
    Oversamples the dataset using one of the following techniques: SMOTE, ImageDataGenerator, 
    manual horizontal/vertical flips, or random oversampling.

    Args:
        X (numpy.ndarray): Input feature array of shape (n_samples, 2304).
        y (numpy.ndarray): Labels array corresponding to the input features.
        smote (bool, optional): Whether to apply SMOTE for oversampling. Default is False.
        img_data_gen (bool, optional): Whether to apply image data augmentation. Default is False.
        manual_flips (bool, optional): Whether to apply manual image flipping augmentation. Default is False.
        rand_over_samp (bool, optional): Whether to apply random oversampling. Default is True.

    Returns:
        tuple: A tuple containing the oversampled feature array and labels array.
    
    Raises:
        ValueError: If none of the oversampling methods are selected.
    """
    if smote:
        # Apply SMOTE to balance the dataset
        return use_smote(X=X, y=y)

    elif img_data_gen:
        # Apply ImageDataGenerator for augmentation
        return use_img_data_gen(X=X, y=y)
    
    # elif manual_flips:
    #     # Apply manual horizontal and vertical flips for augmentation
    #     return use_manual_flips(X=X, y=y)

    # elif rand_over_samp:
    #     # Apply random oversampling
    #     return use_rand_over_samp(X=X, y=y)
    
    else:
        # Raise an error if none of the methods are selected
        raise ValueError("At least one method ('smote', 'img_data_gen', 'manual_flips', or 'rand_over_samp') must be True to perform oversampling.")


def use_smote(X, y):
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
    get_imbalance(y=y, do_print=True)

    # Apply SMOTE to generate synthetic samples for the minority class
    smote = SMOTE(k_neighbors=3, sampling_strategy='minority', random_state=42)
    X_oversampled, y_oversampled = smote.fit_resample(X=X, y=y)

    # Print class imbalance after applying SMOTE
    get_imbalance(y=y_oversampled, do_print=True)

    return X_oversampled, y_oversampled


def use_img_data_gen(X, y):
    """
    Augment the dataset using ImageDataGenerator to balance the class distribution by generating 
    synthetic no-crater images with random transformations (rotation, brightness adjustment, flips).

    Args:
    X (ndarray): Input feature matrix with shape (n_samples, 2304), representing 48x48 pixel images.
    y (ndarray): Target labels (n_samples,), with 0 indicating no crater and 1 indicating a crater.

    Returns:
    tuple: 
        X_oversampled (ndarray): The oversampled feature matrix with original and augmented no-crater images.
        y_oversampled (ndarray): The oversampled target labels with added no-crater samples.
    """
    # Rescale pixel values for ImageDataGenerator (back to the 0-255 range)
    X = X * 255  

    # Print the current class imbalance before augmentation
    get_imbalance(y=y, do_print=True)

    # Get the number of samples needed to balance the dataset
    oversample_size = get_imbalance(y=y, do_print=False)

    # Select indices of the minority class (No crater)
    minority_class_indices = np.where(y == 0)[0]  
    selected_indices = np.random.choice(minority_class_indices, size=oversample_size, replace=False)
    X_minority_selected = X[selected_indices]

    # Configure the ImageDataGenerator for augmentations (rotation, brightness, and flips)
    datagen = ImageDataGenerator(
        rotation_range=90,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
    )

    # Augment the selected minority class images
    X_augmented = np.empty_like(X_minority_selected)
    for i, img in enumerate(X_minority_selected):
        img = img.reshape((1, 48, 48, 1))  # Reshape for ImageDataGenerator to accept (batch, height, width, channels)
        augmented_img = next(datagen.flow(img, batch_size=1))[0]  # Generate augmented image from datagen
        X_augmented[i] = augmented_img.reshape(-1)  # Flatten augmented image back to original shape

    # Concatenate original dataset and augmented images
    X_oversampled = np.vstack((X, X_augmented))

    # Create corresponding labels (0s for no-crater) for the augmented images
    Y_augmented = np.zeros(oversample_size, dtype=int)

    # Concatenate original labels with augmented labels
    y_oversampled = np.hstack((y, Y_augmented))

    # Print the class imbalance after augmentation
    get_imbalance(y=y_oversampled, do_print=True)

    return X_oversampled, y_oversampled


# def use_manual_flips(X, y):
    """
    Manually augment the dataset by applying horizontal and vertical flips to no-crater images 
    to balance the class distribution.

    Args:
    X (ndarray): Input feature matrix with shape (n_samples, 2304), representing 48x48 pixel images.
    y (ndarray): Target labels (n_samples,), with 0 indicating no crater and 1 indicating a crater.

    Returns:
    tuple: 
        X_oversampled (ndarray): The oversampled feature matrix with augmented no-crater images.
        y_oversampled (ndarray): The oversampled target labels with added no-crater samples.
    """
    # Reshape X to its original image size (48x48) for manipulation
    X = X.reshape(len(X), 48, 48)

    # Print class imbalance
    get_imbalance(y=y, do_print=True)
    # Calculate how many extra craters we need to balance the dataset
    oversample_size = get_imbalance(y=y, do_print=False)

    # Select no-crater images
    no_craters = []
    for i in range(len(X)):
        if y[i] == 0:
            no_craters.append(X[i])

    # Shuffle and select only as many as needed for balancing
    shuffle(no_craters)
    no_craters = no_craters[:oversample_size]  # Select exactly n_extra_craters

    # Apply horizontal and vertical flips to the selected images
    augmented_no_craters = []
    for img in no_craters:
        flipped_img = np.fliplr(np.flipud(img))  # Apply both horizontal and vertical flips
        augmented_no_craters.append(flipped_img)

    # # Hardcoded indexes for plotting
    # hardcoded_indexes = [0, 1, 2]  # Modify if needed

    # for idx in hardcoded_indexes:
    #     if idx < len(no_craters):
    #         original_img = no_craters[idx]
    #         flipped_img = augmented_no_craters[idx]

    #         plt.figure()

    #         # Plot original image
    #         plt.subplot(1, 2, 1)
    #         plt.title(f"Original (Index {idx})")
    #         plt.imshow(original_img, cmap='gray')

    #         # Plot flipped image
    #         plt.subplot(1, 2, 2)
    #         plt.title(f"Flipped (Index {idx})")
    #         plt.imshow(flipped_img, cmap='gray')

    #         plt.show()

    # Flatten the augmented images back to 2304 to match the original data structure
    augmented_no_craters = [img.flatten() for img in augmented_no_craters]

    # Concatenate augmented data with the original data
    X_oversampled = np.vstack((X.reshape(len(X), -1), augmented_no_craters))
    y_oversampled = np.hstack((y, np.zeros(len(augmented_no_craters))))

    # Print class imbalance
    get_imbalance(y=y_oversampled, do_print=True)

    return X_oversampled, y_oversampled


# def use_rand_over_samp(X, y):
    """
    Randomly oversamples the minority class in `y` to balance the dataset by reshaping and resampling `X`.

    Args:
    X (ndarray): Input feature matrix, reshaped if needed (n_samples, height, width) or (n_samples, n_features).
    y (ndarray): Target labels (n_samples,).

    Returns:
    tuple: 
        X_oversampled (ndarray): The oversampled feature matrix.
        y_oversampled (ndarray): The oversampled target labels.
    """
    ros = RandomOverSampler(random_state=42)
    X_oversampled, y_oversampled = ros.fit_resample(X.reshape(X.shape[0], -1), y)
    
    return X_oversampled, y_oversampled