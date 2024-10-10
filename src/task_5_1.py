from utils import get_absolute_path, load_data, save_npy_to_output, get_plot_save_path
from imblearn.over_sampling import SMOTE
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from colorama import Fore
from random import shuffle
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
from keras.callbacks import EarlyStopping
import keras_tuner

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
    X_train = load_data(filename=get_absolute_path("Xtrain1.npy"))
    
    # Load training labels
    Y_train = load_data(filename=get_absolute_path("Ytrain1.npy"))
    
    # Load extra training data
    X_train_extra = load_data(filename=get_absolute_path("Xtrain1_extra.npy"))
    
    # Load test data
    X_test = load_data(filename=get_absolute_path("Xtest1.npy"))
    
    return X_train, Y_train, X_train_extra, X_test


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
        print("---------------------------")
        print("Check imbalance:")
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
    Oversamples the dataset using either SMOTE, ImageDataGenerator or Manual Flips.

    Args:
        X (numpy.ndarray): Input feature array of shape (n_samples, 2304).
        y (numpy.ndarray): Labels array corresponding to the input features.
        smote (bool, optional): Whether to apply SMOTE for oversampling. Default is False.
        img_data_gen (bool, optional): Whether to apply image data augmentation. Default is False.
        manual_flips (bool, optional): Whether to apply image horizontal/vertical flipping augmentation. Default is False.

    Returns:
        tuple: A tuple containing the oversampled feature array and labels array.
    
    Raises:
        ValueError: If both smote and img_data_gen are False.
    """
    if smote:
        # Print class imbalance
        get_imbalance(y=y, do_print=True)

        smote = SMOTE(k_neighbors=3, sampling_strategy='minority', random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X, y)

        get_imbalance(y=y_oversampled, do_print=True)  # Print class imbalance

        return X_oversampled, y_oversampled

    elif img_data_gen:
        # Rescale pixel values for ImageDataGenerator
        X = X * 255  

        # Print class imbalance
        get_imbalance(y=y, do_print=True)
        # Get the number of samples needed for oversampling
        oversample_size = get_imbalance(y=y, do_print=False)

        # Select indices of the minority class (No crater)
        minority_class_indices = np.where(y == 0)[0]  
        selected_indices = np.random.choice(minority_class_indices, size=oversample_size, replace=False)
        X_minority_selected = X[selected_indices]

        # Configure the ImageDataGenerator for augmentations
        datagen = ImageDataGenerator(
            rotation_range=90,
            brightness_range=[0.5, 1.5],
            horizontal_flip=True,
            vertical_flip=True,
        )

        # Augment the selected minority class images
        X_augmented = np.empty_like(X_minority_selected)
        for i, img in enumerate(X_minority_selected):
            img = img.reshape((1, 48, 48, 1))  # Reshape for ImageDataGenerator
            augmented_img = next(datagen.flow(img, batch_size=1))[0]  # Generate augmented image
            X_augmented[i] = augmented_img.reshape(-1)  # Flatten back to original shape

        # Concatenate original and augmented images
        X_oversampled = np.vstack((X, X_augmented))

        # Create labels for the augmented images
        Y_augmented = np.zeros(oversample_size, dtype=int)

        # Concatenate original labels and augmented labels
        y_oversampled = np.hstack((y, Y_augmented))

        # Print class imbalance
        get_imbalance(y=y_oversampled, do_print=True)

        return X_oversampled, y_oversampled
    
    elif manual_flips:
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

    elif rand_over_samp:
        ros = RandomOverSampler(random_state=42)
        X_oversampeld, y_oversampled = ros.fit_resample(X.reshape(X.shape[0], -1), y)  # Flatten for resampling

        return X_oversampeld, y_oversampled  # Reshape back after resampling
    
    else:
        # Raise an error if neither method is selected
        raise ValueError("Either 'smote' or 'img_data_gen' must be True to perform oversampling.")


def knn_model(X_train, y_train, X_val, y_val):
    best_k = None
    best_f1_score = 0

    # Iterate over each k to find the best one
    for k in range(1, 101):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)  # Train on training set
        
        # Evaluate on validation set
        y_val_pred = knn.predict(X_val)
        current_f1_score = f1_score(y_true=y_val, y_pred=y_val_pred)
        print(y_val, y_val_pred)

        print(current_f1_score)
        
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_k = k
    
    print("---------------------------")
    print("Using " + Fore.YELLOW + "K-Nearest Neighbours (k-NN)" + Fore.RESET + ":")
    print(f"\tBest k = {best_k}\n\tF1 score = {best_f1_score:.4f}")


def build_cnn_tuner(hp):
    INPUT_SHAPE = (48, 48, 1)

    model = Sequential([
        # Layer 1: Convolutional Layer (Conv2D)
        Conv2D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=64, step=16), 
            kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 2: Convolutional Layer (Conv2D)
        Conv2D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=128, step=32), 
            kernel_size=(3, 3), activation='relu'
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 3: Convolutional Layer (Conv2D)
        Conv2D(
            filters=hp.Int('conv3_filters', min_value=128, max_value=256, step=64), 
            kernel_size=(3, 3), activation='relu'
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)),

        Flatten(),

        Dense(
            units=hp.Int('dense_units', min_value=128, max_value=256, step=64), 
            activation='relu'
        ),
        Dropout(hp.Float('dense_dropout_rate', min_value=0.3, max_value=0.5, step=0.1)),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', values=np.logspace(start=-5, stop=-2, num=4).tolist())),
        loss="binary_crossentropy",
        metrics=['accuracy', F1Score(num_classes=2, average="micro")]
    )

    return model


def cnn_tuner(X_train, y_train, X_val, y_val):
    # Reshape the input data to add the channel dimension
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  # Shape: (num_samples, 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)        # Shape: (num_samples, 48, 48, 1)

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
    ]
    tuner = keras_tuner.Hyperband(
        build_cnn_tuner,
        objective=keras_tuner.Objective(name="val_f1_score", direction="max"),
        max_retries_per_trial=2,
        overwrite=True,
        directory='my_dir',
        project_name='cnn_tuning_hyperband'
    )
    tuner.search_space_summary()
    tuner.search(
        X_train, y_train,
        epochs=10,  # You can set this based on your training duration
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)


def build_cnn():
    INPUT_SHAPE = (48, 48, 1)

    # Initialize the CNN model
    model = Sequential([
        # Layer 1: Convolutional Layer (Conv2D)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),

        # Layer 2: Max Pooling Layer (MaxPooling2D)
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 3: Convolutional Layer (Conv2D)
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),

        # Layer 4: Max Pooling Layer (MaxPooling2D)
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 5: Convolutional Layer (Conv2D)
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),

        # Layer 6: Max Pooling Layer (MaxPooling2D)
        MaxPooling2D(pool_size=(2, 2)),

        # Dropout for regularization to prevent overfitting
        Dropout(0.25),

        # Flatten the output to a 1D vector for the fully connected layer
        Flatten(),

        # Fully connected layer
        Dense(256, activation='relu'),
        Dropout(0.5),  # Dropout for regularization

        # Output layer: Binary classification (0: No crater, 1: Crater)
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    return model


def cnn_model(X_train, y_train, X_val, y_val, model):
    # Reshape the input data to add the channel dimension
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  # Shape: (num_samples, 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)        # Shape: (num_samples, 48, 48, 1)

    EPOCHS = 100

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
    ]

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=['accuracy', F1Score(num_classes=1)]
    )

    # Fit the model
    history = model.fit(
        x=X_train, 
        y=y_train, 
        verbose = 1, 
        epochs = EPOCHS, 
        validation_data=(X_val, y_val),
        shuffle = False,
        # callbacks=callbacks
    )

    return history


def plot_cnn_history(history):
    """
    Plot the accuracy, loss, and F1 score over the number of epochs as separate figures.
    
    Parameters:
    history: History object returned by model.fit()
    """

    # Extract the metrics from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f1 = history.history.get('f1_score')
    val_f1 = history.history.get('val_f1_score')

    epochs = range(0, len(acc))

    # Plot 1: Training and validation accuracy
    plt.figure()
    plt.plot(epochs, acc, label='Training')
    plt.plot(epochs, val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    # Plot 2: Training and validation loss
    plt.figure()
    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    # Plot 3: Training and validation F1 Score
    plt.figure()
    plt.plot(epochs, f1, label='Training')
    plt.plot(epochs, val_f1, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()

    plt.show()


def main():
    X_train, y_train, X_train_extra, X_test = load_all_data()
    print(X_train.shape, y_train.shape)

    ##########################
    # Inspect original dataset
    ##########################
    len_imbalanced = len(X_train)
    # plot_dataset(X=X_train, y=y_train, num_images=20)
    # get_imbalance(y=y_train, do_print = True)

    ####################
    # Normalize datasets
    ####################
    X_train = (X_train).astype('float32')/255.0
    X_test = (X_test).astype('float32')/255.0

    ########################
    # Take care of imbalance
    ########################
    """
    Oversample using SMOTE
    """
    X_oversampled, y_oversampled = oversample_dataset(X=X_train, y=y_train, smote=True)
    # plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images=30)
    # print(X_oversampled.shape, y_oversampled.shape)
    # print(X_oversampled, y_oversampled)

    """
    Oversample using ImageDataGenerator
    """
    # X_oversampled, y_oversampled = oversample_dataset(X=X_train, y=y_train, img_data_gen=True)
    # # plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
    # # print(X_oversampled.shape, y_oversampled.shape)

    """
    Oversample using Manual Horizontal/Vertical Flipping
    """
    # X_oversampled, y_oversampled = oversample_dataset(X=X_train, y=y_train, manual_flips=True)
    # # print(X_oversampled.shape, y_oversampled.shape)
    # # print(X_oversampled, y_oversampled)

    """
    Oversample using RandomOverSampler
    """
    # X_oversampled, y_oversampled = oversample_dataset(X=X_train, y=y_train, rand_over_samp=True)
    # # plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
    # # print(X_oversampled.shape, y_oversampled.shape)


    #####################################################
    # Split Training data into train, validation and test
    #####################################################
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_oversampled, 
        y_oversampled, 
        test_size=0.2,  # 20% for validation and test
        shuffle=True,   # Shuffle the data
        random_state=42, # Ensure reproducibility
        stratify=y_oversampled # Stratify based on labels to ensure class balance
    )
    # get_imbalance(y=y_train_split, do_print=True)
    # get_imbalance(y=y_val_split, do_print=True)
    # print(X_train_split, y_train_split)

    ###################
    # Classifier models
    ###################
    """
    K-Nearest Neighbours (K-NN)
    """
    # knn_model(X_train=X_train_split, y_train=y_train_split, X_val=X_val_split, y_val=y_val_split)

    """
    Convolutional Neural Network (CNN)
    """
    # cnn_tuner(
    #     X_train=X_train_split, 
    #     y_train=y_train_split, 
    #     X_val=X_val_split, 
    #     y_val=y_val_split
    # )

    # cnn = build_cnn()
    # history = cnn_model(
    #     X_train=X_train_split, 
    #     y_train=y_train_split, 
    #     X_val=X_val_split,
    #     y_val=y_val_split,
    #     model=cnn
    # )
    # plot_cnn_history(history=history)

    """
    Support Vector Machine (SVM)
    """


if __name__ == "__main__":
    main()
