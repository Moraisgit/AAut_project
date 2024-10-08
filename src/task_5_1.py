from utils import get_absolute_path, load_data, save_npy_to_output, get_plot_save_path
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# Data Loader Function
def load_all_data():
    X_train = load_data(filename=get_absolute_path("Xtrain1.npy"))
    Y_train = load_data(filename=get_absolute_path("Ytrain1.npy"))
    X_train_extra = load_data(filename=get_absolute_path("Xtrain1_extra.npy"))
    X_test = load_data(filename=get_absolute_path("Xtest1.npy"))
    return X_train, Y_train, X_train_extra, X_test


# Image Preprocessing for CNN (reshaping and normalizing)
def preprocess_data_for_cnn(X):
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    # Reshape to (samples, 48, 48, 1) for grayscale images
    return X.reshape(-1, 48, 48, 1)


# Oversampling Function
def oversample_data(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(
        X.reshape(X.shape[0], -1), y
    )  # Flatten for resampling
    return (
        X_resampled.reshape(-1, 48, 48, 1),
        y_resampled,
    )  # Reshape back after resampling


# CNN Model Definition
def build_cnn_model(input_shape):
    model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(1, activation="sigmoid"),  # Binary classification output
        ]
    )

    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Main function
def main():
    # Load all data
    X_train, Y_train, X_train_extra, X_test = load_all_data()

    # Preprocess the data for the CNN
    X_train_preprocessed = preprocess_data_for_cnn(X_train)

    # Perform oversampling to handle class imbalance
    X_resampled, y_resampled = oversample_data(X_train_preprocessed, Y_train)

    # Split data into training and validation sets
    X_train_resampled, X_val_resampled, y_train_resampled, y_val_resampled = (
        train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    )

    # Build the CNN model
    model = build_cnn_model(input_shape=(48, 48, 1))

    # Train the model
    model.fit(
        X_train_resampled,
        y_train_resampled,
        epochs=10,
        batch_size=32,
        validation_data=(X_val_resampled, y_val_resampled),
    )

    # Predict on the validation set and calculate F1 score
    y_val_pred = (model.predict(X_val_resampled) > 0.5).astype("int32")
    f1 = f1_score(y_val_resampled, y_val_pred)
    print(f"F1 Score on validation set: {f1:.4f}")


if __name__ == "__main__":
    main()
