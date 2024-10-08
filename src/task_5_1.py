from utils import get_absolute_path, load_data, save_npy_to_output, get_plot_save_path
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


# Data Loader Function
def load_all_data():
    X_train = load_data(filename=get_absolute_path("X_train.npy"))
    X_test = load_data(filename=get_absolute_path("X_test.npy"))
    X_test1 = load_data(filename=get_absolute_path("X_test1.npy"))
    Xtrain1_extra = load_data(filename=get_absolute_path("Xtrain1_extra.npy"))
    y_train = load_data(filename=get_absolute_path("y_train.npy"))
    Ytrain1 = load_data(filename=get_absolute_path("Ytrain1.npy"))
    return X_train, X_test, X_test1, Xtrain1_extra, y_train, Ytrain1


# Oversampling Function
def oversample_data(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled


# Train-Test Split Function
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Main function
def main():
    # Load all data
    X_train, X_test, X_test1, Xtrain1_extra, y_train, Ytrain1 = load_all_data()

    # Perform oversampling to handle class imbalance
    X_resampled, y_resampled = oversample_data(X_train, y_train)

    # Split data into training and validation sets
    X_train_resampled, X_val_resampled, y_train_resampled, y_val_resampled = (
        train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    )


if __name__ == "__main__":
    main()
