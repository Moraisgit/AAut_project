"""
Main file of the program. This is the file to run by the Professor.
"""

import common
import oversample
from colorama import Fore


def main():
    ###################################
    # Load and inspect original dataset
    ###################################
    X_train_a, X_train_b, y_train_a, y_train_b, X_test_a, X_test_b = common.load_all_data()
    common.get_imbalance(y=y_train_a, data_format="a")
    common.get_imbalance(y=y_train_b, data_format="b")

    ####################
    # Normalize datasets
    ####################
    X_train_a = (X_train_a).astype('float32')/255.0
    X_train_b = (X_train_b).astype('float32')/255.0
    X_test_a = (X_test_a).astype('float32')/255.0
    X_test_b = (X_test_b).astype('float32')/255.0

    ########################
    # Take care of imbalance
    ########################
    """
    Oversample using SMOTE
    """
    X_oversampled_a, y_oversampled_a = oversample.oversample_dataset(X=X_train_a, y=y_train_a, data_format="a", smote=True)
    X_oversampled_b, y_oversampled_b = oversample.oversample_dataset(X=X_train_b, y=y_train_b, data_format="b", smote=True)
    # print(X_oversampled.shape, y_oversampled.shape)
    # print(X_oversampled, y_oversampled)

    """
    Oversample using ImageDataGenerator
    """

    #####################################################
    # Split Training data into train, validation and test
    #####################################################

    ###################
    # Classifier models
    ###################
    """
    K-Nearest Neighbours (K-NN)
    """


    ##################
    # Save predictions
    ##################


if __name__ == "__main__":
    main()