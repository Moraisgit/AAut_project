"""
Main file of the program. This is the file to run by the Professor.
"""

import common
import oversample
import KNN
import CNN
import SVM
from colorama import Fore


def main():

    ###################################
    # Load and inspect original dataset
    ###################################
    X_train, y_train, X_train_extra, X_test = common.load_all_data()

    len_imbalanced = len(X_train)
    # common.plot_dataset(X=X_train, y=y_train, num_images=20)
    print(Fore.YELLOW + "Original Dataset" + Fore.RESET + ":")
    common.get_imbalance(y=y_train, do_print = True)

    ####################
    # Normalize datasets
    ####################
    X_train = (X_train).astype('float32')/255.0
    X_test = (X_test).astype('float32')/255.0
    X_train_extra = (X_train_extra).astype('float32')/255.0

    ########################
    # Take care of imbalance
    ########################
    """
    Oversample using SMOTE
    """
    # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, smote=True)
    # # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images=30)
    # # print(X_oversampled.shape, y_oversampled.shape)
    # # print(X_oversampled, y_oversampled)

    """
    Oversample using ImageDataGenerator
    """
    # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, img_data_gen=True)
    # # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
    # # print(X_oversampled.shape, y_oversampled.shape)

    """
    Oversample using RandomOverSampler
    """
    X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, rand_over_samp=True)
    # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
    # print(X_oversampled.shape, y_oversampled.shape)

    # """
    # Oversample using Manual Horizontal/Vertical Flipping - NOT USED
    # """
    # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, manual_flips=True)
    # # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images=30)
    # # print(X_oversampled.shape, y_oversampled.shape)

    #####################################################
    # Split Training data into train, validation and test
    #####################################################
    X_train_split, y_train_split, X_val_split, y_val_split, X_test_split, y_test_split = common.split_data(
        X=X_oversampled, 
        y=y_oversampled
    )
    print("After " + Fore.YELLOW + "Training split" + Fore.RESET + ":")
    oversample.get_imbalance(y=y_train_split, do_print=True)
    # oversample.get_imbalance(y=y_val_split, do_print=True)

    ###################
    # Classifier models
    ###################
    # """
    # K-Nearest Neighbours (K-NN) - NOT USED
    # """
    # # knn_model = KNN.knn_model(X_train=X_train_split, y_train=y_train_split, X_val=X_test_split, y_val=y_test_split)
    # # y_pred = knn_model.predict(X=X_test)

    """
    Convolutional Neural Network (CNN)
    """
    # # Tune the CNN
    # CNN.cnn_tuner(
    #     X_train=X_train_split, 
    #     y_train=y_train_split, 
    #     X_val=X_val_split, 
    #     y_val=y_val_split
    # )

    # Use the previously tuned CNN model
    cnn = CNN.build_cnn()
    cnn_model, history = CNN.cnn_model(
        X_train=X_train_split, 
        y_train=y_train_split, 
        X_val=X_val_split,
        y_val=y_val_split,
        X_test=X_test_split,
        y_test = y_test_split,
        model=cnn
    )

    CNN.plot_cnn_history(history=history)

    # Predict on and add the extra dataset
    X_train_split_extra, y_train_split_extra = CNN.cnn_predict_extra(
        model=cnn_model, 
        X_train=X_train_split, 
        y_train=y_train_split, 
        X_train_extra=X_train_extra
    )

    # Train the model with the added dataset
    cnn = CNN.build_cnn()
    cnn_model, history = CNN.cnn_model(
        X_train=X_train_split_extra, 
        y_train=y_train_split_extra, 
        X_val=X_val_split,
        y_val=y_val_split,
        X_test=X_test_split,
        y_test = y_test_split,
        model=cnn
    )

    CNN.plot_cnn_history(history=history)

    # Make the final test predictions
    y_pred = CNN.cnn_predict(model=cnn_model, X=X_test.reshape(X_test.shape[0], 48, 48, 1))

    """
    Support Vector Machine (SVM)
    """
    # # # Tune the SVM
    # # results = SVM.do_svm_grid_search(X_train=X_train_split, y_train=y_train_split, X_val=X_test_split, y_val=y_test_split)
    # # SVM.plot_svm_history(grid_search_results=results)

    # # Use the previously tuned SVM model
    # svm_model = SVM.svm_model(X_train=X_train_split, y_train=y_train_split, X_val=X_test_split, y_val=y_test_split)

    # # Predict on and add the extra dataset
    # X_train_split_extra, y_train_split_extra = SVM.svm_predict_extra(
    #     model=svm_model, 
    #     X_train=X_train_split, 
    #     y_train=y_train_split, 
    #     X_train_extra=X_train_extra
    # )

    # # Train the model with the added dataset
    # svm_model = SVM.svm_model(X_train=X_train_split_extra, y_train=y_train_split_extra, X_val=X_test_split, y_val=y_test_split)

    # # Make the final test predictions
    # y_pred = svm_model.predict(X=X_test)

    ##################
    # Save predictions
    ##################
    common.save_npy_to_output(file_name="y_pred", data=y_pred)


if __name__ == "__main__":
    main()