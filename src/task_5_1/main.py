import utils
import oversample
from sklearn.model_selection import train_test_split
import KNN
import CNN
import SVM


def main():
    X_train, y_train, X_train_extra, X_test = utils.load_all_data()
    print(X_train.shape, y_train.shape)

    ##########################
    # Inspect original dataset
    ##########################
    len_imbalanced = len(X_train)
    # utils.plot_dataset(X=X_train, y=y_train, num_images=20)
    utils.get_imbalance(y=y_train, do_print = True)

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
    X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, smote=True)
    # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images=30)
    # print(X_oversampled.shape, y_oversampled.shape)
    # print(X_oversampled, y_oversampled)

    """
    Oversample using ImageDataGenerator
    """
    # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, img_data_gen=True)
    # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
    # print(X_oversampled.shape, y_oversampled.shape)

    # """
    # Oversample using Manual Horizontal/Vertical Flipping - NOT USED
    # """
    # # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, manual_flips=True)
    # # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images=30)
    # # print(X_oversampled.shape, y_oversampled.shape)

    # """
    # Oversample using RandomOverSampler - NOT USED
    # """
    # # X_oversampled, y_oversampled = oversample.oversample_dataset(X=X_train, y=y_train, rand_over_samp=True)
    # # oversample.plot_oversample_images(X=X_oversampled, y=y_oversampled, len_prior_oversample=len_imbalanced, num_images= 30)
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
    # oversample.get_imbalance(y=y_train_split, do_print=True)
    # oversample.get_imbalance(y=y_val_split, do_print=True)

    ###################
    # Classifier models
    ###################
    # """
    # K-Nearest Neighbours (K-NN) - NOT USED
    # """
    # # KNN.knn_model(X_train=X_train_split, y_train=y_train_split, X_val=X_val_split, y_val=y_val_split)

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

    # # Use the previously tuned CNN model
    # cnn = CNN.build_cnn()
    # history = CNN.cnn_model(
    #     X_train=X_train_split, 
    #     y_train=y_train_split, 
    #     X_val=X_val_split,
    #     y_val=y_val_split,
    #     model=cnn
    # )
    # CNN.plot_cnn_history(history=history)

    """
    Support Vector Machine (SVM)
    """
    SVM.do_svm_grid_search(X_train=X_train_split, y_train=y_train_split, X_val=X_val_split, y_val=y_val_split)


if __name__ == "__main__":
    main()