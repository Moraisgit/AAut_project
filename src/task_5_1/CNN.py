"""
This module implements Concurrent Neural Networks related functions.
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras_tuner
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from colorama import Fore
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import common


def build_cnn_tuner(hp):
    INPUT_SHAPE = (48, 48, 1)

    model = Sequential([
        Conv2D(filters=hp.Int('conv1_filters', min_value=32, max_value=64, step=32), kernel_size=(3, 3), input_shape=INPUT_SHAPE),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=3, strides=3),

        Conv2D(filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32), kernel_size=(3, 3)),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=3, strides=3),

        Conv2D(filters=hp.Int('conv3_filters', min_value=32, max_value=256, step=32), kernel_size=2),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),

        Flatten(),

        Dense(units=hp.Int('dense_units', min_value=128, max_value=1024, step=256)),
        Activation(activation='relu'),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)),

        Dense(units=hp.Int('dense_units', min_value=128, max_value=256, step=128)),
        Activation(activation='relu'),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)),

        Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32)),
        Activation(activation='relu'),
        Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.1)),

        Dense(units=1),
        Activation(activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', values=[0.00001, 0.0001, 0.001, 0.01])),
        loss="binary_crossentropy",
        metrics=['accuracy', f1_score_m, recall_m, precision_m]
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
        objective=keras_tuner.Objective(name="val_f1_score_m", direction="max"),
        max_retries_per_trial=2,
        overwrite=True
    )
    tuner.search_space_summary()
    tuner.search(
        X_train, y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)


def build_cnn():
    INPUT_SHAPE = (48, 48, 1)

    # Initialize the CNN model
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), input_shape=INPUT_SHAPE),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=3, strides=3),

        Conv2D(filters=32, kernel_size=(3, 3)),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=3, strides=3),

        Conv2D(filters=32, kernel_size=2),
        Activation(activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),

        Flatten(),

        Dense(units=1024),
        Activation(activation='relu'),
        Dropout(rate=0.5),

        Dense(units=256),
        Activation(activation='relu'),
        Dropout(rate=0.5),

        Dense(units=56),
        Activation(activation='relu'),
        Dropout(rate=0.5),

        Dense(units=1),
        Activation(activation='sigmoid')
    ])

    model.summary()

    return model


def cnn_model(X_train, y_train, X_val, y_val, X_test, y_test, model):
    # Reshape the input data to add the channel dimension
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)  # Shape: (num_samples, 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

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
        metrics=['accuracy', f1_score_m, recall_m, precision_m]
    )

    # Fit the model
    history = model.fit(
        x=X_train, 
        y=y_train, 
        verbose=1, 
        epochs=EPOCHS, 
        validation_data=(X_val, y_val),
        shuffle = False,
        callbacks=callbacks
    )

    y_pred = cnn_predict(model=model, X=X_test)

    # Calculate the F1 score on the validation set
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

    # Output the best parameters and the best F1 score
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Concurrent Neural Network (CNN)" + Fore.RESET + ":")
    print(f"\tF1 Score on Validation Set: {f1}")

    return model, history


def cnn_predict(model, X):
    y_pred_prob = model.predict(X)

    y_pred = ((y_pred_prob > 0.5).astype(int)).flatten().tolist()  # Convert probabilities to 0 or 1 labels

    return y_pred


def cnn_predict_extra(model, X_train, y_train, X_train_extra):
    """
    Predicts on the extra data, filters samples with probabilities above the threshold,
    and appends the best (most confident) samples to the training set, ensuring class balance.

    Args:
        model (CNN): The trained CMM model.
        X_train_extra (numpy.ndarray): The extra input data.
        y_train (numpy.ndarray): The original training labels.
        threshold (float): The probability threshold to consider for filtering.

    Returns:
        X_train (numpy.ndarray): The updated training input features (shuffled).
        y_train (numpy.ndarray): The updated training labels (shuffled).
    """
    THRESHOLD = 0.9

    # Get the predicted probabilities
    y_proba = model.predict(X_train_extra.reshape(X_train_extra.shape[0], 48, 48, 1))
    print(len(X_train_extra), len(y_proba))
    
    # Convert the probabilities to a flat array
    y_proba_flat = y_proba.flatten()

    # Select samples where the probability is higher than the threshold
    selected_class_0_idx = np.where(y_proba_flat <= (1 - THRESHOLD))
    selected_class_1_idx = np.where(y_proba_flat >= THRESHOLD)

    # Corresponding samples
    selected_class_0 = X_train_extra[selected_class_0_idx]
    selected_class_1 = X_train_extra[selected_class_1_idx]

    # Get the predicted probabilities for the selected samples of each class
    prob_class_0 = 1 - y_proba_flat[selected_class_0_idx]
    prob_class_1 = y_proba_flat[selected_class_1_idx]

    # Balance the samples: take the minimum count between class 0 and class 1
    min_count = min(len(selected_class_0), len(selected_class_1))

    # Sort the samples by the highest probability and select the top samples
    sorted_class_0_idx = np.argsort(prob_class_0)[::-1]
    sorted_class_1_idx = np.argsort(prob_class_1)[::-1]

    # Get the top `min_count` samples
    best_class_0 = selected_class_0[sorted_class_0_idx[:min_count]]
    best_class_1 = selected_class_1[sorted_class_1_idx[:min_count]]

    # Corresponding labels
    best_labels_0 = np.zeros(len(best_class_0))
    best_labels_1 = np.ones(len(best_class_1))

    # Combine the selected samples and labels
    X_combined = np.vstack([best_class_0, best_class_1])
    y_combined = np.concatenate([best_labels_0, best_labels_1])

    # Append to the original dataset
    X_train_updated = np.vstack([X_train, X_combined])
    y_train_updated = np.concatenate([y_train, y_combined])

    # Shuffle the data using sklearn's shuffle
    X_train_shuffled, y_train_shuffled = shuffle(X_train_updated, y_train_updated, random_state=42)

    print("After " + Fore.YELLOW + "Data Augmentation (extra dataset)" + Fore.RESET + ":")
    common.get_imbalance(y=y_train_shuffled)

    return X_train_shuffled, y_train_shuffled


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
    f1 = history.history.get('f1_score_m')
    val_f1 = history.history.get('val_f1_score_m')

    epochs = range(0, len(acc))

    # Plot 1: Training and validation accuracy
    plt.figure()
    plt.plot(epochs, acc, label='Training')
    plt.plot(epochs, val_acc, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 2: Training and validation loss
    plt.figure()
    plt.plot(epochs, loss, label='Training')
    plt.plot(epochs, val_loss, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 3: Training and validation F1 Score
    plt.figure()
    plt.plot(epochs, f1, label='Training')
    plt.plot(epochs, val_f1, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


def f1_score_m(y_true, y_pred):
    """
    Calculates the F1 score
    Code adapted from: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    """
    # # of True positives 
    # (both predicted and true value is 1 so y_pred*y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # # of True positives + # of False negatives = # of real positives
    # (value of y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    real_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # # of True positives + # of False positives = # of predicted positives
    # (value of y_pred is 1),
    # round and clip is for the values to stay binary (1 or 0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    # precision formula, k.epsilon is a small number to avoid dividing by 0
    precision = true_positives / (predicted_positives + K.epsilon())

    # recall formula, k.epsilon is a small number to avoid dividing by 0
    recall = true_positives / (real_positives + K.epsilon())

    # Calculate f1 score using the formula
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def recall_m(y_true, y_pred):
    # # of True positives 
    # (both predicted and true value is 1 so y_pred*y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # # of True positives + # of False negatives = # of real positives
    # (value of y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # recall formula, k.epsilon is a small number to avoid dividing by 0
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    # # of True positives 
    # (both predicted and true value is 1 so y_pred*y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # # of True positives + # of False positives = # of real positives
    # (value of y_true is 1),
    # round and clip is for the values to stay binary (1 or 0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    # precision formula, k.epsilon is a small number to avoid dividing by 0
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision