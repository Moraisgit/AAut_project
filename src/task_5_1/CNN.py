from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
from keras.callbacks import EarlyStopping
import keras_tuner
import numpy as np
import matplotlib.pyplot as plt


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