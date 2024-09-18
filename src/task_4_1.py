import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

def generate_train_validation_data(X_train, y_train):
    X_train_split, X_validation_split, y_train_split, y_validation_split = train_test_split(
        X_train,  # Input features
        y_train,  # Target labels corresponding to X_train
        test_size=0.2,  # 20% of the data will go to validation set
        shuffle=True,   # Shuffle the data before splitting
        random_state=42  # Ensure reproducibility of the split (same result each time you run the code)
    )
    return X_train_split, X_validation_split, y_train_split, y_validation_split

def load_data(filename: str) -> np.ndarray:
    return np.load(file=filename)

def main():
    # Our output will be compared with the teachers output using SSE metric
    X_test = load_data("X_test.npy")   # Test data for the model: Do not use in training
    y_train = load_data("y_train.npy") # Expected output for the training data

    X_train = load_data("X_train.npy") # Training data for the model


    samples = np.arange(0, 200, 1)
    
    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(5, 2)

    # For Daily Averages of Air Temperature
    axis[0, 0].scatter(samples, X_train[:,0])
    axis[0, 0].set_title("x1")

    # For Water Temperature
    axis[0, 1].scatter(samples, X_train[:,1])
    axis[0, 1].set_title("x2")

    # For Wind Speed
    axis[1, 0].scatter(samples, X_train[:,2])
    axis[1, 0].set_title("x3")

    # For Wind Direction
    axis[1, 1].scatter(samples, X_train[:,3])
    axis[1, 1].set_title("x4")
    
    # For Illumination
    axis[2, 0].scatter(samples, X_train[:,4])
    axis[2, 0].set_title("x5")

    # For Daily Averages of Air Temperature
    axis[2, 1].scatter(samples, y_train[:,0])
    axis[2, 1].set_title("y1")

    # For Water Temperature
    axis[3, 0].scatter(samples, y_train[:,1])
    axis[3, 0].set_title("y2")

    # For Wind Speed
    axis[3, 1].scatter(samples, y_train[:,2])
    axis[3, 1].set_title("y3")

    # For Wind Direction
    axis[4, 0].scatter(samples, y_train[:,3])
    axis[4, 0].set_title("y4")
    
    # For Illumination
    axis[4, 1].scatter(samples, y_train[:,4])
    axis[4, 1].set_title("y5")


    plt.show()


if __name__ == "__main__":
    main()
