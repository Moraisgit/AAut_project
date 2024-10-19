"""
This module trains and evaluates a k-NN classifier, selecting the best k based on F1 score.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from colorama import Fore


def knn_model(X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a k-NN classifier using different values of k, selecting the 
    best k based on the F1 score evaluated on the validation set.

    Args:
        X_train (numpy.ndarray): Training feature array of shape (n_train_samples, n_features).
        y_train (numpy.ndarray): Training labels array of shape (n_train_samples,).
        X_val (numpy.ndarray): Validation feature array of shape (n_val_samples, n_features).
        y_val (numpy.ndarray): Validation labels array of shape (n_val_samples,).

    Returns:
        tuple: A tuple containing the best value of k and the corresponding F1 score.
    """
    best_k = None
    best_f1_score = 0

    # Iterate over k values from 1 to 100 to find the best k based on F1 score
    for k in range(1, 101):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X=X_train, y=y_train)  # Train k-NN on the training set
        
        # Predict on validation set
        y_val_pred = model.predict(X=X_val)
        current_f1_score = f1_score(y_true=y_val, y_pred=y_val_pred, average='macro')

        # Print the current k and F1 score
        print(y_val, y_val_pred)
        print(f"k = {k}, F1-Score = {current_f1_score}")
        
        # Track the best k based on the highest F1 score
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_k = k
    
    # Print the final best k and F1 score
    print("---------------------------")
    print("Using " + Fore.YELLOW + "K-Nearest Neighbours (k-NN)" + Fore.RESET + ":")
    print(f"\tBest k = {best_k}\n\tF1 score = {best_f1_score:.4f}")

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X=X_train, y=y_train)  # Train k-NN on the training set
    
    return model