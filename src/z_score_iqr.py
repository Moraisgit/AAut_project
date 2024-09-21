def remove_outliers_iqr(X: np.ndarray, y: np.ndarray) -> np.ndarray:

    print("Initial shapes: " + str(X.shape) + str(y.shape))

    X = pd.DataFrame(X)
    y = pd.Series(y)

    # INitialize the list to store the indices of the outlierrs
    outliers_indices = []

    for column in X.columns:
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Detect outliers
        outliers = X[(X[column] < lower) | (X[column] > upper)].index
        outliers_indices.extend(outliers)

    # Remove duplicates in outliers indices
    outliers_indices = list(set(outliers_indices))

    # Drop the outlier rows from both X and y
    X_clean = X.drop(index=outliers_indices).reset_index(drop=True)
    y_clean = y.drop(index=outliers_indices).reset_index(drop=True)

    print("New shapes: " + str(X_clean.shape) + str(y_clean.shape) + "\n")

    return X_clean, y_clean


def remove_outliers_with_z_score(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Detect and remove outliers from the training data based on Z-scores.

    Parameters:
    X_train (np.ndarray): The input features of the training data.
    y_train (np.ndarray): The target labels of the training data.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Cleaned features and labels without outliers.
    """
    print("Initial shapes: " + str(X_train.shape) + str(y_train.shape))

    # Define the threshold for Z-score to identify outliers
    # Fine-tuned to lose the 25% of outliers - 200/4=50
    threshold = 1.87

    # Extract all columns from X_train
    cols = [X_train[:, i] for i in range(X_train.shape[1])]
    # Calculate the mean of each column
    means = [np.mean(col) for col in cols]
    # Calculate the standard deviation of each column
    stds = [np.std(col) for col in cols]
    # Calculate the Z-scores for each column
    z_scores = [(col - means[i]) / stds[i] for i, col in enumerate(cols)]

    # Identify rows with absolute Z-scores greater than the threshold
    rows_to_remove = set()  # Use a set to avoid duplicate indices
    for z_col in z_scores:
        for row_idx, z_score in enumerate(z_col):
            if np.abs(z_score) > threshold:
                rows_to_remove.add(row_idx)

    # Determine the indices of rows to keep (those not identified as outliers)
    rows_to_keep = list(set(range(X_train.shape[0])) - rows_to_remove)
    
    # Keep only the non-outlier rows in X_train
    X_clean = X_train[rows_to_keep, :]
    # Keep only the non-outlier rows in y_train
    y_clean = y_train[rows_to_keep]

    print("New shapes: " + str(X_clean.shape) + str(y_clean.shape) + "\n")
    
    return X_clean, y_clean
