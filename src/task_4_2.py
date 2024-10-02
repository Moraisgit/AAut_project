import numpy as np
from utils import get_absolute_path, load_data, save_npy_to_output
from sklearn.linear_model import Ridge
from itertools import product
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.metrics import r2_score


def regression_matrix(y, u, n, m, d):
    N = len(y)
    p = max(n, d + m)

    # Initialize X and Y
    X = []
    Y = []

    # Loop over the range where we can form full regressors
    for k in range(p, N):
        # Construct phi(k) as a combination of past y and u values

        # Collect the past n values of y: y(k-1), ..., y(k-n)
        phi_y = [y[k - i] for i in range(1, n + 1)]

        # Collect the past m+1 values of u: u(k-d), ..., u(k-d-m)
        phi_u = [u[k - d - i] for i in range(0, m + 1)]

        # Concatenate phi_y and phi_u to form the full regressor
        phi = np.concatenate([phi_y, phi_u])

        # Append the regressor to X
        X.append(phi)

        # Append the corresponding output y(k)
        Y.append(y[k])

    return np.array(X), np.array(Y)


def compute_SEE(y_real: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) between the actual and predicted values.

    Parameters:
    y_real (np.ndarray): The actual target values (observed).
    y_predicted (np.ndarray): The predicted target values from the model.

    Returns:
    float: The Sum of Squared Errors (SSE) for inliers.
    """

    # Compute SSE for the inliers
    return np.sum((y_real - y_predicted)**2)


def find_best_parameters(y_train, u_train):
    n_range = range(0, 10, 1)
    m_range = range(0, 10, 1)
    d_range = range(0, 10, 1)
    best_score = np.inf
    best_params = None
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

    for n, m, d, alpha in product(n_range, m_range, d_range, lambda_values):
        X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=m, d=d)

        if X_train_matrix.shape[0] < 10:
            continue

        X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
            X_train_matrix, 
            y_train_matrix, 
            test_size=0.2, 
            shuffle=False
        )

        ridge_reg = Ridge(
            alpha=alpha, 
            fit_intercept=True
        ).fit(X=X_train_matrix_split, y=y_train_matrix_split)

        y_pred = ridge_reg.predict(X=X_val_matrix_split)

        score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

        if score < best_score:
            best_score = score
            best_params = (n, m, d, alpha)

    return best_params


def recursive_predict(y_train, u_test, model, best_params):
    n, m, d, _ = best_params

    N = len(u_test)
    p = max(n, d + m)

    # Initialize the predicted output with zeros initially
    y_pred = np.zeros(N)

    # Set initial conditions using the last n values of y_train
    y_pred[:p] = y_train[-p:]

    # Predict y(k) iteratively for the test set
    for k in range(p, N):
        # Collect the past n values of y_pred
        phi_y = [y_pred[k - i] for i in range(1, n + 1)]

        # Collect the past m+1 values of u_test
        phi_u = [u_test[k - d - i] for i in range(0, m + 1)]

        # Concatenate phi_y and phi_u to form the full regressor
        phi = np.concatenate([phi_y, phi_u])

        # Predict y(k) using the trained model
        y_pred[k] = model.predict([phi])[0]

    # Return the full predicted output
    return y_pred


def chosen_model_with_best_parameters(y_train, u_train, best_params):
    n, m, d, alpha = best_params

    X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=m, d=d)

    X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
        X_train_matrix, 
        y_train_matrix, 
        test_size=0.2, 
        shuffle=False
    )

    model_reg = Ridge(
        alpha=alpha, 
        fit_intercept=True
    ).fit(X=X_train_matrix_split, y=y_train_matrix_split)

    y_pred = model_reg.predict(X=X_val_matrix_split)

    score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

    return model_reg

def main():
    # Load the data
    u_test = load_data(
        filename=get_absolute_path("u_test.npy")
    )  # Test data for the model
    y_train = load_data(
        filename=get_absolute_path("output_train.npy")
    )  # Expected output for the training data
    u_train = load_data(
        filename=get_absolute_path("u_train.npy")
    )  # Input data for the training

    # best_params = find_best_parameters(y_train=y_train, u_train=u_train)

    # BEST PARAMETERS
    BEST_N=9
    BEST_M=9
    BEST_D=6
    BEST_ALPHA=1e-05
    best_params = (BEST_N, BEST_M, BEST_D, BEST_ALPHA)

    model_reg = chosen_model_with_best_parameters(y_train=y_train, u_train=u_train, best_params=best_params)

    y_pred = recursive_predict(y_train, u_test, model=model_reg, best_params=best_params)

    # Output the last 400 elements of the predicted y_test
    y_pred_last_400 = y_pred[-400:]  # From index 110 to 509 in the test data
    save_npy_to_output("y_pred", y_pred_last_400)


if __name__ == "__main__":
    main()
