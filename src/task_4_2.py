import numpy as np
from utils import get_absolute_path, load_data, save_npy_to_output
from sklearn.linear_model import Ridge
from itertools import product
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import r2_score


# Function to construct the regression matrix
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


def teste_alphas(X_train: np.ndarray, Y_train: np.ndarray):

    lambdas = [0.5, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0]
    NUM_FOLDS = 8

    ridge_reg_avg_scores = []
    for alpha in lambdas:
        ridge_reg = Ridge(alpha=alpha, fit_intercept=True).fit(X=X_train, y=Y_train)

        # Perform cross-validation and calculate average score
        ridge_reg_scores = cross_validate(
            estimator=ridge_reg, X=X_train, y=Y_train, cv=NUM_FOLDS
        )["test_score"]
        ridge_reg_avg_scores.append(sum(ridge_reg_scores) / NUM_FOLDS)

    # Find best lambda value and corresponding score for Ridge Regression
    max_ridge_avg_scores = max(ridge_reg_avg_scores)
    max_ridge_lambda = lambdas[ridge_reg_avg_scores.index(max_ridge_avg_scores)]

    # Refit with best found lambda to get access to the best coefs and intercept
    ridge_reg = Ridge(alpha=max_ridge_lambda, fit_intercept=True, max_iter=5000).fit(
        X=X_train, y=Y_train
    )
    print(
        f"\tBest average score (R²) = {max_ridge_avg_scores} (lambda = {max_ridge_lambda})"
    )


# Function to test different combinations of n, m, d with a constant alpha
def tune_nmd(y_train, u_train, alpha, n_range, m_range, d_range):
    best_score = -np.inf
    best_params = None

    # Iterate over all combinations of n, m, and d
    for n, m, d in product(n_range, m_range, d_range):
        X_train, Y_train = regression_matrix(y_train, u_train, n, m, d)

        # Train the Ridge regression model with a constant alpha
        ridge_reg = Ridge(alpha=alpha, fit_intercept=True)

        # Perform 5-fold cross-validation and compute mean R² score
        r2 = np.mean(cross_val_score(ridge_reg, X_train, Y_train, cv=5, scoring="r2"))

        # Track the best combination of parameters
        if r2 > best_score:
            best_score = r2
            best_params = (n, m, d)

    return best_params, best_score


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

    n, m, d, alpha = 9, 9, 6, 0
    X_train, Y_train = regression_matrix(y_train, u_train, n, m, d)

    # Fit the final Ridge model with the best parameters
    ridge_reg = Ridge(alpha=alpha, fit_intercept=True)
    ridge_reg.fit(X=X_train, y=Y_train)

    # Predict the output using the fitted Ridge regression model
    Y_pred = ridge_reg.predict(X_train)

    # Calculate the R^2 score
    r2 = r2_score(Y_train, Y_pred)

    # Output the R^2 score and coefficients
    print(f"R^2 score: {r2}")
    print(f"Ridge coefficients: {ridge_reg.coef_}")

    # Define the range of parameters to try
    # n_range = range(1, 10)  # n from 1 to 9
    # m_range = range(1, 10)  # m from 1 to 9
    # d_range = range(0, 10)  # d from 0 to 9

    # Define a constant alpha value
    # alpha = 0.5

    # Tune n, m, d parameters
    # best_params, best_score = tune_nmd(y_train, u_train, alpha, n_range, m_range, d_range)
    # print(f"\nBest parameters (n, m, d): {best_params}, Best R²: {best_score:.5f}")


if __name__ == "__main__":
    main()
