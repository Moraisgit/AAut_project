import numpy as np
import matplotlib.pyplot as plt
from utils import get_absolute_path, load_data, save_npy_to_output, get_plot_save_path
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from itertools import product
from sklearn.model_selection import train_test_split
from colorama import Fore
from typing import Tuple


def plot_u_and_y(y: np.array, u: np.array, train: bool = True) -> None:
    """
    Plot the time series data for y and u on the same plot.

    Parameters:
    y (np.array): The target variable data to be plotted.
    u (np.array): The input variable data to be plotted.
    train (bool): Flag to indicate if the plot is for training or testing data. 
                  Defaults to True. Saves the plot with a different name based on this flag.
    
    Returns:
    None
    """
    time = np.arange(len(y))  # Create a time axis based on the length of the arrays

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot u
    ax.plot(time, u, label='Input')

    # Plot y
    ax.plot(time, y, label='Output')

    # Adding titles and labels
    ax.set_title('Input and Output Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    # Save the plot depending on whether it's training or testing data
    if train:
        plt.savefig(get_plot_save_path("Input_Output_train"), bbox_inches='tight')
    else:
        plt.savefig(get_plot_save_path("Input_Output_test"), bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.savefig(get_plot_save_path("u_y_test"), bbox_inches='tight')
    plt.show()


def plot_sse_and_parameters(y_train: np.array, u_train: np.array, best_params: list) -> None:
    """
    Generate and plot the Sum of Squared Errors (SSE) for various combinations of model parameters (n, m, d).

    Parameters:
    y_train (np.ndarray): The actual target values for training.
    u_train (np.ndarray): The feature values for training.
    best_params (tuple): A tuple containing the best parameters (BEST_N, BEST_M, BEST_D, BEST_ALPHA).

    Returns:
    None: Displays plots and saves them as images.
    """
    
    BEST_N, BEST_M, BEST_D, BEST_ALPHA = best_params
    n_values = range(0, 10, 1)  # Possible values for parameter n
    m_values = range(0, 10, 1)  # Possible values for parameter m
    d_values = range(0, 10, 1)  # Possible values for parameter d

    ##########################################
    # Make plot of SSE depending on n, m and d
    ##########################################
    # sse = []
    # combinations = []
    
    # # Iterate over all combinations of n, m, d
    # for n, m, d in product(n_values, m_values, d_values):
    #     X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=m, d=d)

    #     # Skip if the matrix is too small
    #     if X_train_matrix.shape[0] < 10:
    #         continue

    #     # Split data into training and validation sets
    #     X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
    #         X_train_matrix, 
    #         y_train_matrix, 
    #         test_size=0.2, 
    #         shuffle=False
    #     )

    #     # Train the Ridge regression model
    #     model_reg = Ridge(
    #         alpha=BEST_ALPHA, 
    #         fit_intercept=True
    #     ).fit(X=X_train_matrix_split, y=y_train_matrix_split)

    #     # Make predictions on the validation set
    #     y_pred = model_reg.predict(X=X_val_matrix_split)

    #     # Calculate SSE and store it
    #     sse.append(compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred))
    #     combinations.append((n, m, d))

    # # Convert combinations to arrays
    # n_values = np.array([c[0] for c in combinations])
    # m_values = np.array([c[1] for c in combinations])
    # d_values = np.array([c[2] for c in combinations])
    # sse = np.array(sse)

    # # Find the index of the minimum SSE value
    # min_sse_idx = np.argmin(sse)

    # # Set the sizes of the points, making the minimum SSE point bigger
    # sizes = np.full_like(sse, fill_value=20)
    # sizes[min_sse_idx] = 100

    # # Plot SSE values for different combinations of n, m, and d
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # scatter = ax.scatter(n_values, m_values, d_values, c=sse, s=sizes, cmap='viridis')

    # # Set axis labels and title
    # ax.set_xlabel('n')
    # ax.set_ylabel('m')
    # ax.set_zlabel('d')
    # ax.set_title('SSE for combinations of (n, m, d)')

    # # Set limits for each axis
    # ax.set_xlim(-0.5, 9.5)
    # ax.set_ylim(-0.5, 9.5)
    # ax.set_zlim(-0.5, 9.5)

    # # Set ticks for each axis
    # ax.set_xticks(np.arange(0, 10, 1))
    # ax.set_yticks(np.arange(0, 10, 1))
    # ax.set_zticks(np.arange(0, 10, 1))

    # # Add a color bar to show SSE values
    # color_bar = plt.colorbar(scatter, ax=ax)
    # color_bar.set_label('SSE')

    # # Highlight the minimum SSE point
    # ax.scatter(n_values[min_sse_idx], m_values[min_sse_idx], d_values[min_sse_idx], 
    #            color='red', s=150, edgecolors='black', label='Min SSE')

    # ax.legend()
    # # Adjust layout and save the plot
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(get_plot_save_path("SSE_n_m_d"), bbox_inches='tight')

    # Initialize subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    #########################
    # Plot SSE depending on n
    #########################
    sse_n = []
    for n in n_values:
        X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=BEST_M, d=BEST_D)

        if X_train_matrix.shape[0] < 10:
            continue

        X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
            X_train_matrix, y_train_matrix, test_size=0.2, shuffle=False)

        model_reg = Ridge(alpha=BEST_ALPHA, fit_intercept=True).fit(X=X_train_matrix_split, y=y_train_matrix_split)
        y_pred = model_reg.predict(X=X_val_matrix_split)
        sse_n.append(compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred))

    # Plot SSE vs n
    ax[0].plot(n_values, sse_n, marker='o', linestyle='-')
    ax[0].set_xlabel('n')
    ax[0].set_ylabel('SSE')
    ax[0].set_title('SSE over n (m=9, d=6)')
    ax[0].set_xticks(n_values)  # Ensure all n values are ticks
    ax[0].grid(True)
    for spine in ax[0].spines.values():
        spine.set_linewidth(1.5)

    #########################
    # Plot SSE depending on m
    #########################
    sse_m = []
    for m in m_values:
        X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=BEST_N, m=m, d=BEST_D)

        if X_train_matrix.shape[0] < 10:
            continue

        X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
            X_train_matrix, y_train_matrix, test_size=0.2, shuffle=False)

        model_reg = Ridge(alpha=BEST_ALPHA, fit_intercept=True).fit(X=X_train_matrix_split, y=y_train_matrix_split)
        y_pred = model_reg.predict(X=X_val_matrix_split)
        sse_m.append(compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred))

    # Plot SSE vs m
    ax[1].plot(m_values, sse_m, marker='o', linestyle='-')
    ax[1].set_xlabel('m')
    ax[1].set_ylabel('SSE')
    ax[1].set_title('SSE over m (n=9, d=6)')
    ax[1].set_xticks(m_values)  # Ensure all m values are ticks
    ax[1].grid(True)
    for spine in ax[1].spines.values():
        spine.set_linewidth(1.5)

    #################################
    # Plot SSE depending on d
    #################################
    sse_d = []
    for d in d_values:
        X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=BEST_N, m=BEST_M, d=d)

        if X_train_matrix.shape[0] < 10:
            continue

        X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
            X_train_matrix, y_train_matrix, test_size=0.2, shuffle=False)

        model_reg = Ridge(alpha=BEST_ALPHA, fit_intercept=True).fit(X=X_train_matrix_split, y=y_train_matrix_split)
        y_pred = model_reg.predict(X=X_val_matrix_split)
        sse_d.append(compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred))

    # Plot SSE vs d
    ax[2].plot(d_values, sse_d, marker='o', linestyle='-')
    ax[2].set_xlabel('d')
    ax[2].set_ylabel('SSE')
    ax[2].set_title('SSE over d (n=9, m=9)')
    ax[2].set_xticks(d_values)  # Ensure all d values are ticks
    ax[2].grid(True)
    for spine in ax[2].spines.values():
        spine.set_linewidth(1.5)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(get_plot_save_path("SSE_n_m_d_plots"), bbox_inches='tight')

    # Show all plots
    plt.show()


def compute_SEE(y_real: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) between the actual and predicted values.

    Parameters:
    y_real (np.ndarray): The actual target values (observed).
    y_predicted (np.ndarray): The predicted target values from the model.

    Returns:
    float: The Sum of Squared Errors (SSE).
    """

    # Compute SSE for the inliers
    return np.sum((y_real - y_predicted)**2)


def regression_matrix(y: np.array, u: np.array, n: int, m: int, d: int) -> Tuple[np.array, np.array]:
    """
    Constructs a regression matrix based on past values of the dependent variable y and the independent variable u.

    Parameters:
    y (np.array): The dependent variable values (observed).
    u (np.array): The independent variable values (input).
    n (int): The n parameter.
    m (int): The m parameter.
    d (int): The d parameter.

    Returns:
    Tuple[np.array, np.array]:
        - X (np.array): The regressor matrix X
        - Y (np.array): The output vector Y
    """
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


def find_parameters_and_compare_models(y_train: np.array, u_train: np.array) -> None:
    """
    Finds the best hyperparameters for various regression models by comparing 
    their performance on training data.

    Parameters:
    y_train (np.array): The dependent variable training data.
    u_train (np.array): The independent variable training data.

    Returns:
    None: The function prints the best parameters and scores for each regression model.
    """
    
    # Define ranges for parameters and hyperparameters
    n_values = range(0, 15, 1)
    m_values = range(0, 15, 1)
    d_values = range(0, 15, 1)
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # l1_ratio_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

    # Initialize best scores and parameters list for each model
    best_score_linear = np.inf
    best_score_ridge = np.inf
    best_score_lasso = np.inf
    # best_score_elastic_net = np.inf
    r2_score_linear = 0
    r2_score_ridge = 0
    r2_score_lasso = 0
    r2_score_elastic_net = 0
    best_params_linear = None
    best_params_ridge = None
    best_params_lasso = None
    # best_params_elastic_net = None

    # Iterate through all combinations of n, m, d, and alpha
    for n, m, d, alpha in product(n_values, m_values, d_values, lambda_values):
    # # For faster testing
    # n=9
    # m=9
    # d=6
    # for alpha in lambda_values:
        # Create the regression matrix for the current parameter set
        X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=m, d=d)

        # Skip if there are not enough samples to split - 10 could be another arbitrary number
        if X_train_matrix.shape[0] < 10:
            continue

        # Split the dataset into training and validation sets
        X_train_matrix_split, X_val_matrix_split, y_train_matrix_split, y_val_matrix_split = train_test_split(
            X_train_matrix, 
            y_train_matrix, 
            test_size=0.2, 
            shuffle=False
        )

        ########################
        # Test Linear Regression
        ########################
        linear_reg = LinearRegression(fit_intercept=True)
        linear_reg.fit(X=X_train_matrix_split, y=y_train_matrix_split)

        # Predict and compute the score
        y_pred = linear_reg.predict(X=X_val_matrix_split)
        score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

        # Update best score and parameters if the current score is better
        if score < best_score_linear:
            best_score_linear = score
            best_params_linear = (n, m, d)
            r2_score_linear = r2_score(y_pred=y_pred, y_true=y_val_matrix_split)

        #######################
        # Test Ridge Regression
        #######################
        ridge_reg = Ridge(alpha=alpha, fit_intercept=True)
        ridge_reg.fit(X=X_train_matrix_split, y=y_train_matrix_split)

        # Predict and compute the score
        y_pred = ridge_reg.predict(X=X_val_matrix_split)
        score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

        # Update best score and parameters if the current score is better
        if score < best_score_ridge:
            best_score_ridge = score
            best_params_ridge = (n, m, d, alpha)
            r2_score_ridge = r2_score(y_pred=y_pred, y_true=y_val_matrix_split)

        #######################
        # Test Lasso Regression
        #######################
        lasso_reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000)
        lasso_reg.fit(X=X_train_matrix_split, y=y_train_matrix_split)

        # Predict and compute the score
        y_pred = lasso_reg.predict(X=X_val_matrix_split)
        score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

        # Update best score and parameters if the current score is better
        if score < best_score_lasso:
            best_score_lasso = score
            best_params_lasso = (n, m, d, alpha)
            r2_score_lasso = r2_score(y_pred=y_pred, y_true=y_val_matrix_split)

        ################################################
        # Test ElasticNet Regression - THIS DOES NOT RUN
        ################################################
        # for l1_ratio in l1_ratio_values:
        #     elastic_net_reg = ElasticNet(
        #         alpha=alpha, 
        #         l1_ratio=l1_ratio,
        #         fit_intercept=True,
        #         max_iter=50000
        #     )
        #     elastic_net_reg.fit(X=X_train_matrix_split, y=y_train_matrix_split)

        #     # Predict and compute the score
        #     y_pred = elastic_net_reg.predict(X=X_val_matrix_split)
        #     score = compute_SEE(y_real=y_val_matrix_split, y_predicted=y_pred)

        #     # Update best score and parameters if the current score is better
        #     if score < best_score_elastic_net:
        #         best_score_elastic_net = score
        #         best_params_elastic_net = (n, m, d, alpha, l1_ratio)
        #         r2_score_elastic_net = r2_score(y_pred=y_pred, y_true=y_val_matrix_split)

    # Print results for Linear Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Linear Regression" + Fore.RESET + ":")
    print(f"\tBest parameters:")
    print(f"\t\tn = {best_params_linear[0]} \n\t\tm = {best_params_linear[1]} \n\t\td = {best_params_linear[2]}")
    print(f"\tPrediction score (SSE) = {best_score_linear}")
    print(f"\tPrediction score (R²) = {r2_score_linear}")

    # Print results for Ridge Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Ridge Regression" + Fore.RESET + ":")
    print(f"\tBest parameters:")
    print(f"\t\tn = {best_params_ridge[0]} \n\t\tm = {best_params_ridge[1]} \n\t\td = {best_params_ridge[2]} \n\t\tlambda = {best_params_ridge[3]}")
    print(f"\tPrediction score (SSE) = {best_score_ridge}")
    print(f"\tPrediction score (R²) = {r2_score_ridge}")

    # Print results for Lasso Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Lasso Regression" + Fore.RESET + ":")
    print(f"\tBest parameters:")
    print(f"\t\tn = {best_params_lasso[0]} \n\t\tm = {best_params_lasso[1]} \n\t\td = {best_params_lasso[2]} \n\t\tlambda = {best_params_lasso[3]}")
    print(f"\tPrediction score (SSE) = {best_score_lasso}")
    print(f"\tPrediction score (R²) = {r2_score_lasso}")

    # # Print results for ElasticNet Regression
    # print("---------------------------")
    # print("Using " + Fore.YELLOW + "ElasticNet Regression" + Fore.RESET + ":")
    # print(f"\tBest parameters:")
    # print(f"\t\tn = {best_params_elastic_net[0]} \n\t\tm = {best_params_elastic_net[1]} \n\t\td = {best_params_elastic_net[2]} \n\t\tlambda = {best_params_elastic_net[3]} \n\t\tl1_ratio = {best_params_elastic_net[4]}")
    # print(f"\tPrediction score (SSE) = {best_score_elastic_net}")
    # print(f"\tPrediction score (R²) = {r2_score_elastic_net}")


def recursive_predict(y_train: np.array, u_test: np.array, model, best_params: list) -> np.array:
    """
    Predict future values using a recursive approach based on the trained model 
    and provided input data.

    Parameters:
    y_train (np.array): The training target values.
    u_test (np.array): The test input values to be used for predictions.
    model: The trained regression model used for predictions.
    best_params (list): A list containing the best parameters [n, m, d, alpha] 
                        used for constructing the prediction.

    Returns:
    np.array: An array of predicted values for the test set.
    """
    n, m, d, _ = best_params  # Unpack parameters for prediction

    N = len(u_test)  # Length of the test input array
    p = max(n, d + m)  # Calculate the required past values for prediction

    # Initialize the predicted output array with zeros
    y_pred = np.zeros(N)

    # Set initial conditions using the last p values from the training target values
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

    return y_pred


def chosen_model_with_best_parameters(y_train: np.array, u_train: np.array, best_params: list):
    """
    Train a regression model using the specified best parameters and evaluate 
    its performance on a validation set.

    Parameters:
    y_train (np.array): The training target values.
    u_train (np.array): The training input values used for predictions.
    best_params (list): A list containing the best parameters [n, m, d, alpha] 
                        for constructing the regression model.

    Returns:
    model_reg: The trained Ridge Regression model.
    """
    n, m, d, alpha = best_params  # Unpack the best parameters

    # Create the regression matrix
    X_train_matrix, y_train_matrix = regression_matrix(y=y_train, u=u_train, n=n, m=m, d=d)

    # Ridge Regression model
    model_reg = Lasso(
        alpha=alpha, 
        fit_intercept=True,
        max_iter=20000
    ).fit(X=X_train_matrix, y=y_train_matrix)

    # Print the results for the Lasso Regression model
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Lasso Regression" + Fore.RESET + ":")
    print(f"\tBest parameters:")
    print(f"\t\tn = {best_params[0]} \n\t\tm = {best_params[1]} \n\t\td = {best_params[2]} \n\t\tlambda = {best_params[3]}")
    print(f"\tModel parameters:")
    print(f"\t\tCofficients = {model_reg.coef_}")

    return model_reg


def main():
    """
    Main function to execute the workflow for training and evaluating the model.
    """
    # Load the data
    u_test = load_data(
        filename=get_absolute_path("u_test.npy")
    )  # Test input data for the model
    y_train = load_data(
        filename=get_absolute_path("output_train.npy")
    )  # Expected output for the training data
    u_train = load_data(
        filename=get_absolute_path("u_train.npy")
    )  # Train input data for the model

    # # Plot the training input and output
    # plot_u_and_y(y=y_train, u=u_train, train=True)

    # # Find the best parameters for different regression models
    # find_parameters_and_compare_models(y_train=y_train, u_train=u_train)

    ##########################
    # BEST PARAMETERS OBTAINED
    ##########################
    BEST_N=9
    BEST_M=9
    BEST_D=6
    BEST_ALPHA=0.0001 # Best lambda for Lasso
    best_params = (BEST_N, BEST_M, BEST_D, BEST_ALPHA)

    # # Plot the SSE and selected parameters
    # plot_sse_and_parameters(y_train=y_train, u_train=u_train, best_params=best_params)

    # Train the chosen regression model with the best parameters
    model_reg = chosen_model_with_best_parameters(y_train=y_train, u_train=u_train, best_params=best_params)

    # Use the trained model to predict the output for the test input using recursive prediction
    y_pred = recursive_predict(y_train, u_test, model=model_reg, best_params=best_params)

    # # Plot the predicted test output and input
    # plot_u_and_y(y=y_pred, u=u_test, train=False)

    # Output the last 400 elements of the predicted y_test
    y_pred_last_400 = y_pred[-400:]  # From index 110 to 509 in the test data
    print("Shape of the predicted output =", y_pred_last_400.shape)
    save_npy_to_output(file_name="y_pred", data=y_pred_last_400)


if __name__ == "__main__":
    main()