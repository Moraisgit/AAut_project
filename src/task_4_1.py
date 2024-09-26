import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore
from typing import Tuple
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    Ridge,
    RANSACRegressor,
    LinearRegression,
)
from utils import get_absolute_path, load_data, get_plot_save_path, save_npy_to_output
from sklearn.model_selection import cross_validate
# from sklearn.utils import shuffle


def plot_training_data(X_train: np.ndarray, individual_plots: bool = False) -> None:
    """
    Plot training data features as scatter plots.

    Parameters:
    X_train (np.ndarray): The training data with features to be plotted.
    individual_plots (bool): If True, save individual plots for each feature;
                             if False, create subplots for all features.

    Returns:
    None: This function saves the plots to files and does not return any value.
    """
    # Sample range corresponding to the number of rows in X_train
    samples = np.arange(start=0, stop=X_train.shape[0], step=1)

    # Titles for each feature plot
    titles_X = [
        "Daily Averages of Air Temperature (x1)",
        "Water Temperature (x2)",
        "Wind Speed (x3)",
        "Wind Direction (x4)",
        "Illumination (x5)",
    ]

    # Check if X_train is a DataFrame and convert to NumPy array if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values  # Convert to NumPy array for consistent indexing

    if individual_plots:
        # Save individual plots for each feature
        for i in range(5):
            plt.figure(figsize=(6, 4))  # Create a new figure for each plot
            plt.scatter(x=samples, y=X_train[:, i], color="blue", s=10)  # Scatter plot
            plt.title(label=titles_X[i])  # Set the title for the plot
            plt.xlabel(xlabel="Sample Index")  # X-axis label
            plt.ylabel(ylabel="Value")  # Y-axis label
            plt.grid(visible=True)  # Add gridlines for better readability

            # Save each plot separately using the image name
            plt.savefig(get_plot_save_path(image_name=f"plot_{i+1}.png"), bbox_inches='tight')
            plt.close()  # Close the plot after saving to free up memory

        print("Individual plots saved as plot_1.png, plot_2.png, ..., plot_5.png")
        print()
    else:
        # Create subplots for all features in a specified layout
        figure, axis = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))  # 3 rows and 2 columns
        figure.suptitle(
            t="Scatter plots of the independent variables after IQR outlier removal",
            fontsize=16,
        )
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between plots

        # Plot the first 4 figures in a 2x2 grid
        for i in range(4):
            row = i // 2
            col = i % 2
            axis[row, col].scatter(samples, X_train[:, i], color="blue", s=10)
            axis[row, col].set_title(titles_X[i])  # Set the title for the subplot
            axis[row, col].set_xlabel("Sample Index")  # X-axis label
            axis[row, col].set_ylabel("Value")  # Y-axis label
            axis[row, col].grid(True)  # Add gridlines for better readability

        # Remove the unused subplot (axis[2,1])
        figure.delaxes(ax=axis[2, 1])

        # Plot the last figure (centered in the last row)
        axis[2, 0].scatter(samples, X_train[:, 4], color="blue", s=10)
        axis[2, 0].set_title(titles_X[4])  # Set title for the last subplot
        axis[2, 0].set_xlabel("Sample Index")  # X-axis label
        axis[2, 0].set_ylabel("Value")  # Y-axis label
        axis[2, 0].grid(True)  # Add gridlines for better readability

        # Save the entire figure to a file
        plt.savefig(get_plot_save_path(image_name="Cleaned_data.png"), bbox_inches='tight')
        print("Figure saved as Cleaned_data.png")
        print()


def plot_outliers_inliers(inlier_mask: np.ndarray, inliers: np.ndarray, outliers: np.ndarray) -> None:
    """
    Plot inliers and outliers of a dataset based on a boolean mask.

    Parameters:
    inlier_mask (np.ndarray): A boolean array where True indicates inliers and False indicates outliers.
    inliers (np.ndarray): Data points classified as inliers.
    outliers (np.ndarray): Data points classified as outliers.

    Returns:
    None: This function saves the generated plots to files and does not return any value.
    """
    # First plot: X from 1 to len(inlier_mask), plotting inliers (blue) and outliers (red)
    X = np.arange(1, len(inlier_mask) + 1)
    
    fig, ax = plt.subplots()
    
    # Initialize indices for inliers and outliers
    inlier_idx = 0
    outlier_idx = 0
    
    # Flags to track whether we've already added the legend for inliers and outliers
    plotted_inlier = False
    plotted_outlier = False
    
    # Iterate through inlier_mask and plot points accordingly
    for i, is_inlier in enumerate(inlier_mask):
        if is_inlier:
            # Plot inliers with blue color
            ax.stem([X[i]], [inliers[inlier_idx]], linefmt='b-', markerfmt='bo', basefmt=" ",
                    label="Inliers" if not plotted_inlier else "")
            inlier_idx += 1
            plotted_inlier = True
        else:
            # Plot outliers with red color
            ax.stem([X[i]], [outliers[outlier_idx]], linefmt='r-', markerfmt='ro', basefmt=" ",
                    label="Outliers" if not plotted_outlier else "")
            outlier_idx += 1
            plotted_outlier = True

    # Add a horizontal line at y=0 to indicate the baseline
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Add legend, title, and labels
    ax.legend()
    ax.set_title(label="Inliers and Outliers")
    ax.set_xlabel(xlabel="Samples")
    ax.set_ylabel(ylabel="Toxic algae concentration")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(get_plot_save_path("Inliers and Outliers"), bbox_inches='tight')

    # Second plot: Inliers only
    X_inliers = np.arange(1, len(inliers) + 1)  # X-axis values for inliers only
    
    fig, ax = plt.subplots()  # Create a new figure for inliers only
    ax.stem(X_inliers, inliers, linefmt='b-', markerfmt='bo', basefmt=" ", label="Inliers")
    
    # Add a horizontal line at y=0 for baseline
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # Set custom limits for the x-axis
    ax.set_xlim([-5, len(inliers) + 10])

    # Add title and labels for the second plot
    ax.set_title(label="Inliers")
    ax.set_xlabel(xlabel="Samples")
    ax.set_ylabel(ylabel="Toxic algae concentration")

    # Adjust layout and save the inliers plot
    plt.tight_layout()
    plt.savefig(get_plot_save_path("Inliers"), bbox_inches='tight')


def remove_outliers_with_ransac(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use RANSAC to fit a model and remove outliers from the dataset.

    Parameters:
    X_train (np.ndarray): The independent variables (features).
    y_train (np.ndarray): The dependent variable (target).

    Returns:
    Tuple[np.ndarray, np.ndarray]: 
        - X_inliers: The filtered independent variables without outliers.
        - y_inliers: The filtered dependent variable without outliers.
        - X_outliers: The outlier independent variables.
        - y_outliers: The outlier dependent variables.
    """
    
    # Create the RANSAC model with a base estimator (LinearRegression is used in this case)
    ransac = RANSACRegressor(estimator=LinearRegression())

    # Fit the RANSAC model on the training data
    ransac.fit(X=X_train, y=y_train)

    # Get the mask that indicates inliers (True) and outliers (False)
    inlier_mask = ransac.inlier_mask_ 
    
    # Separate the training data into inliers and outliers based on the mask
    X_inliers = X_train[inlier_mask]
    y_inliers = y_train[inlier_mask]
    X_outliers = X_train[~inlier_mask]
    y_outliers = y_train[~inlier_mask]

    # Plot the inliers and outliers using the plot_outliers_inliers function
    plot_outliers_inliers(inlier_mask=inlier_mask, inliers=y_inliers, outliers=y_outliers)

    # Print the number of inliers and outliers
    print(f"Number of inliers: {Fore.BLUE}{len(X_inliers)}{Fore.RESET}")
    print(f"Number of outliers: {Fore.BLUE}{len(X_outliers)}{Fore.RESET}")
    print()

    # Return the inliers
    return X_inliers, y_inliers, inlier_mask


def toxic_algae_model(
    X_test: np.ndarray, intercept: float, coefs: np.ndarray
) -> np.ndarray:
    """
    Predict the target values based on the linear regression model.

    Parameters:
    X_test (np.ndarray): New data points (n_samples, n_features).
    intercept (float): The intercept term from the model.
    coefs (np.ndarray): The coefficients (slopes) for each feature from the model.

    Returns:
    np.ndarray: Predicted target values.
    """
    # Calculate the predicted values: ŷ = β0 + Σ (βi * Xi)
    y_pred = intercept + np.dot(X_test, coefs)

    return y_pred


def compute_SEE(y_real: np.ndarray, y_predicted: np.ndarray, inlier_mask: np.ndarray) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) between the actual and predicted values,
    considering only the inliers based on the inlier_mask.

    Parameters:
    y_real (np.ndarray): The actual target values (observed).
    y_predicted (np.ndarray): The predicted target values from the model.
    inlier_mask (np.ndarray): A boolean array indicating inliers (True) and outliers (False).

    Returns:
    float: The Sum of Squared Errors (SSE) for inliers.
    """
    
    # Filter the real and predicted values based on the inlier mask
    y_real_inliers = y_real[inlier_mask]
    y_predicted_inliers = y_predicted[inlier_mask]

    # Compute SSE for the inliers
    return np.sum((y_real_inliers - y_predicted_inliers)**2)


def compare_models(
        X_clean: np.ndarray, 
        y_clean: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray,
        inlier_mask: np.ndarray
) -> None:
    """
    Compare different regression models (Ridge, Lasso, and ElasticNet) on the cleaned dataset.

    Parameters:
    X_clean (np.ndarray): The independent variables (features) after outlier removal.
    y_clean (np.ndarray): The dependent variable (target) after outlier removal.
    """
    # Set number of folds for cross-validation and lambda values for regularization
    NUM_FOLDS = 5
    lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1] # For ElasticNet

    # # Shuffle the clean data before training the models
    # X_clean, y_clean = shuffle(X_clean, y_clean)

    #######################
    # Test Ridge Regression
    #######################
    # Train Ridge Regression with different lambdas
    ridge_reg_avg_scores = []
    for alpha in lambdas:
        ridge_reg = Ridge(alpha=alpha, fit_intercept=True, max_iter=5000).fit(X=X_clean, y=y_clean)
        
        # Perform cross-validation and calculate average score
        ridge_reg_scores = cross_validate(
            estimator=ridge_reg,
            X=X_clean,
            y=y_clean,
            cv=NUM_FOLDS
        )["test_score"]
        ridge_reg_avg_scores.append(sum(ridge_reg_scores) / NUM_FOLDS)
    
    # Find best lambda value and corresponding score for Ridge Regression
    max_ridge_avg_scores = max(ridge_reg_avg_scores)
    max_ridge_lambda = lambdas[ridge_reg_avg_scores.index(max_ridge_avg_scores)]

    # Refit with best found lambda to get access to the best coefs and intercept
    ridge_reg = Ridge(alpha=max_ridge_lambda, fit_intercept=True, max_iter=5000).fit(X=X_clean, y=y_clean)

    # Compute prediction
    ridge_reg_y_pred = toxic_algae_model(
        X_test=X_test, 
        intercept=ridge_reg.intercept_, 
        coefs=ridge_reg.coef_
    )

    # Calculate SSE score for the inliers
    ridge_reg_sse = compute_SEE(y_real=y_train, y_predicted=ridge_reg_y_pred, inlier_mask=inlier_mask)

    # Print results for Ridge Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Ridge Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_ridge_avg_scores} (lambda = {max_ridge_lambda})")
    print(f"\tPrediction score for inliers (SSE) = {ridge_reg_sse} (lambda = {max_ridge_lambda})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {ridge_reg.intercept_}\n\t\tCoefficients: {ridge_reg.coef_}")

    #######################
    # Test Lasso Regression
    #######################
    # Train Lasso Regression with different lambdas
    lasso_reg_avg_scores = []
    for alpha in lambdas:
        lasso_reg = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000).fit(X=X_clean, y=y_clean)
        
        # Perform cross-validation and calculate average score
        lasso_reg_scores = cross_validate(
            estimator=lasso_reg,
            X=X_clean,
            y=y_clean,
            cv=NUM_FOLDS
        )["test_score"]
        lasso_reg_avg_scores.append(sum(lasso_reg_scores) / NUM_FOLDS)
    
    # Find best lambda value and corresponding score for Lasso Regression
    max_lasso_avg_scores = max(lasso_reg_avg_scores)
    max_lasso_lambda = lambdas[lasso_reg_avg_scores.index(max_lasso_avg_scores)]

    # Refit with best found lambda to get access to the best coefs and intercept
    lasso_reg = Lasso(alpha=max_lasso_lambda, fit_intercept=True, max_iter=5000).fit(X=X_clean, y=y_clean)
    
    # Compute prediction
    lasso_reg_y_pred = toxic_algae_model(
        X_test=X_test, 
        intercept=lasso_reg.intercept_, 
        coefs=lasso_reg.coef_
    )

    # Calculate SSE score for the inliers
    lasso_reg_sse = compute_SEE(y_real=y_train, y_predicted=lasso_reg_y_pred, inlier_mask=inlier_mask)

    # Print results for Lasso Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Lasso Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_lasso_avg_scores} (lambda = {max_lasso_lambda})")
    print(f"\tPrediction score for inliers (SSE) = {lasso_reg_sse} (lambda = {max_lasso_lambda})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {lasso_reg.intercept_}\n\t\tCoefficients: {lasso_reg.coef_}")

    ############################
    # Test ElasticNet Regression
    ############################
    # Train ElasticNet with different lambdas and l1_ratios
    elastic_net_reg_avg_scores = []
    max_elastic_net_avg_scores = 0
    for alpha in lambdas:
        for ratio in l1_ratio:
            elastic_net_reg = ElasticNet(alpha=alpha, fit_intercept=True, l1_ratio=ratio, max_iter=6000).fit(X=X_clean, y=y_clean)
            
            # Perform cross-validation and calculate average score
            elastic_net_reg_scores = cross_validate(
                estimator=elastic_net_reg,
                X=X_clean,
                y=y_clean,
                cv=NUM_FOLDS
            )["test_score"]
            elastic_net_reg_avg_scores.append(sum(elastic_net_reg_scores) / NUM_FOLDS)
            
            # Track the best ElasticNet model based on average score
            if max(elastic_net_reg_avg_scores) > max_elastic_net_avg_scores:
                max_elastic_net_lambda = alpha
                max_elastic_net_ratio = ratio

    # Refit with best found lambda to get access to the best coefs and intercept
    elastic_net_reg = ElasticNet(
        alpha=max_elastic_net_lambda, 
        fit_intercept=True, 
        l1_ratio=max_elastic_net_ratio, 
        max_iter=6000
    ).fit(X=X_clean, y=y_clean)

    # Compute prediction
    elastic_net_reg_y_pred = toxic_algae_model(
        X_test=X_test, 
        intercept=elastic_net_reg.intercept_, 
        coefs=elastic_net_reg.coef_
    )

    # Calculate SSE score for the inliers
    elastic_net_reg_sse = compute_SEE(y_real=y_train, y_predicted=elastic_net_reg_y_pred, inlier_mask=inlier_mask)

    # Print results for ElasticNet Regression
    print("---------------------------")
    print("Using " + Fore.YELLOW + "ElasticNet Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max(elastic_net_reg_avg_scores)} (lambda = {max_elastic_net_lambda}, l1_ratio = {max_elastic_net_ratio})")
    print(f"\tPrediction score for inliers (SSE) = {elastic_net_reg_sse} (lambda = {max_elastic_net_lambda}, l1_ratio = {max_elastic_net_ratio})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {elastic_net_reg.intercept_}\n\t\tCoefficients: {elastic_net_reg.coef_}")

    ############################
    # Final comparison of models
    ############################
    # Store the R² scores for comparison
    r_squared = [max_ridge_avg_scores, max_lasso_avg_scores, max_elastic_net_avg_scores]

    # Store the SSE scores for comparison
    sse = [lasso_reg_sse, ridge_reg_sse, elastic_net_reg_sse]
    
    # Determine the best model based on R²
    if max(r_squared) == max_ridge_avg_scores:
        print("---------------------------")
        print(f"Best model is Ridge Regression with R² = {max_ridge_avg_scores}")
    elif max(r_squared) == max_lasso_avg_scores:
        print("---------------------------")
        print(f"Best model is Lasso Regression with R² = {max_lasso_avg_scores}")
    elif max(r_squared) == max_elastic_net_avg_scores:
        print("---------------------------")
        print(f"Best model is ElasticNet Regression with R² = {max_elastic_net_avg_scores}")

     # Determine the best model based on SSE
    if min(sse) == ridge_reg_sse:
        print("---------------------------")
        print(f"Best model is Ridge Regression with SSE = {ridge_reg_sse}")
    elif min(sse) == lasso_reg_sse:
        print("---------------------------")
        print(f"Best model is Lasso Regression with SSE = {lasso_reg_sse}")
    elif min(sse) == elastic_net_reg_sse:
        print("---------------------------")
        print(f"Best model is ElasticNet Regression with SSE = {elastic_net_reg_sse}")

    ###############################
    # Plot Ridge Regression results
    ###############################
    fig1, ax1 = plt.figure(), plt.gca()
    ax1.plot(lambdas, ridge_reg_avg_scores)

    y_min_1, y_max_1 = ax1.get_ylim()
    ax1.set_yticks(np.linspace(y_min_1, y_max_1, 5))
    ax1.set_title("Ridge Regression")
    ax1.set_xlabel("Lambda (λ)")
    ax1.set_ylabel("R²")

    # Save the figure for Ridge Regression
    plt.tight_layout()
    plt.savefig(get_plot_save_path("Ridge Regression R²"), bbox_inches='tight')

    ###############################
    # Plot Lasso Regression results
    ###############################
    fig2, ax2 = plt.figure(), plt.gca()
    ax2.plot(lambdas, lasso_reg_avg_scores)

    y_min_2, y_max_2 = ax2.get_ylim()
    ax2.set_yticks(np.linspace(y_min_2, y_max_2, 5))
    ax2.set_title("Lasso Regression")
    ax2.set_xlabel("Lambda (λ)")
    ax2.set_ylabel("R²")

    # Save the figure for Lasso Regression
    plt.tight_layout()
    plt.savefig(get_plot_save_path("Lasso Regression R²"), bbox_inches='tight')

    # Display the plots
    plt.show()


def chosen_model(X_clean: np.ndarray, y_clean: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Train and evaluate the Ridge Regression model on the cleaned dataset with a chosen regularization parameter.

    Parameters:
    X_clean (np.ndarray): The independent variables (features) after outlier removal.
    y_clean (np.ndarray): The dependent variable (target) after outlier removal.

    Returns:
    Tuple[float, list[float]]: The intercept and coefficients of the Ridge Regression model.
    """
    # Set number of folds for cross-validation and the regularization parameter (alpha)
    NUM_FOLDS = 5
    BEST_ALPHA = 10
    
    # # Shuffle the clean data before training the model
    # X_clean, y_clean = shuffle(X=X_clean, y=y_clean)

    # Chosen model
    ridge_reg = Ridge(alpha=BEST_ALPHA, fit_intercept=True).fit(X=X_clean, y=y_clean)
    
    # Perform cross-validation and store the results
    ridge_reg_scores = cross_validate(
        estimator=ridge_reg,
        X=X_clean,
        y=y_clean,
        cv=NUM_FOLDS
    )["test_score"]

    # Extract the maximum R² score from the cross-validation results
    max_ridge_scores = max(ridge_reg_scores)

    # Print the results of the Ridge Regression model
    print("---------------------------")
    print("Using " + Fore.YELLOW + "Ridge Regression" + Fore.RESET + ":")
    print(f"\tBest average score (R²) = {max_ridge_scores} (lambda = {BEST_ALPHA})")
    print(f"\tModel parameters:")
    print(f"\t\tIntercept: {ridge_reg.intercept_}\n\t\tCoefficients: {ridge_reg.coef_}")

    return ridge_reg.intercept_, ridge_reg.coef_


def main() -> None:
    """
    Main function to execute the workflow for training and evaluating the model.
    It handles loading data, removing outliers, splitting data, and fitting a regression model.
    """
    # Our output will be compared with the teachers output using SSE metric
    X_test = load_data(
        filename=get_absolute_path("X_test.npy")
    )  # Test data for the model
    y_train = load_data(
        filename=get_absolute_path("y_train.npy")
    )  # Expected output for the training data
    X_train = load_data(
        filename=get_absolute_path("X_train.npy")
    )  # Training data for the model

    # Remove outliers from training data
    X_clean, y_clean, inlier_mask = remove_outliers_with_ransac(X_train=X_train, y_train=y_train)

    # # Plot the cleaned training data
    # plot_training_data(X_train=X_clean, individual_plots=True)

    # compare_models(
    #     X_clean=X_clean, 
    #     y_clean=y_clean, 
    #     X_test=X_test, 
    #     y_train=y_train, 
    #     inlier_mask=inlier_mask
    # )

    # Get β0 and βi
    intercept, coefficients = chosen_model(X_clean=X_clean, y_clean=y_clean)

    # Predict using the trained model
    y_pred = toxic_algae_model(X_test=X_test, intercept=intercept, coefs=coefficients)

    # Calculate SSE score for the inliers
    sse = compute_SEE(y_real=y_train, y_predicted=y_pred, inlier_mask=inlier_mask)
    print(f"\tPrediction score for inliers (SSE) = {sse}")

    # Save the predictions to a file
    save_npy_to_output(file_name="y_pred.npy", data=y_pred)


if __name__ == "__main__":
    main()
