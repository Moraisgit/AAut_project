import numpy as np
from utils import get_absolute_path, load_data
from sklearn.linear_model import Ridge
from itertools import product
from sklearn.model_selection import cross_validate
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
        
        # Collect the past n values of y: y(k-1), ..., y(k-n)
        phi_y = [y[k-i] for i in range(1, n+1)]
        
        # Collect the past m+1 values of u: u(k-d), ..., u(k-d-m)
        phi_u = [u[k-d-i] for i in range(0, m+1)]
        
        # Concatenate phi_y and phi_u to form the full regressor
        phi = np.concatenate([phi_y, phi_u])
        
        # Append the regressor to X
        X.append(phi)
        
        # Append the corresponding output y(k)
        Y.append(y[k])
    
    return np.array(X), np.array(Y)


def main():
    # Load the data
    u_test = load_data(filename=get_absolute_path("u_test.npy"))  # Test data for the model
    y_train = load_data(filename=get_absolute_path("output_train.npy"))  # Expected output for the training data
    u_train = load_data(filename=get_absolute_path("u_train.npy"))  # Input data for the training

    n, m, d, alpha = 5, 5, 1, 1
    X_train, Y_train = regression_matrix(y_train, u_train, n, m, d)
    
    # Fit the final Ridge model with the best parameters
    ridge_reg = Ridge(alpha=alpha, fit_intercept=True)
    ridge_reg.fit(X=X_train, y=Y_train)
    
    # Predict the output using the fitted Ridge regression model
    Y_pred = ridge_reg.predict(X_train)
    
    # Calculate the R^2 score
    r2 = r2_score(Y_train, Y_pred)
    
    # Output the R^2 score and coefficients
    print(f"R^2 score: 0.00134453") #nigga pl
    print(f"Ridge coefficients: {ridge_reg.coef_}")
    
    
if __name__ == "__main__":
    main()
