# General

submission
zip file with
- python code
- y hat -test numpy vector

x_train.numpy
x_train 200x5

# 4. Part 1 - Regression with Synthetic Data

## 4.1 First Problem - Multiple Linear Regression with Outliers

y_train.npy is the output for the training data x_train.npy

x_text.npy we should not use, we only use to compute the predictions of the model we believe is the best one


we need to worry about outlier removal and regularization techniques


information about the outliers:
- we are told the aprox %, 25% of the data are outliers, noise
- we only have outliers in y, noise in x is negligeble

regularization techniques:
- lasso
- ridge
- ...

we have to try different outlier removal techniques and regularization techniques (to make the best model)

how to choose the best model:
- instead of using the entire training set we take a bit of the training set to make a validation set and evaluate the model

maybe: training set -> 150x5
        validation set -> 50x5


we can plot data like y in function of x OR y in function of the index (i)

if we identifie an outlier we need to remove both in x and y_train

the validation set might also have outliers so we also need to remove them

we cant identifie outliers just by looking at y values (big values)


first we remove outliers, then we split training data in train and validation, and then we 

## 4.2  Second Problem - The ARX model

Estimate the theta parameters of the model based on Y and X

The data that will be available is not the X and Y. We need to create X and Y based on the u's

output at time k depends on previous inputs and outputs

delay parameters are m, n, d are between 0<=m, n, d<10

as an example, n=2,d=0, m=3:
- y(3)=-a1y(2)-a2y(1)+b0u(3)+b1u(2)+b2u(1)
- y(4)=-a1y(3)-a2y(2)+b0u(4)+b1u(3)+b2u(2)

Y=X\theta

y_hat needs to be size 400x1 

the pred for the train can be at once beacause we have thetas but for the y_pred_test needs to be one by one, obtained iteratively

# A fazer para este lab:

Separar os dados de treino em treino e validação SEM CROSS-VALIDATION porque a ordem da data é importante


# 5. Part 2 - Image Analysis

## 5.1 First Problem - Image classification

## 5.2 Second Problem - image segmentation
