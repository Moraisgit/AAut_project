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

to view the image 

```python
im=np.reshape(kon, (48,48))
plt.imshow(im)
```

the extra dataset does not have labels, we should use it

we need to submit labels and not probabilities output

### Neural networks:

- library: tensorflow, keras libraries

- define NN: number of units for each layer; activation function (RELU, sigmoid, ...); specify the input form, specially in the first layer; specify a output layer - if we are doing regression then linear, if we are doing classification then sigmoid (binary classification problem - OUR CASE) or softmax (more than 2 classes)

- training, speficify gradient descent

- - loss function: regression (SSE); classification (binary cross entropy - binary classification which means 2 classes, categorical cross entropy which means more than 2 classes)

- - optimizer: sgd; adam (USE THIS ONE); online; mini-batch; batch. Specify learning rate

we have training data, split into validation

standerdize/scale the input to instead of being 0 to 255, to be 0 to 1. Do this for training, validation and testing

max number of iterations

plot the loss function in function of the epoch number. it should look like a curvy triangle in the origin (for the training loss), the validation loss should also look like that. If the valitation loss starts to increase, it means overfitting - maybe we do not have enough training data, or simplify the model cuz it may be too complex for the training data - layers of number of units

we want the best weights for the validation set, before the validation loss increases (use max iterations). callback early stopping? (specify the patience parameter), callback keepbest? (keep the best weights for the validation)

`model.predict` will give probabilitie outputs, we need to use `np.argmax` to convert to labels

```python
train_images=(X).astype('float32')/255.0
train_labels = keras.utils.to_categorical(y,2) # One hot encoding of class labels
test_images = (X).astype('float32')/255.0
results_MLP = np.argmax(model_MLP.predict(test_images),1)
```

### Imbalance

Imbalance 90% class 1 and 10% class 0 - it will almost always predict class 1 and not class 0. To know if we took care of imbalance, we should look at f1-score and balanced accuracy


## 5.2 Second Problem - image segmentation

Image segmentation problem. Given some input images 48x48 grayscale we want as output a binary image 48x48 where 0 is the background and 1 is the forgound (craters). In essence classifie each pixel of the image, give a label to each individual pixel.

We are given two formats of data to work with.

- Fomart A: each row is one pixel and the number of features is 49. For each pixel we provide the 7x7 neighborhood. A 7x7 patch. Each block of 1764 rows corresponds to an image, 1764 < 48x48 because the outer pixels are not included as there would be no neighborhood. The y output is the label of each pixel. With the function `reconstruct_from_patches_2d` we do `(rows,(48,48))` to recreate each block in a image. Before using the function reshape to (1764,7,7).

- Format B: each row is an image 48x48. The y output is the label for each pixel. We can use `extract_from_patches_2d`.