"""
Authors: Diana Luna, Janan Jahed
Filename: lab1.py
Description: Functions to complete exercise 1 of Learning-based Lab Assignment
#1 - Image recognition using deep networks.
"""

from tensorflow import keras
import matplotlib.pyplot as plt
import kagglehub 
import numpy as np


def download_mnist_local():
    """
    Function that downloads the latest version of MNIST database
    Stores it locally in your pc, it just shows you where it is stored

    Returns:
        Print statement for you to know in which path its store the db
    """
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)


def mnist_train_test():
    """
    Function that stores the MNIST DB in train and test tuples

    Returns:
        The x_train, y_train, x_test and y_test test tuples loaded with MNIST
        data
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def linear_activation(x_train, y_train, x_test, y_test):
    """
    Function that applies linear activation (default) to the model
    (fully connected network)

    Arguments:
        x_train, y_train, x_test, y_test: test tuples loaded with MNIST data

    Returns:
        Plots for loss and accuracy, plus print statements
    """

    # data preparation

    # flatenning 2 spatial dimensions - 60000x28x28 to 60000x784
    xtrain = np.reshape(x_train, (x_train.shape[0], 28*28))
    xtest = np.reshape(x_test, (x_test.shape[0], 28*28))

    print(xtrain.shape, xtest.shape)

    # rescale values between 0 and 1 by dividing them by 255
    xtrain = np.array(xtrain)/255
    xtest = np.array(xtest)/255

    # convert train and testset labels -> separate network units
    ytrain = keras.utils.to_categorical(y_train, 10)
    ytest = keras.utils.to_categorical(y_test, 10)
    print(ytrain[0], ytest[0])

    # model definition
    # mlp - 784 input units into 256 units
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_shape=(784,)))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    # trainning and evaluation

    # 12 epochs, random 20%, 128 image label pairs
    history = model.fit(xtrain, ytrain, batch_size=128, epochs=12, verbose=1,
                        validation_split=0.2)

    plots(history)
    evaluate_model(model, xtest, ytest)
    # result of model evaluation is: 0.2788776755332947 0.9218000173568726
    # loss should be around 0.28 and accuracy around 0.92


def relu_activation(x_train, y_train, x_test, y_test):
    """
    Function that applies rectified (relu) activation to the model
    (fully connected network)

    Arguments:
        x_train, y_train, x_test, y_test: test tuples loaded with MNIST data

    Returns: Plots for loss and accuracy, plus print statements
    """

    # data preparation
    # flatenning 2 spatial dimensions - 60000x28x28 to 60000x784
    xtrain = np.reshape(x_train, (x_train.shape[0], 28*28))
    xtest = np.reshape(x_test, (x_test.shape[0], 28*28))

    print(xtrain.shape, xtest.shape)

    # rescale values between 0 and 1 by dividing them by 255
    xtrain = np.array(xtrain)/255
    xtest = np.array(xtest)/255

    # convert train and testset labels -> separate network units
    ytrain = keras.utils.to_categorical(y_train, 10)
    ytest = keras.utils.to_categorical(y_test, 10)
    print(ytrain[0], ytest[0])

    # model definition
    # mlp - 784 input units into 256 units
    # specify rectified activation in the 1st hidden layer
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_shape=(784,), activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

    # trainning and evaluation
    # 12 epochs, random 20%, 128 image label pairs
    history = model.fit(xtrain, ytrain, batch_size=128, epochs=12, verbose=1,
                        validation_split=0.2)

    plots(history)

    evaluate_model(model, xtest, ytest)
    # result of model evaluation is: 0.07321187853813171 0.9794999957084656

    """
    Function that applies rectified (relu) activation to the DCN model

    Arguments:
        x_train, y_train, x_test, y_test: test tuples loaded with MNIST data

    Returns: Plots for loss and accuracy, plus print statements
    """

    # data preparation
    # flatenning 3 spatial dimensions - reshape to 60000,28,28,1 and
    # 10000,28,28,1
    xtrain = np.reshape(x_train, (60000, 28, 28, 1))
    xtest = np.reshape(x_test, (10000, 28, 28, 1))

    print(xtrain.shape, xtest.shape)

    # rescale values between 0 and 1 by dividing them by 255
    xtrain = np.array(xtrain)/255
    xtest = np.array(xtest)/255

    # convert train and testset labels -> separate network units
    ytrain = keras.utils.to_categorical(y_train, 10)
    ytest = keras.utils.to_categorical(y_test, 10)
    print(ytrain[0], ytest[0])

    # model definition
    # dcn - 32 filters into 64, 3x3 pixel filter
    # specify rectified activation in the 1st hidden layer
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=float(1)),
                  metrics=['accuracy'])

    # trainning and evaluation
    # 12 epochs, random 20%, 128 image label pairs
    history = model.fit(xtrain, ytrain, batch_size=128, epochs=6, verbose=1,
                        validation_split=0.2)

    plots(history)
    evaluate_model(model, xtest, ytest)
    # result of model evaluation is: 0.04101130738854408 0.9873999953269958

def deep_conv_networks(x_train, y_train, x_test, y_test):
    """
    Function that applies the DCN model

    Arguments:
        x_train, y_train, x_test, y_test: test tuples loaded with MNIST data

    Returns: Plots for loss and accuracy, plus print statements
    """

    # data preparation
    # flatenning 3 spatial dimensions - reshape to 60000,28,28,1 and
    # 10000,28,28,1
    xtrain = np.reshape(x_train, (60000, 28, 28, 1))
    xtest = np.reshape(x_test, (10000, 28, 28, 1))

    print(xtrain.shape, xtest.shape)

    # rescale values between 0 and 1 by dividing them by 255
    xtrain = np.array(xtrain)/255
    xtest = np.array(xtest)/255

    # convert train and testset labels -> separate network units
    ytrain = keras.utils.to_categorical(y_train, 10)
    ytest = keras.utils.to_categorical(y_test, 10)
    print(ytrain[0], ytest[0])

    # model definition
    # dcn - 32 filters into 64, 3x3 pixel filter
    # specify rectified activation in the 1st hidden layer
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=float(1)),
                  metrics=['accuracy'])

    # trainning and evaluation
    # 12 epochs, random 20%, 128 image label pairs
    history = model.fit(xtrain, ytrain, batch_size=128, epochs=6, verbose=1,
                        validation_split=0.2)

    plots(history)
    evaluate_model(model, xtest, ytest)
    # result of model evaluation is: 0.04101130738854408 0.9873999953269958

def deep_conv_networks_dropout(x_train, y_train, x_test, y_test):
    """
    Function that applies the DCN model using dropout method

    Arguments:
        x_train, y_train, x_test, y_test: test tuples loaded with MNIST data

    Returns: Plots for loss and accuracy, plus print statements
    """

    # data preparation
    # flatenning 3 spatial dimensions - reshape to 60000,28,28,1 and
    # 10000,28,28,1
    xtrain = np.reshape(x_train, (60000, 28, 28, 1))
    xtest = np.reshape(x_test, (10000, 28, 28, 1))

    print(xtrain.shape, xtest.shape)

    # rescale values between 0 and 1 by dividing them by 255
    xtrain = np.array(xtrain)/255
    xtest = np.array(xtest)/255

    # convert train and testset labels -> separate network units
    ytrain = keras.utils.to_categorical(y_train, 10)
    ytest = keras.utils.to_categorical(y_test, 10)
    print(ytrain[0], ytest[0])

    # model definition
    # dcn - 32 filters into 64, 3x3 pixel filter
    # specify rectified activation in the 1st hidden layer
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=float(1)),
                  metrics=['accuracy'])

    # trainning and evaluation
    # 12 epochs, random 20%, 128 image label pairs
    history = model.fit(xtrain, ytrain, batch_size=128, epochs=6, verbose=1,
                        validation_split=0.2)

    plots(history)
    evaluate_model(model, xtest, ytest)
    # result of model evaluation is: 0.03073989227414131 0.9902999997138977


def plots(history):
    """
    Function in charge of generating all the three different plots needed in
    this first part
    Generate plots for loss and accuracy

    Arguments:
        history: the model already fitted

    Returns:
        Plots both for loss and accuracy
    """
    # generate plots - loss and accuracy
    # plot - loss curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # plot - accuracy curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()


def evaluate_model(model, xtest, ytest):
    """
    Function that evaluate model (loss and accuracy)

    Arguments:
        model: corresponding model that is going to be evaluated
        xtest, ytest: test set tuples to evaluate performance in model

    Returns:
        loss, accuracy: loss and accuracy of the model's performance based on
        test set
    """
    loss, accuracy = model.evaluate(xtest, ytest, verbose=0)
    print(f"Linear activation result: {loss} {accuracy}")
    return loss, accuracy


# main to call all functions
if __name__ == "__main__":
    # in case you don't have MNIST DB
    download_mnist_local()
    # obtain laoded data in test tuples
    x_train, y_train, x_test, y_test = mnist_train_test()
    # questions 1-3
    linear_activation(x_train, y_train, x_test, y_test)
    print("\n---------------------------------------\n")
    # questions 4-5
    relu_activation(x_train, y_train, x_test, y_test)
    print("\n---------------------------------------\n")
    # question 6
    deep_conv_networks(x_train, y_train, x_test, y_test)
    print("\n---------------------------------------\n")
    # question 7
    deep_conv_networks_dropout(x_train, y_train, x_test, y_test)
