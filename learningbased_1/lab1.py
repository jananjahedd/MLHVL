"""
Authors: Diana Luna, Janan Jahed
Filename: lab1.py
Description: Functions to complete exercise 1 of Learning-based Lab Assignment #1 - Image recognition using deep networks.
"""
# ***************** I HAVE TO CLEAN CODE ***********************

from tensorflow import keras
import matplotlib.pyplot as plt
#import kagglehub
import numpy as np

# download latest version of MNIST dataset
# path = kagglehub.dataset_download("hojjatk/mnist-dataset")
# print("Path to dataset files:", path)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#xtrain = np.reshape(x_train, (x_train.shape[0], 28*28))
#xtest = np.reshape(x_test, (x_test.shape[0], 28*28))
xtrain = np.reshape(x_train, (60000, 28, 28, 1))
xtest = np.reshape(x_test, (10000, 28, 28, 1))

print(xtrain.shape, xtest.shape)

xtrain = np.array(xtrain)/255
xtest = np.array(xtest)/255

ytrain = keras.utils.to_categorical(y_train, 10)
ytest = keras.utils.to_categorical(y_test, 10)
print (ytrain[0])

#model = keras.Sequential()
#model.add(keras.layers.Dense(256, input_shape=(784,), activation='relu'))
#model.add(keras.layers.Dense(10, activation='softmax'))
#model.summary()

## EXC 6-7
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
#model.add(keras.layers.Dropout(rate=0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu"))
#model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

#model.compile(loss='categorical_crossentropy',
#optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

## EXC 6-7
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(learning_rate=float(1)), 
              metrics=['accuracy'])
         
#history = model.fit(xtrain, ytrain, batch_size=128, epochs=12, verbose=1, validation_split=0.2)

## EXC 6-7
history = model.fit(xtrain, ytrain, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Plot the loss curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=1)
plt.show()

# Plot the Accuracy Curves
plt.figure(figsize=[8, 6]) 
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16) 
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

