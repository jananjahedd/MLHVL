"""
Authors: Diana Luna, Janan Jahed
Filename: challenge_based.py
Description: A CNN model for CIFAR-10 object recognition task
"""

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


def build_cifar10_model():
    """
    CNN for CIFAR-10 object recognition

    Architecture:
        Conv2D (32 filters) -> Conv2D (32 filters) -> MaxPool -> Dropout
        Conv2D (32 filters) -> Conv2D (32 filters) -> MaxPool -> Dropout
        Flatten -> Dense (512) -> Dropout
        Dense (10, softmax)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 3),
        padding="same"
    ))
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    optimizer = keras.optimizers.RMSprop(
        learning_rate=float(0.0001),
        weight_decay=1e-6
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def plot_training_history(history):
    """
    plot the train and val accuracy and loss

    Args:
        history: the keeras History object from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.savefig("cifar10_history.png", dpi=200)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"train shape: {x_train.shape}")
    print(f"train label shape: {y_train.shape}")
    print(f"test shape: {x_test.shape}")
    print(f"test label: {y_test.shape}")
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print("\nAfter preprocessing:")
    print(f"x_train range: [{x_train.min()}, {x_train.max()}]")
    print(f"y_train shape: {y_train.shape}")
    model = build_cifar10_model()
    print("\nSummary:")
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=20,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=True
    )
    plot_training_history(history)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nfinal test Loss: {loss:.4f}")
    print(f"final test Accuracy: {accuracy:.4f}")
