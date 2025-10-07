"""
Authors: Diana Luna, Janan Jahed
Filename: lab2.py
Description: The functions required to complete part 2 of assignemen 1.
"""
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from numpy.lib.stride_tricks import as_strided


def image_to_column(input_layer, filter_height, filter_width):
    """
    Args:
        input_layer: Input image with shape (height, width, channels)
        filter_height: Height of the convolution filter
        filter_width: Width of the convolution filter

    Returns:
        Reshaped patches for matrix multiplication

    Citation:
    https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
    """
    image_height, image_width, num_channels = input_layer.shape
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    stride_height, stride_width, stride_channel = input_layer.strides
    view_shape = (output_height, output_width, filter_height, filter_width,
                  num_channels)
    view_strides = (stride_height, stride_width, stride_height, stride_width,
                    stride_channel)
    patches = as_strided(input_layer, shape=view_shape, strides=view_strides)
    patch_size = filter_height * filter_width * num_channels
    num_patches = output_height * output_width
    return patches.reshape(num_patches, patch_size).T


def convolve(input_layer, filters):
    """
    Apply the convolution operation to input layer with given filters

    Args:
        input_layer: Input image with shape (height, width, channels)
        filters: Convolution filters with shape (num_filters, height, width
        channels)

    Returns:
        Feature maps with shape (output_height, output_width, num_filters)
    """
    image_height, image_width, num_input_channels = np.shape(input_layer)
    num_filters, filter_height, filter_width, _ = np.shape(filters)
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    patch_size = filter_height * filter_width * num_input_channels
    reshaped_filter = filters.reshape(num_filters, patch_size).T
    image_column = image_to_column(input_layer, filter_height, filter_width)
    result = image_column.T @ reshaped_filter
    output_maps = result.reshape(output_height, output_width, num_filters)
    return output_maps


def relu_activation(feature_map):
    """
    A function to apply ReLU to feature map

    Args:
        feature_map: Input feature map

    Returns:
        Feature map with ReLU applied (all negative values set to 0)
    """
    return np.maximum(0, feature_map)


def max_pooling(feature_map, pool_size=(2, 2)):
    """
    Apply max pooling operation to feature map.

    Args:
        feature_map: Input feature map
        pool_size: Tuple specifying pooling window size (height, width)

    Returns:
        Downsampled feature map
    """
    pool_h, pool_w = pool_size
    input_h, input_w = np.shape(feature_map)
    output_h = input_h // pool_h
    output_w = input_w // pool_w

    pooled_map = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            h_start = i * pool_h
            h_end = h_start + pool_h
            w_start = j * pool_w
            w_end = w_start + pool_w

            pool_region = feature_map[h_start:h_end, w_start:w_end]
            pooled_map[i, j] = np.max(pool_region)

    return pooled_map


def normalise_feature_map(feature_map):
    """
    A funtion to normalise feature map to have zero mean and unit standard
    deviation

    Args:
        feature_map: Input feature map

    Returns:
        Normalised feature map
    """
    mean = np.mean(feature_map)
    std = np.std(feature_map)
    normalised_map = (feature_map - mean) / std
    return normalised_map


def fully_connected_layer(feature_maps, weights):
    """
    A function to apply a fully connected layer operation

    Args:
        feature_maps: Input feature maps
        weights: the weight matrix with shape (input_size, output_size)

    Returns:
        Output activations
    """
    flattened = feature_maps.reshape(-1)
    output = flattened @ weights
    return output


def softmax(activations):
    """
    A function to apply softmax function to convert activations to
    probabilities

    Args:
        activations: input activation values

    Returns:
        The prrobability distribution over classes
    """
    exp_calc = np.exp(activations - np.max(activations))
    probabilities = exp_calc / np.sum(exp_calc)
    return probabilities


# this is the main which runs every function we implemented above in order
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    example_image = x_train[0]
    print(example_image)
    print(example_image.shape)

    image = x_train[0]
    image = image.astype('float32') / 255
    image = image.reshape(image.shape[0], image.shape[1], 1)
    print(image.shape)

    horizontal_filter = np.array([
        [1,  1,  1],
        [0,  0,  0],
        [-1, -1, -1]
    ])

    vertical_filter = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    filters = np.array([horizontal_filter, vertical_filter]).reshape(2, 3, 3,
                                                                     1)
    print(filters.shape)

    feature_maps = convolve(image, filters)
    print(feature_maps.shape)

    horizontal_map = feature_maps[:, :, 0]
    vertical_map = feature_maps[:, :, 1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.reshape(28, 28), cmap='gray')
    axes[0].set_title('original image')
    axes[1].imshow(horizontal_map, cmap='gray')
    axes[1].set_title('horizontal edge')
    axes[2].imshow(vertical_map, cmap='gray')
    axes[2].set_title('vertical edge')
    plt.show()

    apply_relu = relu_activation(vertical_map)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(vertical_map, cmap='gray')
    axes[0].set_title('no ReLU')
    axes[1].imshow(apply_relu, cmap='gray')
    axes[1].set_title('with ReLU')
    plt.show()

    pooled_result = max_pooling(apply_relu, pool_size=(2, 2))
    plt.imshow(pooled_result, cmap='gray')
    plt.title('After 2x2 Max Pooling')
    plt.show()

    normalised = normalise_feature_map(pooled_result)
    print(f"mean: {np.mean(normalised):.6f}")
    print(f"std: {np.std(normalised):.6f}")

    input_size = pooled_result.size
    num_output_units = 10
    weights = np.random.randn(input_size, num_output_units) * 0.01

    fc_output = fully_connected_layer(pooled_result, weights)
    print(f"shape: {fc_output.shape}")

    probabilities = softmax(fc_output)
    print(f"sum: {np.sum(probabilities)}")
    print(f"class: {np.argmax(probabilities)}")
    print(f"probabilites: {probabilities}")
