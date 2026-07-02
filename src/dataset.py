import tensorflow as tf
def load_cifar10():
    """
    Load the CIFAR-10 dataset.

    Returns:
        Tuple: A tuple containing the training and testing datasets.
    """
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return (X_train, Y_train), (X_test, Y_test)
