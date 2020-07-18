from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_mnist(batch_size = 32, shuffle_size = 100):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuffle_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(shuffle_size).batch(batch_size)
    return train_dataset, test_dataset

def load_cifar(batch_size = 32, shuffle_size = 100):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train / 255
    x_test = x_test / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    x_train = x_train.reshape(-1, 32*32*3)
    x_test = x_test.reshape(-1, 32*32*3)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuffle_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(shuffle_size).batch(batch_size)
    return train_dataset, test_dataset
