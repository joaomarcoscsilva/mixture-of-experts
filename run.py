import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import argparse
import data, model, train
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--n-experts', type = int, metavar = 'N', 
        help = 'Number of experts')
parser.add_argument('-d', '--dataset', type = str, metavar = 'D', default = 'MNIST',
        help = 'Dataset to be used. Must be either MNIST or CIFAR10.')
parser.add_argument('-e', '--epochs', type = int, metavar = 'E', default = 10,
        help = 'Number of epochs to train for.')
parser.add_argument('-w', '--use-wandb', action = 'store_true',
        help = 'If set, log metrics to Weights and Biases')

cmd = parser.parse_args()

if cmd.dataset == 'MNIST':
    input_shape = (28*28,)
    train_data, test_data = data.load_mnist()
else:
    input_shape = (32*32*3,)
    train_data, test_data = data.load_cifar()

if cmd.use_wandb:
    config = {'Experts' : cmd.n_experts, 'Dataset' : cmd.dataset, 'Epochs' : cmd.epochs}
else:
    config = None

m = model.MOE_Model(model.get_experts(cmd.n_experts, input_shape), model.get_gate(cmd.n_experts, input_shape))
m.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])
train.train_model(m, train_data, cmd.epochs, test_data, config)  
