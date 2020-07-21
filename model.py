from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf

def get_experts(n_experts, input_shape = (28*28,)):
    return [keras.Sequential([
        Dense(32, activation = 'relu', input_shape = input_shape),
        Dense(10, activation = 'softmax')]) 

        for i in range(n_experts)]

def get_gate(n_experts, input_shape = (28*28,)):
    return keras.Sequential([
        Dense(32, activation = 'relu', input_shape = input_shape),
        Dense(n_experts, activation = 'softmax')])


class MOE_Model (keras.Model):
    def __init__(self, experts, gate, optimizer = None):
        super(MOE_Model, self).__init__()
        self.experts = experts
        self.gate = gate
        self.optimizer = optimizer

        if self.optimizer == None:
            self.optimizer = keras.optimizers.Adam(0.001)
    
    def call(self, inputs):
        gates = tf.expand_dims(self.gate(inputs), -1)
        values = tf.stack([expert(inputs) for expert in self.experts], axis = -1)
        return tf.matmul(values, gates)

    # The following functions are actually never called in the implemented 
    # training loop (that uses the keras fit function with categorical crossentropy)
    # However, if a custom training loop was made, simply calling the step function on each batch would work
    
    # Notice that the loss function is not defined in terms of model.call, but in terms of the predicted probabilities
    # While both aproaches are equivalent for classification, they are 
    # actually different for regression, treating MoE as a mixture of gaussians
    # If this code were to be adapted to perform regression, all that would need to be done is to implement the correct 
    # probabilities function and then call the step function in a custom training loop

    def probabilities(self, inputs, outputs):
        values = self.call(inputs)
        return tf.reduce_sum(tf.multiply(tf.expand_dims(outputs, -1),values), axis = 1)

    def calculate_loss(self, inputs, true_outputs):
        probs = self.probabilities(inputs, true_outputs)
        return -tf.math.log(probs)
        
    def grad(self, inputs, true_outputs):
        with tf.GradientTape() as tape:
            l = self.calculate_loss(inputs, true_outputs)
        gradients = tape.gradient(l, self.trainable_variables)
        return gradients, l

    def step(self, inputs, true_outputs):
        grads, l = self.grad(inputs, true_outputs)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return l

