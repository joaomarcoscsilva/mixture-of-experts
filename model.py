from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf

def get_experts(n_experts, input_shape = (28*28,)):
    return [keras.Sequential([
        Dense(32, activation = 'relu', input_shape = input_shape),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax')]) 

        for i in range(n_experts)]

def get_gate(n_experts, input_shape = (28*28,)):
    return keras.Sequential([
        Dense(32, activation = 'relu', input_shape = input_shape),
        Dense(64, activation = 'relu'),
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
    
    def probabilities(self, inputs, outputs):
        values = self.call(inputs)
        return tf.reduce_sum(tf.multiply(tf.expand_dims(outputs, -1),values), axis = 1)


    def loss(self, inputs, true_outputs):
        probs = self.probabilities(inputs, true_outputs)
        return -tf.math.log(probs)
        
    def grad(self, inputs, true_outputs):
        with tf.GradientTape() as tape:
            l = self.loss(inputs, true_outputs)
        gradients = tape.gradient(l, self.trainable_variables)
        return gradients, l

    def step(self, inputs, true_outputs):
        grads, l = self.grad(inputs, true_outputs)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return l

