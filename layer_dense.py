import numpy as np
import testdata
import matplotlib.pyplot as plt
from sympy import *
np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #inputs per batch
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        x = Symbol('x')
        y = -np.log(correct_confidences)
        self.weights *= y.diff(x, self.weights)
        self.biases *= y.diff(x, self.biases)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        
        correct_confidences = y_pred_clipped[range(samples), y_true]
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
def optimizerRandom(dense1, dense2):
    dense1.weights = 0.05 * np.random.rand(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.rand(1, 3)

points, classes = testdata.create_data(100, 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_calculator = Loss_CategoricalCrossentropy()

     
dense1.forward(points)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_calculator.calculate(activation2.output, classes)
dense2.backward(activation2.output, classes)
dense1.backward(activation2.output, classes)


# predictions = np.argmax(activation2.output, axis=1)
# accuracy = np.mean(predictions == classes)
print(loss)
