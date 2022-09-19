import numpy as np

# Derivative of sigmoid

def inverseSigmoid(t):
    return -np.log(1/t-1)
# Class definition
class NeuralNetwork:
    def __init__(self, x, y, neurons):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1], neurons)
        self.weights2 = np.random.rand(neurons,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def forward(self):
        self.layer1 = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid.calc(self.layer1)
        self.layer2 = np.dot(self.layer1, self.weights2)
        self.layer2 = sigmoid.calc(self.layer2)
        return self.layer2
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid.derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid.derivative(self.output), self.weights2.T)*sigmoid.derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    
    def train(self):
        self.output = self.forward()
        self.backprop()
    
class sigmoid:
    def calc(t):
        return 1/(1+np.exp(-t))
    def derivative(p):
        return p * (1 - p)

