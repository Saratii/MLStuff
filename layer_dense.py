import numpy as np
import testdata
import matplotlib.pyplot as plt
np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #inputs per batch
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, partialDerivativeOfLossFunctionWithRespectToEachBias, partialDerivativeOfLossFunctionWithRespectToEachWeights):
        self.biases += 0.001 * partialDerivativeOfLossFunctionWithRespectToEachBias
        self.weights += 0.001 * partialDerivativeOfLossFunctionWithRespectToEachWeights

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

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

points, classes = testdata.create_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_calculator = Loss_CategoricalCrossentropy()


lowest_loss = 999999999
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()
    
iterations = []
losses = []    
for i in range(1000000):
    dense1.weights = 0.05 * np.random.rand(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.rand(1, 3)
     
    dense1.forward(points)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_calculator.calculate(activation2.output, classes)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == classes)
    
    if loss < lowest_loss:
        print('New set of weights found, iteration:', i, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        iterations.append(i)
        losses.append(loss)
        lowest_loss = loss
print("done")
plt.plot(iterations, losses)