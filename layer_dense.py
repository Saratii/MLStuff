import numpy as np
import testdata
import matplotlib.pyplot as plt

np.random.seed(0)
learningRate = 1000
numberOfLayers = 2
neuronsPerLayer = [1, 1]
numberOfInputs = 2
numberOfOutputs = 3
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #inputs per batch
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    def backward(self, weightGradient, biasGradient, learningRate):
        self.weights += weightGradient / learningRate 
        self.biases += biasGradient / learningRate
    def derivative(self):
        return self.inputs, np.ones_like(self.biases)

class Activation_ReLU:
    def forward(self, inputs): 
        self.output = np.maximum(0, inputs)
        # self.output(torch.relu(inputs))
    def derivative(self):
        return np.where(self.output > 0, 1, 0)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # self.output(torch.softmax(inputs))
    def derivative(self):
        return [[self.output[i]*((1 if i == j else 0) - self.output[j]) for i in range(len(self.output))] for j in range(len(self.output))]

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        self.sample_losses = sample_losses
        data_loss = np.mean(sample_losses)
        return data_loss
        
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_predicted, y_known):
        samples = len(y_predicted)
        y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_known]
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def derivative(self):
        return -1/self.sample_losses

def gradientDescent(listOfLayers, listOfActivationFunctions, lossCalculator):
    dlda_dadz = listOfActivationFunctions[len(listOfActivationFunctions)-1].derivative() * lossCalculator.derivative()
    for i in reversed(range(len(listOfLayers))):
        weightDeriv, biasDeriv = listOfLayers[i].derivative()
        listOfLayers[i].backward(weightDeriv * dlda_dadz, biasDeriv * dlda_dadz)
        dlda_dadz *= listOfActivationFunctions[i-1].derivative() * listOfLayers[i].weights     
    
def train(listOfLayers, activationLayers, lossCalculator, points, classes):
    for point in points:
        previousResults = point
        for i in range(len(listOfLayers)):
            listOfLayers[i].forward(previousResults)
            activationLayers[i].forward(listOfLayers[i].output)
            previousResults = activationLayers[i].output
    loss = lossCalculator.calculate(previousResults, classes)
    gradientDescent(listOfLayers, activationLayers, lossCalculator)
    return loss

def score(predicted, actual):
    correct = 0
    for i in range(len(predicted)):
        wrong = False
        for j in range(len(predicted[i])):
            if predicted[i][actual[i]] < predicted[i][j]:
                wrong = True
        if not wrong:
            correct+=1
    return correct/len(actual)

listOfLayers = []
activationlayers = []
loss_calculator = Loss_CategoricalCrossentropy()
for i in range(numberOfLayers + 1):
    listOfLayers.append(Layer_Dense(neuronsPerLayer[i-1] if i>0 else numberOfInputs, neuronsPerLayer[i] if i < numberOfLayers else numberOfOutputs))
    activationlayers.append(Activation_ReLU() if i<numberOfLayers else Activation_Softmax())

points, classes = testdata.create_data(100, 3)

for i in range(10):
    loss = train(listOfLayers, activationlayers, loss_calculator, points, classes)
    accuracy = score(activationlayers[len(activationlayers) -1].output, classes) 
    print(f'Loss: {loss} Iteration: {i} Percentage Correct: {accuracy}')


2

