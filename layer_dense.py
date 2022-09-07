import numpy as np
import testdata
import matplotlib.pyplot as plt

np.random.seed(0)
learningRate = 1000
numberOfLayers = 2
neuronsPerLayer = [2, 2]
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
        self.inputs = inputs
        # self.output(torch.relu(inputs))
    def derivative(self):
        return np.where(self.output > 0, 1, 0)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # self.output(torch.softmax(inputs))
        self.inputs = inputs
    def derivative(self):
        return [\
            [\
                [\
                    self.output[j]*((1 if j == k else 0) - self.output[k]) for k in range(len(self.output[i]))\
                ] for j in range(len(self.output[i]))\
            ] for i in range(len(self.output))\
        ]

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        self.sample_losses = sample_losses
        self.data_loss = np.mean(sample_losses) #fleep flop fleebodobo bop fleebo deebo flop bo boop
        return self.data_loss
        
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_predicted, y_known):
        samples = len(y_predicted)
        y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_known]
        negative_log_likelihoods = -np.log(correct_confidences)
        self.predicted = y_predicted
        self.known = y_known
        return negative_log_likelihoods
    def derivative(self):
        return self.predicted - [[1 if self.known[i] == j else 0 for j in range(len(self.predicted[i]))] for i in range(len(self.predicted))]
        # return -1/self.sample_losses

def gradientDescent(listOfLayers, listOfActivationFunctions, lossCalculator):
    dlda_dadz = listOfActivationFunctions[len(listOfActivationFunctions)-1].derivative() * lossCalculator.derivative()
    for i in reversed(range(len(listOfLayers))):
        weightDeriv, biasDeriv = listOfLayers[i].derivative()
        listOfLayers[i].backward(weightDeriv * dlda_dadz, biasDeriv * dlda_dadz)
        dlda_dadz *= listOfActivationFunctions[i-1].derivative() * listOfLayers[i].weights     
    
def train(listOfLayers, activationLayers, lossCalculator, points, classes):
    predictions = []
    for point in points:
        previousResults = point
        for i in range(len(listOfLayers)):
            listOfLayers[i].forward(previousResults)
            activationLayers[i].forward(listOfLayers[i].output)
            previousResults = activationLayers[i].output
        predictions.append(previousResults)
    loss = lossCalculator.calculate(predictions, classes)
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

# for i in range(10):
#     loss = train(listOfLayers, activationlayers, loss_calculator, points, classes)
#     accuracy = score(activationlayers[len(activationlayers) -1].output, classes) 
#     print(f'Loss: {loss} Iteration: {i} Percentage Correct: {accuracy}')

def breb_applyGrads(learningRate, wGrad, bGrad, layers):
    for i in range(len(layers)):
        for j in range(len(layers[i].weights)):
            for k in range(len(layers[i].weights[j])):
                layers[i].weights[j][k] += wGrad[i][j][k] / learningRate
        for j in range(len(layers[i].biases)):
            layers[i].biases[j] += bGrad[i][j] / learningRate

def breb_processInput(points, layers, activations):
    for i in range(len(layers)):
        layers[i].forward(points)
        activations[i].forward(layers[i].output)
        points = activations[i].output

def breb_updateGrads(points, wGrad, bGrad, layers, activations, loss_calculator, classes):
    breb_processInput(points, layers, activations)
    loss_calculator.calculate(activations[-1].output, classes)
    outputDerivatives = loss_calculator.derivative()
    finalActivationDerivatives = activations[-1].derivative()[0][0]
    neuronValues = [[0 for _ in range(len(layers[i].output[0]))] for i in range(len(layers))]
    for i in range(numberOfOutputs):
        neuronValues[-1][i] = outputDerivatives[i] * finalActivationDerivatives[i]
    for i in reversed(range(len(layers))):
        for j in range(len(layers[i].weights)):
            for k in range(len(layers[i].weights[j])):
                wGrad[i][j] = layers[i].inputs[0][j] * neuronValues[i][k]
        for j in range(len(layers[i].biases[0])):
            bGrad[i] = neuronValues[j]
        if i == 0:
            continue
        for j in range(len(layers[i].weights)):
            neuronValues[i-1][j] = 0
            for k in range(len(layers[i].weights[j])):
                neuronValues[i-1][j] += layers[i].weights[j][k] * neuronValues[i][k]

def breb_train(training_points, layers, activations, loss_calculator, classes):
    wGrad = [[[0 for _ in range(len(layers[i].weights[j]))] for j in range(len(layers[i].weights))] for i in range(len(layers))]
    bGrad = [[0 for _ in range(len(layers[i].biases[0]))] for i in range(len(layers))]
    loss = 0
    
    breb_updateGrads(training_points, wGrad, bGrad, layers, activations, loss_calculator, classes)
    loss += loss_calculator.data_loss
    print(loss)
    breb_applyGrads(learningRate / len(training_points), wGrad, bGrad, layers)

for i in range(10):
    breb_train(points[:1], listOfLayers, activationlayers, loss_calculator, classes[:1])