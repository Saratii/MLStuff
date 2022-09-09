import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
data = pd.read_csv('mnist_train.csv')
testdata = pd.read_csv('mnist_test.csv')
customData = pd.read_csv('customDrawing.csv')

inputCount = 784 #28 x 28
layer1NeuronCount = 10
outputCount = 10
numIterations = 1000
retrain = True

def processData(data):
    data = np.array(data)
    np.random.shuffle(data)
    m, n = data.shape
    data_dev = data[0:].T
    y_dev = data_dev[0]
    x_dev = data_dev[1:n]
    x_dev = x_dev / 255
    data_train = data[0:m].T
    y_train = data_train[0]
    x_train = data_train[1:n]
    x_train = x_train / 255
    return m, x_train, y_train, data_train, x_dev, y_dev

m, x_train, y_train, data_train, x_dev, y_dev = processData(data)
testm, testx_train, testy_train, testdata_train, testx_dev, testy_dev = processData(testdata)
cm, cx_train, cy_train, cdata_train, cx_dev, cy_dev = processData(customData)

#_, m_train = x_train.shape


def initParams():
    w1 = np.random.rand(layer1NeuronCount, inputCount) - 0.5
    b1 = np.random.rand(layer1NeuronCount, 1) - 0.5
    w2 = np.random.rand(outputCount, layer1NeuronCount) - 0.5
    b2 = np.random.rand(outputCount, 1) - 0.5
    return w1, b1, w2, b2
def relu(z):
    return np.maximum(0, z)
def dRelu(z):
    return z>0
def softMax(z):
    return np.exp(z)/sum(np.exp(z))
def forwardProp(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softMax(z2)
    return z1, a1, z2, a2
def oneHot(y):
    oneHotY = np.zeros((y.size, y.max()+1))
    oneHotY[np.arange(y.size), y] = 1
    oneHotY = oneHotY.T
    return oneHotY
def backProp(z1, a1, z2, a2, w1, w2, x, y):
    oneHotY = oneHot(y)
    dz2 = a2 - oneHotY
    dw2 = 1/m*dz2.dot(a1.T)
    db2 = 1/m*np.sum(dz2)
    dz1 = w2.T.dot(dz2) * dRelu(z1)
    dw1 = 1/m*dz1.dot(x.T)
    db1 = 1/m*np.sum(dz1)
    return dw1, db1, dw2, db2
def updateParams(dw1, db1, dw2, db2, w1, b1, w2, b2, learningRate):
    w1 -= learningRate * dw1
    w2 -= learningRate * dw2
    b1 -= learningRate * db1
    b2 -= learningRate * db2
    return w1, b1, w2, b2
def getPredictions(a2):
    return np.argmax(a2, 0)
def getAccuracy(predictions, y):
    return np.sum(predictions == y) / y.size
def gradientDescent(x, y, iterations, learningRate):
    w1, b1, w2, b2 = initParams()
    previousAcc = 0
    for i in range(iterations):
        z1, a1, z2, a2 = forwardProp(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backProp(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = updateParams(dw1, db1, dw2, db2, w1, b1, w2, b2, learningRate)
        if i % 10 == 0:
            acc = getAccuracy(getPredictions(a2), y)
            if acc > previousAcc:
                print(f'Iteration: {i} Accuracy: {acc}')
            elif acc <= previousAcc:
                print(f'Iteration: {i} Accuracy: {acc}')
            previousAcc = acc

    return w1, b1, w2, b2
def makePredictions(x, w1, b1, w2, b2):
    _, _, _, a2 = forwardProp(w1, b1, w2, b2, x)
    return getPredictions(a2)
def testPredictions(index, w1, b1, w2, b2, test=False, custom=False):
    if test:
        currentImage = testx_train[:, index, None]
        label = testy_train[index]
    elif custom:
        currentImage = cx_train[:, index, None]
        label = cy_train[index]
    else: 
        currentImage = x_train[:, index, None]
        label = y_train[index]
    prediction = makePredictions(currentImage, w1, b1, w2, b2)
    
    print(f'Prediction: {prediction} Label: {label}')
    currentImage = currentImage.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(currentImage, interpolation='nearest')
    plt.show()
def readWB():
    f = open("weights&biases.txt", "w")
    w1 = f.readline()
    b1 = f.readline()
    w2 = f.readline()
    b2 = f.readline()
    return w1, b1, w2, b2
def writeWB(w1, b1, w2, b2):
    f = open("weights&biases.txt", "w")
    f.write(str(w1))
    f.write(str(b1))
    f.write(str(w2))
    f.write(str(b2))
    f.close()
if retrain:
    w1, b1, w2, b2 = gradientDescent(x_train, y_train, numIterations, 0.1)
    writeWB(w1, b1, w2, b2)
else: 
    w1, b1, w2, b2 = readWB()

print(f'Training Accuracy {getAccuracy(makePredictions(x_dev, w1, b1, w2, b2), y_dev)}')

print(f'Testing Accuracy {getAccuracy(makePredictions(testx_dev, w1, b1, w2, b2), testy_dev)}\n')

# for i in range(numTestingPoints):
#     testPredictions(i, w1, b1, w2, b2, test=True)
testPredictions(0, w1, b1, w2, b2, custom=True)