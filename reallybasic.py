from tkinter import W
import numpy as np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
def squaredDifference(yPredicted, yActual):
    return (yPredicted - yActual)**2

X = np.array(3)
y = np.array(1)
W1 = 0.1
B1 = 0.4
W2 = np.array([0.1, 0.2])
B2 = np.array([0.1, 0.2])
learningRate = 0.01

def forward():
    layer1 = X * W1 + B1
    activation1 = max(0, layer1) #ReLU
    layer2 = activation1 * W2 + B2
    activation2 = softmax(layer2)
    loss = squaredDifference(activation2, y)
    loss = np.sum(loss) / len(loss)
    return layer1, activation1, layer2, activation2, loss

def backward(activation2, activation1, layer1):
    global W1, W2, B1, B2
    dlda2 = 2 * (activation2 - y)  #dl/da2 (a2 - y)^2 = 2(a2 - y1) * d/dy (a2 - y1) = 1 --> 2(a2 - y) * 1 
    da2dl2 = [[activation2[0]*(1-activation2[0]), -activation2[0] * activation2[1]], [-activation2[1] * activation2[0], activation2[1] * (1 - activation2[1])]]
    dl2dw2 = activation1
    dldw2 = dlda2 * da2dl2 * dl2dw2
    dldb2 = dlda2 * da2dl2
    dl2da1 = W2
    da1dl1 = 0 if layer1 < 0 else 1
    dl1dw1 = X
    dldw1 = dlda2 * da2dl2 * dl2da1 * da1dl1 * dl1dw1
    dldb1 = dlda2 * da2dl2 * dl2da1 * da1dl1
    W2 -= dldw2 * learningRate
    B2 -= dldb2 * learningRate
    W1 -= dldw1 * learningRate
    B1 -= dldb1 * learningRate
layer1, activation1, layer2, activation2, loss = forward()
print(loss)
backward(activation2, activation1, layer1)
layer1, activation1, layer2, activation2, loss = forward()
print(loss)
print("Predicted:", activation2)
print("Loss:", loss)
