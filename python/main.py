import numpy as np 
import Layers

X1=np.array(([0,0,1,0.2],[0,1,1,0.8],[1,0,1,0.21],[1,1,1,0.292],[0.5,1,1,0.933]), dtype=float)
y1=np.array(([0],[1],[1],[0],[1]), dtype=float)
X2 = []
with open('X.txt', 'r') as inputs:
    for line in inputs.readlines():
        X2.append(line.split(','))
    inputs.close()
with open('Y.txt', 'r') as expected:
    y2raw = expected.read()
    expected.close()
for i in range(len(X2)):
    X2[i].pop()
y2raw = y2raw.replace(' ', '')
y2raw = y2raw.split(',')
y2raw.pop()
y2 = []
for i in range(len(y2raw)):
    y2.append([y2raw[i]])
y2 = np.array(y2, dtype=float)
X2 = np.array(X2, dtype=float)

def train(X, y, neurons, iterations):
    NN = Layers.NeuralNetwork(X,y, neurons)
    for i in range(iterations):
        if i % 100 ==0: 
            print ("for iteration # " + str(i) + "\n")
            print ("Actual Output: \n" + str(y))
            print ("Predicted Output: \n" + str(NN.forward()))
            print ("Loss: \n" + str(np.mean(np.square(y - NN.forward())))) # mean sum squared loss
            print ("\n")
        NN.train()
    return NN

NN = train(X2, y2, neurons=10, iterations=1000)
