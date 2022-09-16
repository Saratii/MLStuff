import numpy as np 
import Layers
# Each row is a training example, each column is a feature  [X1, X2, X3]
X1=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y1=np.array(([0],[1],[1],[0]), dtype=float)

# Define useful functions    

def encode(thing):
    if thing == '+':
        return 2.2938e-7
    elif thing == '-':
        return 1.9202e-9
    else: 
        return int(thing)

math = open("CSV_Files/math.txt", "r")
math = math.readlines()
numLines = len(math)
mathDataInput = []
mathDataExpected = []

for i in range(numLines):
    line = math[i].replace('\n','')
    line = line.replace('+', ',+,')
    line = line.replace('-', ',-,')
    line = line.replace('=', ',')
    line = line.replace(' ', '')
    line = line.split(',')
    mathDataInput.append(line)
for i in range(len(mathDataInput)):
    for j in range(len(mathDataInput[0])):
        mathDataInput[i][j] = encode(mathDataInput[i][j])
    mathDataExpected.append([mathDataInput[i].pop(0)])
X2 = np.array(mathDataInput, dtype=float)
y2 = np.array(mathDataExpected, dtype=float)

def train(X, y, neurons, iterations):
    NN = Layers.NeuralNetwork(X,y, neurons)
    for i in range(iterations):
        if i % 100 ==0: 
            print ("for iteration # " + str(i) + "\n")
            print ("Input : \n" + str(X))
            print ("Actual Output: \n" + str(y))
            print ("Predicted Output: \n" + str(NN.forward()))
            print ("Loss: \n" + str(np.mean(np.square(y - NN.forward())))) # mean sum squared loss
            print ("\n")
    
        NN.train(X, y)
    return NN

NN = train(X1, y1, neurons=10, iterations=1000)
