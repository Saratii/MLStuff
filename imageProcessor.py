from PIL import Image
import os
def processIm(im):
    width, height = im.size
    if width>height:
        im.crop((width - height, 0, height, height))
    elif height>width:
        im.crop(0, height - width, width, width)
    im = im.resize((20, 20))
    return im
def reset():
    with open('X.txt', 'w') as inputs:
        inputs.close()
    with open('Y.txt', 'w') as expected:
        expected.close()
    with open('testY.txt', 'w') as testInputs:
        testInputs.close()
    with open('testX.txt', 'w') as testExpected:
        testExpected.close()
def makePixelString(im):
    pixelString = ""
    x, y = im.size
    for i in range(x):
        for j in range(y):
            pixel = im.getpixel((i, j))
            grey = round(sum(pixel)//3/255, 1)
            pixelString += str(grey) + ","
    return pixelString

reset()
inputArray = ""
outputArray = ""
testInputArray = ""
testOutputArray = ""
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/Cats'):
    im = Image.open('Cats/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    inputArray += pixelString + "\n"
    outputArray += "0,1\n"
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/Dogs'):
    im = Image.open('Dogs/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    inputArray += pixelString + "\n"
    outputArray += "1,0\n"
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/DogsTest'):
    im = Image.open('DogsTest/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    testInputArray += pixelString + "\n"
    testOutputArray += "1,0\n"
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/CatsTest'):
    im = Image.open('CatsTest/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    testInputArray += pixelString + "\n"
    testOutputArray += "0,1\n"
with open('X.txt', 'a') as inputs:
        inputs.write(inputArray)
        inputs.close()
with open('Y.txt', 'a') as expected:
    expected.write(str(outputArray))
    expected.close()
with open('testX.txt', 'a') as testInputs:
        testInputs.write(testInputArray)
        testInputs.close()
with open('testY.txt', 'a') as testOutputs:
    testOutputs.write(str(testOutputArray))
    testOutputs.close()