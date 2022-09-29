from PIL import Image
import os
def processIm(im):
    width, height = im.size
    if width>height:
        im.crop((width - height, 0, height, height))
    elif height>width:
        im.crop(0, height - width, width, width)
    im = im.resize((10, 10))
    return im
def reset():
    with open('X.txt', 'w') as inputs:
        inputs.close()
    with open('Y.txt', 'w') as expected:
        expected.close()
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
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/Cats'):
    im = Image.open('Cats/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    inputArray += pixelString + "\n"
    outputArray += "0,"
for file in os.listdir('/Users/propleschmaren/Desktop/MLStuff/Dogs'):
    im = Image.open('Dogs/'+file)
    processedIm = processIm(im)
    pixelString = makePixelString(processedIm)
    inputArray += pixelString + "\n"
    outputArray += "1,"
with open('X.txt', 'a') as inputs:
        inputs.write(inputArray)
        inputs.close()
with open('Y.txt', 'a') as expected:
    expected.write(str(outputArray))
    expected.close()