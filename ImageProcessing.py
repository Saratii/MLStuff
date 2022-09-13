from PIL import Image, ImageOps
import os
Pixels = 600
processedImages = []
folder_dir = "catImages"
# folder_dir = "ElephantImages"
label = 1

def processImage(file):
    image = Image.open(file)
    width, height = image.size
    centerX = width/2
    centerY = height/2
    if width>height:
        image = image.crop((centerX - centerY, 0, centerX + centerY, height))
    elif height>width:
        image = image.crop((centerX - centerY, 0, width, width))
    image = image.resize((Pixels, Pixels))
    image = ImageOps.grayscale(image)
    return image

def createCSV():
    labels = ['label,']
    for i in range(1, Pixels+1):
        for j in range(1, Pixels+1):
            labels.append(f'{i}x{j},')
    labels[Pixels*Pixels] = f'{Pixels}x{Pixels}'
    with open('CSV_Files/AnimalData.csv', 'w') as f: 
        row = ""
        for i in range(len(labels)):
            row = row + labels[i]
        f.write(row)
        f.write('eefafliafliaf')
        f.close()

def appendCSV(image, label):
    pixels = []
    for i in range(Pixels):
        for j in range(Pixels):
            pixels.append(f'{image.getpixel((i, j))},')
    row = ""
    pixels[Pixels**2-1] = image.getpixel((Pixels-1, Pixels-1))
    for pixel in pixels:
        row = row + str(pixel)
    with open('CSV_Files/AnimalData.csv', 'a') as f:
        f.write(f'\n{label},{row}')
        f.close()

# createCSV()

for image in os.listdir(folder_dir):
    dir = str(folder_dir)+'/'+str(image)
    processedImages.append(processImage(dir))
# for image in processedImages:
#     image.show()

for i in range(len(processedImages)):
    appendCSV(processedImages[i], label)



#Elephant = 0
#Cat = 1