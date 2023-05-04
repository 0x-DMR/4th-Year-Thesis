# Greyscale.py is used to convert the classes.dex file to a grey scaled image.
# This image will then be used to classify the malware through the use of a Convolutional Neural Network. 

import os                                                                       # Provides the ability to interact with the operating system
import numpy as np                                                              # Provides the ability to use numerical data to convert the .dex files to greyscale images
import cv2                                                                      # Provides the ability to commit the newly generated image to storage
from math import ceil, sqrt                                                     # Provides the ability to round up the nearest integer (ceil) and return the squared root of a number (sqrt)

locationOfData = '/home/dmr/AdwareDexFiles/feiwo'                               # Declare the location of the classes.dex files

for filename in os.listdir(locationOfData):                                     # Loop through all files in given directory 
    if filename.endswith('.dex'):                                               # Check if the file is a valid filetype, which is .dex
        
        with open(os.path.join(locationOfData, filename), 'rb') as dfile:       # Read the whole file to data
            data = dfile.read()
        
        dataLength = len(data)                                                  # Convert the Data length to bytes

        d = np.frombuffer(data, dtype=np.uint8)                                 # Initialise d which is a vector of the bytes of dataLength

        sqrtLength = int(ceil(sqrt(dataLength)))                                # Generate the image to be similar to a square so compute square root and round up

        requiredLength = sqrtLength*sqrtLength                                  # Required length in bytes.

        paddingLength = requiredLength - dataLength                             # Number of bytes to need to be padded 

        paddedImage = np.hstack((d, np.zeros(paddingLength, np.uint8)))         # Pad the image and use 0s to do so at the end of the image

        imageToSave = np.reshape(paddedImage, (sqrtLength, sqrtLength))         # Reshape 1D array to a 2D array by multipling x sqrtLength by itself, this is the image which is created

        outputFile = os.path.splitext(filename)[0] + '.png'                     # Save the image that was created with the same name as a png file
        cv2.imwrite(os.path.join(locationOfData, outputFile), imageToSave)      # Save the image to the location mentioned previously