# The following section is used to generate information about the datasets so these 
# variables can be initialised in the CNN Model. The information gathered here will
# also be used to determine an appropiate size of the training and testing datasets.
import os                                                   # Import os to change the information surrounding the enviroment 
from keras.preprocessing.image import ImageDataGenerator    # Create tensorflow image batches using data augmentation.
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
locationOfDataset = '/home/dmr/DatasetForExperiments/scareware/'
batches = ImageDataGenerator().flow_from_directory(directory=locationOfDataset, 
                                                    target_size=(64,64), 
                                                    batch_size=100)

# Print information about Images 
imgs, labels = next(batches)
print("Number of Images and Size of Images: ",imgs.shape) # Images with width x length x depth
print("Number of Images and Families: ",labels.shape) # Lablels with batch_size, number of classes

# This section will cover the necessary steps used to split the scareware dataset
# into training and testing to evaluate the effectiveness of this convolutional neural 
# network model based on the scareware samples.
import numpy as np                                   # Provides the ability to use numerical data to convert the .dex files to greyscale images
import scipy as sp                                   # Provides the use of algorithms to be used 
from PIL import Image                                # Allow for processing of images

# Splitting the Dataset into Training and testing with a test size of 30% of overall 
# samples in the dataset. 
from sklearn.model_selection import train_test_split    # Used to split the dataset

# Split up the data set, 30% testing 70% training
trainX, testX, trainY, testY = train_test_split(imgs/255.,labels, test_size=0.3)

# Print the information regarding the training and testing sample sizes
#print("Size of Training Set: ", trainX.shape)                   # Print training set information
#print("Size of Testing Set: ",testX.shape)                      # Print testing set information
print("Number of Samples in Training Set: ",trainY.shape)   # Print training set information
print("Number of Samples in Testing Set: ",testY.shape)     # Print testing set information

# The Convolutional Neural Network Model
# This section will encompass the aspects surrounding the convultional neural network
# which was used for the experimentation of this research. 
import keras                                            # Import keras
import tensorflow                                       # Import tensorflow
from keras.models import Sequential, Input, Model       # Used for CNN model regarding Models
from keras.layers import Dense, Dropout, Flatten        # Used for CNN model regarding Layers
from keras.layers import Conv2D, MaxPooling2D           # Used for CNN model regarding Layers
from tensorflow.keras.layers import BatchNormalization  # Layer used to normalise inputs
from tensorflow.keras.metrics import AUC                # Used estimate to Area Under Curve

noOfFamilies = 11 # Number of Families in the Sample

# Function used to create the layers of the convolutional neural network model. These parameters were
# constantly changed throughout testing in order to obtain the best possible results. Within this model 
# there are 10 main layers and each layers parameters are explained in detail. 
def CNNModel():
    CNNModel = Sequential()        # Sequential Model is used the experiments

    # Convolutional Layer: With relu Activation, 30 filters, and a kernel size of 3 x 3
    CNNModel.add(Conv2D(30, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(64,64,3)))                # Layer 1 

    # Max Pooling Layer: With a pool size of 2 x 2
    CNNModel.add(MaxPooling2D(pool_size=(2, 2)))            # Layer 2
    
    # Convolutional Layer: 18 filters and kernel size of 3 x 3 
    CNNModel.add(Conv2D(18, (3, 3), activation='relu'))     # Layer 3
    
    # Max Pooling Layer: Pool size of 2 x 2 
    CNNModel.add(MaxPooling2D(pool_size=(2, 2)))            # Layer 4
    
    # Drop Out Layer: This layer used to drop 25% of neurons in the model .
    CNNModel.add(Dropout(0.25))                             # Layer 5
    
    # Flatten Layer
    CNNModel.add(Flatten())                                 # Layer 6
    
    # Dense/Fully Connected Layer : 128 Neurons using the relu activation function
    CNNModel.add(Dense(128, activation='relu'))             # Layer 7
    
    # Drop Out Layer: This layer used to drop 50% of neurons in the model.
    CNNModel.add(Dropout(0.5))                              # Layer 8
    
    # Dense/Fully Connected Layer : 50 Neurons using the relu activation function
    CNNModel.add(Dense(50, activation='relu'))              # Layer 9
    
    # Dense/Fully Connected Layer : number of families neurons, using the softmax activation function
    CNNModel.add(Dense(noOfFamilies, activation='softmax')) # Layer 10
    
    CNNModel.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return CNNModel     # Return the Convolutional Neural Network Model


# Measure the effectiveness of the model. This section is used to generate the results of the 
# corresponding convolutional neural network. For the purpose of this mode, there are three main 
# measurements which have to be declared. These are recall, preicision and f1. Listed below
# are the neccessary formulas which are required to obtain these results.
from keras import backend as K

# Recall Measurement
def recallMeasurement(testY, yPrediction):
    true_positives = K.sum(K.round(K.clip(testY * yPrediction, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(testY, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Precision Measurement
def precisionMeasurement(testY, yPrediction):
    true_positives = K.sum(K.round(K.clip(testY * yPrediction, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(yPrediction, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# F1 Measurement
def f1Measurement(testY, yPrediction):
    precision = precisionMeasurement(testY, yPrediction)
    recall = recallMeasurement(testY, yPrediction)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Initialise the Model
CNNModel = CNNModel()
# Print the Information of the Model. This includes the information surrounding 
# the layers of the models as well as the number of total parameters
CNNModel.summary()

trainY.shape                            # Returns the shape of the trainY array 
trainY_new = np.argmax(trainY, axis=1)  # Converts the trainY array to an array of integers 
trainY_new                              # Returns the array of integer labels.



# This section of the model imports the class_weight function from the sklearn.utils 
# library. This is used to compute the class weights for a multi-class classification problem.
# The weightOfModel computes the class weights based on the class distribution found in the 
# training set. The class_weight parameter is set to balanced which means that the class 
# weights are automatically adjusted in acorrdance with the information obtained from the 
# input data.

from sklearn.utils import class_weight # Used to compute the class weights for a multi-class classification system
weightOfModel = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(trainY_new),
                                                 y = trainY_new)

# This code is used to map each unique class label in the training set trainY_new variable. 
# This is then uses the class_weight variable initialised above. 
weightOfModel = {l:c for l,c in zip(np.unique(trainY_new), weightOfModel)}

# This is used to compile the Convolutional Neural network Model. It uses the adam optimiser, 
# computes loss using binary_crossentropy and then calculates the measurements listed before.
CNNModel.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=[f1Measurement,precisionMeasurement, 
                    recallMeasurement,
                    AUC()])


# This section of the model is used to trained the convolutional neural network model. 
# This is achieved by using the fit() method from the Keras API of the TensorFlow library which 
# was imported previously. Here is where the number of epochs are declared for the model also
CNNModel.fit(trainX, trainY, validation_data=(testX, testY), 
                epochs=80,  
                class_weight=weightOfModel)


# Evaluates the trained  model on the test data and returns the values of the  metrics
scores = CNNModel.evaluate(testX, testY) 

# Evaluates the trained model on the test data and declares the loss, accuracy 
# f1_score, precision and recall
loss, accuracy, f1_score, precision, recall = CNNModel.evaluate(testX, testY)

# Used to Print the Final CNN Accuracy of the model
print('Final CNN accuracy: ', scores[1])