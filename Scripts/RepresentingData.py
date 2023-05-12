# This script is used to print out information surrounding the datasets. This shows
# the number of samples, and which family these samples correspond to in a bar chart.
from PIL import Image             # Allow for processing of imagess
import matplotlib.pyplot as plt   # Used to create visualisations

# Delcaring the Path of the Dataset
locationOfDataset = '/home/dmr/DatasetForExperiments/adware/'

# Gather the information about the dataset, how many images and families
from keras.preprocessing.image import ImageDataGenerator
batches = ImageDataGenerator().flow_from_directory(directory=locationOfDataset, 
                                                    target_size=(64,64), 
                                                    batch_size=10000)

# Show the number of families of malware in the dataset
batches.class_indices


# Print information about the images themselves to reaffirm findings 
imgs, labels = next(batches)
imgs.shape
labels.shape     # Prints (BatchSize, NumberOfClasses)

# Plot the information about the quantity of samples and families
classes = batches.class_indices.keys()
perc = (sum(labels)/labels.shape[0])*100
plt.xticks(rotation='vertical')          # Plot the bar chart
plt.bar(classes,perc)                    # Print the resulting bar chart
