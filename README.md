# Music-Genre-Classification

The GTZAN dataset can be downloaded from https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection.

Once the data is downloaded, unzip the genre folder and run buildModelInputFiles.py. 
Note that in order for this to work, the script must be called from the directory containing the genre folder.

The execution of this sript take a LONG time. buildModelInputFiles.py performs data augmentation on the files of 
the GTZAN dataset to create a larger dataset for ther purpose of training a convolutional neural network, increasing 
the number of data files from 1,000 to 10,000. This script generates images of mel spectrograms for each of the 10,000 data files.
