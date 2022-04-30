#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:15:42 2022

@author: kingrice
"""

import splitfolders
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks


def generate_model(input_shape):

    model = keras.Sequential()
    
    #1st Layer
    model.add(keras.layers.Conv2D(16, (3,3), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.BatchNormalization())
    
    #2nd Layer
    model.add(keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.BatchNormalization())
    
    #3rd Layer
    model.add(keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.BatchNormalization())
    
    #4th Layer
    model.add(keras.layers.Conv2D(128, (3,3), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.BatchNormalization())
    
    #5th Layer
    model.add(keras.layers.Conv2D(256, (3,3), strides=(1,1), activation = 'relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.BatchNormalization())
    
    #Flatten output
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.3))
    
    #Output Layer
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    #compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def createAudioImages(inputFolder,inputFile,audioFolder,imageFolder, outputFile):

  mp3_file = os.path.join(inputFolder,inputFile)
  sound = AudioSegment.from_mp3(mp3_file)
  wav_file = os.path.join(inputFolder, outputFile + '.wav')
  sound.export(wav_file,format='wav')
  y,sr = librosa.load(wav_file)
  mels = librosa.feature.melspectrogram(y=y,sr=sr)
  melSpec = plt.imshow(librosa.power_to_db(mels,ref=np.max))
  plt.savefig(os.path.join(inputFolder, outputFile + '.png'))

  imageFolder = os.path.join(inputFolder,imageFolder)
  os.makedirs(imageFolder)

  audioFolder= os.path.join(inputFolder,audioFolder)
  os.makedirs(audioFolder)

  song = AudioSegment.from_file(wav_file, 'wav')
  chunk_length_ms = 3000
  chunks = make_chunks(song, chunk_length_ms)

  #Export all of the individual chunks as wav files
  for i, chunk in enumerate(chunks):
      if i < 10:
          chunk_name = wav_file.split('/')[1].split('.wav')[0] + '.chunk{0}.wav'.format(i)
          chunk_path = os.path.join(audioFolder,chunk_name)

          img_name = chunk_name.split('.wav')[0] + '.png'
          img_path = os.path.join(imageFolder,img_name)
          
          chunk.export(chunk_path, format='wav')
          y,sr = librosa.load(chunk_path)
          mels = librosa.feature.melspectrogram(y=y,sr=sr)
          melSpec = plt.imshow(librosa.power_to_db(mels,ref=np.max))
          plt.savefig(img_path)
          print('exporting ', chunk_name)

def classifySong(imageFolder, model):
    
    all_predictions = []
    for im in os.listdir(imageFolder):
        image_data = keras.preprocessing.image.load_img(os.path.join(imageFolder,im),
                                                    color_mode='rgba',target_size=(256,256))
        image = keras.preprocessing.image.img_to_array(image_data)
        image = np.reshape(image,(1,256,256,4))
        prediction = model.predict(image/255)
        prediction = prediction.reshape((10,)) 
        all_predictions.append(prediction)
    
    all_predictions = np.asarray(all_predictions)
    overall_prediction = np.mean(all_predictions, axis=0)
    #class_label = np.argmax(overall_prediction)
      
    class_labels = ['blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock']
      
    classification = class_labels[np.argmax(overall_prediction)]
      
    color_data = [1,2,3,4,5,6,7,8,9,10]
    cmap = mpl.cm.get_cmap('jet')
    cnorm = mpl.colors.Normalize(vmin=0, vmax=10)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(x=class_labels,height=overall_prediction,
    color=cmap(cnorm(color_data)))
    plt.xticks(rotation=45)
    plt.xlabel('Musical Genres')
    plt.ylabel('Probability')
    ax.set_title('{} (classification: {})'.format(im.split('.chunk')[0],classification))

if __name__ == '__main__':
    
    # Following the execution of buildModelInputFiles.py, navigate to the 
    # 'model_inputs' folder.
    
    splitfolders.ratio('images', output='sorted',
    seed=1337, ratio=(.8, .15, .05), group_prefix=None, move=False)

    train_dir = 'sorted/train/'
    test_dir = 'sorted/test/'
    val_dir = 'sorted/val/'
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,color_mode='rgba',
                                                        class_mode='categorical',
                                                        batch_size=128)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,color_mode='rgba',
                                                        class_mode='categorical',
                                                        batch_size=128)
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(val_dir,color_mode='rgba',
                                                        class_mode='categorical',
                                                        batch_size=128)
    
    #flow_from_directory loads the images as size 256x256 as default.
    input_shape = (256,256,4)
    
    #build the model
    augmented_model = generate_model(input_shape)
    augmented_model.summary()
    
    #train the CNN
    history = augmented_model.fit(train_generator,epochs=100,validation_data=val_generator)
    
    #evaluate the CNN on the test set
    test_error, test_accuracy = augmented_model.evaluate(test_generator, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    # generate the plots for the CNN history
    print(history.history.keys())
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN Model Accuracy (Augmented Data)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN Model Loss (Augmented Data)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    # construct dat inputs unrelated to the GTZAN dataset to test the performance of the model
    
    # Running this portion of the code should take place in a different folder
    # that is seprate from the 'model_inputs' folder. In the same directory as 
    # 'model_inputs', create a 'model_test' folder that contains directories
    # for each of the seperat experiments. The example here operates on a file called 
    # '01. UB40 - Red Red Wine_sample.mp3' located at 'model_tests/RedWine'
    inputFolder = 'RedWine'
    inputFile = '01. UB40 - Red Red Wine_sample.mp3'
    audioFolder = 'audio'
    imageFolder = 'images'
    outputFile = 'UB40 - Red Red Wine'
    createAudioImages(inputFolder,inputFile,audioFolder,imageFolder, outputFile)
    
    # Determine the genre classification of the test file
    imageFolder = 'RedWine/images'
    classifySong(imageFolder, augmented_model)
    
    
    