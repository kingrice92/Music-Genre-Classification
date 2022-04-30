#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:15:42 2022

@author: kingrice
"""

import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

identifiers = []
with open('genres/input.mf') as inputFile:
    for line in inputFile:
        genreSplit = line.rstrip().split('\t')
        filenameSplit = genreSplit[0].split('genres')
        identifier = [genreSplit[1], 'genres'+filenameSplit[1]]
        identifiers.append(identifier)

identifiers = np.asarray(identifiers)
genres = np.unique(identifiers[:,0])

# Make sure that the folder 
os.makedirs('full_songs')
os.makedirs('model_inputs')
imageFolder = os.path.join('model_inputs','images')
os.makedirs(imageFolder)

audioFolder= os.path.join('model_inputs','audio')
os.makedirs(audioFolder)

for genre in genres:
    imageGenreFolder = os.path.join(imageFolder, genre)
    os.makedirs(imageGenreFolder)

    audioGenreFolder = os.path.join(audioFolder, genre)
    os.makedirs(audioGenreFolder)
    
    fullGenreFolder = os.path.join('full_songs',genre)
    os.makedirs(fullGenreFolder)
    
    genreFiles = identifiers[identifiers[:,0]==genre][:,1]
    for wavFile in genreFiles:
        # save whole song as spectograph
        y_full,sr_full = librosa.load(wavFile)
        mels_full = librosa.feature.melspectrogram(y=y_full,sr=sr_full)
        melSpec_full = plt.imshow(librosa.power_to_db(mels_full,ref=np.max))
        full_name = wavFile.split('/')[2].split('.wav')[0] + '.png'
        full_path = os.path.join(fullGenreFolder,full_name)
        plt.savefig(full_path)

        song = AudioSegment.from_file(wavFile, 'wav')
        chunk_length_ms = 3000
        chunks = make_chunks(song, chunk_length_ms) #Make chunks of one sec
        #del chunks[-1]
        #Export all of the individual chunks as wav files
        
        ## need to make sure that the file gets saved properly here
        for i, chunk in enumerate(chunks):
            if i < 10:
                chunk_name = wavFile.split('/')[2].split('.wav')[0] + '.chunk{0}.wav'.format(i)
                
                chunk_path = os.path.join(audioGenreFolder,chunk_name)
                chunk.export(chunk_path, format='wav')
                y,sr = librosa.load(chunk_path)
                mels = librosa.feature.melspectrogram(y=y,sr=sr)
                melSpec = plt.imshow(librosa.power_to_db(mels,ref=np.max))
                img_name = chunk_name.split('.wav')[0] + '.png'
                img_path = os.path.join(imageGenreFolder,img_name)
                plt.savefig(img_path)
                if i%9==0:
                  print('exporting ', chunk_name)