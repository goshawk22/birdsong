# -*- coding: utf-8 -*-
import numpy as np
import librosa
import csv
from os import listdir
from math import floor
import ray
import pickle
import json


ray.init()

class loaddata():
    def __init__(self):
        self.N_FFT = 1024         
        self.HOP_SIZE = 1024       
        self.N_MELS = 128
        self.WIN_SIZE = 1024      
        self.WINDOW_TYPE = 'hann' 
        self.FEATURE = 'mel'      
        self.FMIN = 1400

        with open('unbiasedIds.pkl', 'rb') as handle:
            self.labels = pickle.load(handle)
            #print(type(self.labels))
        with open("labels.csv", 'r') as handle:
            reader = csv.reader(handle)
            self.labelIDs = {rows[1]:rows[2] for rows in reader}
    
    def getSpecies(self, file):
        #id = file[2:(len(file)-5)]
        #print(id)
        label = self.labels[file]
        return label

    def spectogram(self, file):
        data, sr = librosa.load("/media/hdd/birdsong/" + file, sr=11025)
        length = librosa.get_duration(y = data, sr = sr)
        if length < 5:
            print(file)
        data, sr = librosa.load("/media/hdd/birdsong/" + file, sr=11025, duration=5 * floor(length/5))
        length = librosa.get_duration(y = data, sr = sr)

        numSplits = floor(length/5)

        splits = np.split(data, numSplits)

        newData = []
        for split in splits:
            S = librosa.feature.melspectrogram(y=split,sr=sr,
                                        n_fft=self.N_FFT,
                                        hop_length=self.HOP_SIZE, 
                                        n_mels=self.N_MELS, 
                                        htk=True, 
                                        fmin=self.FMIN, 
                                        fmax=sr/2)
            S = S.reshape(128,54)
            newData.append(S)

        #print(len(newData))
        return newData

    def createData(self, dir):
        X_train = []
        Y_train = []
        englishName = []
        files = list(self.labels)
        print(len(files))
        files = np.array_split(np.array(files),16)
        futures = [createDataSplit.remote(i, self.labelIDs) for i in files]
        out = ray.get(futures)
        print(type(out))
        print(len(out))
        #print(len(X_train), len(Y_train))
        for x in out:
            for xt in x[0]:
                X_train.append(xt)
            for yt in x[1]:
                Y_train.append(yt)
            for en in x[2]:
                englishName.append(en)
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        englishName = np.array(englishName)
        print(X_train.shape, Y_train.shape, englishName.shape)
        np.savez_compressed(dir, X_train=X_train, Y_train=Y_train, englishName = englishName)

@ray.remote
def createDataSplit(files, labelIDs):
    X_train = []
    Y_train = []
    realFiles = listdir("/media/hdd/birdsong")
    englishName = []
    ld = loaddata()
    for f in files:
        if f in realFiles:
            for split in ld.spectogram(f):
                X_train.append(split)
                Y_train.append(int(labelIDs[ld.getSpecies(f)]))
                englishName.append(ld.getSpecies(f))
    
    return X_train, Y_train, englishName

ld = loaddata()
ld.createData("largeData")