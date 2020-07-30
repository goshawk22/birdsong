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
        self.HOP_SIZE = 216
        self.N_MELS = 256
        self.WIN_SIZE = 1024
        self.WINDOW_TYPE = 'hann'
        self.FEATURE = 'mel'
        self.FMIN = 1000

        with open('renamedIds.json', 'r') as handle:
            self.labels = json.load(handle)
        with open("labels.csv", 'r') as handle:
            reader = csv.reader(handle)
            self.labelIDs = {rows[1]:rows[2] for rows in reader}
            #print(self.labelIDs)
        
    def getSpecies(self, file):
        #id = file[2:(len(file)-5)]
        #print(id)
        label = self.labels[file]
        return label

    def spectogram(self, file):
        data, sr = librosa.load("/media/hdd/birdsong/" + file, duration=5, sr=11025)

        newData = []
        S = librosa.feature.melspectrogram(y=data,sr=sr)

        melspectrogram = np.mean(S, axis=0)

        S_db = np.mean(librosa.amplitude_to_db(np.abs(S), ref=np.max), axis=0)

        #short term fourier transform
        stft = np.abs(librosa.stft(data))

        #mfcc
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40), axis=0)

        #chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=0)

        #spectral contrast
        #contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr), axis=0)

        #tonetz
        tonetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr), axis=0)



        #print(len(newData))
        return melspectrogram, S_db, mfcc, chroma, tonetz


    def createData(self, dir):
        Y_train = []
        mel = []
        db = []
        mfcc = []
        chroma = []
        tone = []
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
            for m in x[0]:
                mel.append(m)
            for d in x[1]:
                db.append(m)
            for m in x[2]:
                mfcc.append(m)
            for c in x[3]:
                chroma.append(m)
            for t in x[4]:
                tone.append(m)
            for yt in x[5]:
                Y_train.append(yt)
                print(yt)
            for en in x[6]:
                englishName.append(en)
        mel = np.array(mel)
        db = np.array(db)
        mfcc = np.array(mfcc)
        chroma = np.array(chroma)
        tone = np.array(tone)
        Y_train = np.array(Y_train)
        englishName = np.array(englishName)
        print(mel.shape, db.shape, mfcc.shape, chroma.shape, tone.shape, Y_train.shape, englishName.shape)
        np.savez_compressed(dir, mel=mel, db=db, mfcc=mfcc, chroma=chroma, tone=tone, Y_train=Y_train, englishName = englishName)

@ray.remote
def createDataSplit(files, labelIDs):
    Y_train = []
    mel = []
    db = []
    mfcc = []
    chroma = []
    tone = []
    realFiles = listdir("/media/hdd/birdsong")
    englishName = []
    ld = loaddata()
    for f in files:
        if f in realFiles:
            spe = ld.getSpecies(f)
            if spe in labelIDs:
            #if spe in self.valids:
                temp = ld.spectogram(f)
                mel.append(temp[0])
                db.append(temp[1])
                mfcc.append(temp[2])
                chroma.append(temp[3])
                tone.append(temp[4])
                Y_train.append(int(labelIDs[spe]))
                englishName.append(spe)
    
    return mel, db, mfcc, chroma, tone, Y_train, englishName

ld = loaddata()
ld.createData("allSplitData")