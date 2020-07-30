import numpy as np
#from loaddata import loaddata
import librosa  # python package for music and audio analysis
#import librosa.display
import matplotlib.pyplot as plt
import librosa.display
import pickle
import soundfile as sf

file = "xc25119.flac"
N_FFT = 1024    
HOP_SIZE = 216
N_MELS = 256
WIN_SIZE = 1024
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1000


data, sr = librosa.load("/home/adam/GitHub/birdsong/songs/songs/" + file, sr=11025, duration=5)

newData = []
S = librosa.feature.melspectrogram(y=data,sr=sr)

melspectrogram = np.mean(S, axis=0)
print("Melspectogtram: ", melspectrogram.shape)

S_db = np.mean(librosa.amplitude_to_db(np.abs(S), ref=np.max), axis=0)
print("S_db", S_db.shape)

#short term fourier transform
stft = np.abs(librosa.stft(data))

#mfcc
mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40), axis=0)
print("MFCC", mfcc.shape)

#chroma
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=0)
print("Chroma", chroma.shape)
#spectral contrast
#contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr), axis=0)

#tonetz
tonetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr), axis=0)
print("Tonetz", tonetz.shape)

newData = np.vstack((melspectrogram, S_db, mfcc, chroma, tonetz))
print(newData.shape)