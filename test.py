import numpy as np
#from loaddata import loaddata
import librosa  # python package for music and audio analysis
#import librosa.display

file = "xc25119.flac"
N_FFT = 1024         
HOP_SIZE = 1024       
N_MELS = 128
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 1400


data, sr = librosa.load("/home/adam/GitHub/birdsong/songs/songs/" + file, sr=11025)

S = librosa.feature.melspectrogram(y=data,sr=sr,
                                        n_fft=N_FFT,
                                        hop_length=HOP_SIZE, 
                                        n_mels=N_MELS, 
                                        htk=True, 
                                        fmin=FMIN, 
                                        fmax=sr/2)
dir = "file"
S = None
mydict = {'Hello': 'x', 'Goodbye': 'y'}
print(list(mydict))
print(np.split(np.array(list(mydict)), 2))
