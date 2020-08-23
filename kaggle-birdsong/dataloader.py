import numpy as np
import ray
import csv
import os
import librosa
import pickle
ray.init()

transformations = ['mel', 'tone', 'y', 'chroma', 'mfcc']

@ray.remote
class loaddata():
    def __init__(self, xdatadir, species, i):
        self.x_data = xdatadir
        self.speciesDirs = os.listdir(xdatadir)
        with open('speciesIDS.pkl', 'rb') as f:
            self.speciesIDS = pickle.load(f)
        self.species = species[i]
        self.i = i

    def spectogram(self, split, sr):
        mel = []
        mfcc = []
        chroma = []
        tone = []
        S = librosa.feature.melspectrogram(y=split,sr=sr)
        mel.append(S)
        #short term fourier transform
        stft = np.abs(librosa.stft(split))
        #mfcc
        mfcc.append(librosa.feature.mfcc(y=split, sr=sr, n_mfcc=40))
        #chroma
        chroma.append(librosa.feature.chroma_stft(S=stft, sr=sr))
        #tone
        tone.append(librosa.feature.tonnetz(y=librosa.effects.harmonic(split), sr=sr))

        return mel, mfcc, chroma, tone

    def spectogram_mel(self, split, sr):
        mel = []
        S = librosa.feature.melspectrogram(y=split,sr=sr)
        mel.append(S)
        return mel

    def spectogram_mfcc(self, split, sr):
        mfcc = []
        #mfcc
        mfcc.append(librosa.feature.mfcc(y=split, sr=sr, n_mfcc=128))
        return mfcc

    def spectogram_chroma(self, split, sr):
        chroma = []
        #short term fourier transform
        stft = librosa.stft(split)
        #chroma
        chroma.append(librosa.feature.chroma_stft(S=stft, n_chroma=16))

        return chroma
    def spectogram_tone(self, split, sr):
        tone = []
        tone.append(librosa.feature.tonnetz(y=librosa.effects.harmonic(split), sr=sr))

        return tone

    def splitFile(self, filename, species, split_size=5):
        path = "".join([self.x_data, "/", species, "/", str(filename)])
        try:
            tfile, tsr = librosa.load(path)
            dur = librosa.get_duration(y=tfile, sr=tsr)

            if int(dur) >= 5:
                num_splits = np.floor(int(dur)/split_size)
                files = []
                for i in range(int(num_splits)):
                
                    try:
                        file, sr = librosa.load(path, duration=split_size, offset=i*split_size, sr=11025)
                    except ZeroDivisionError:
                        print(path)
                        errorSpecies = species
                        return None, errorSpecies
                    files.append(file)
                return files, sr
            else:
                return None, None
        except ValueError:
            print(path)
            return None, None

    def convertData(self):
        Y_train = []
        mel = []
        db = []
        mfcc = []
        chroma = []
        tone = []

        for spec in self.species:
            iter = 0
            errors = []
            for filename in os.listdir('/media/hdd/split-birdsong/birdsong/'+spec):
                while iter < 20:
                    print("Converted 1 file: ", spec)
                    splits, sr = self.splitFile(str(filename), str(spec))
                    if not splits == None:
                        for x in splits:
                            temp = self.spectogram(x, sr)
                            mel.append(temp[0])
                            mfcc.append(temp[1])
                            chroma.append(temp[2])
                            tone.append(temp[3])
                            Y_train.append(self.speciesIDS[spec])
                            '''
                            if transformation == 'mel':
                                temp = self.spectogram_mel(x, sr)
                                mel.append(temp)
                            if transformation == 'mfcc':
                                temp = self.spectogram_mfcc(x, sr)
                                mfcc.append(temp)
                            if transformation == 'chroma':
                                temp = self.spectogram_chroma(x, sr)
                                chroma.append(temp)
                            if transformation == 'tone':
                                temp = self.spectogram_tone(x, sr)
                                tone.append(temp)
                            if transformation == 'y':
                                Y_train.append(self.speciesIDS[spec])
                            '''
                    else:
                        errors.append(sr)
                    iter += 1
        with open('errors_'+self.i+'.pkl', 'wb') as handle:
            pickle.dump(errors, handle)
        return [Y_train, mel, mfcc, chroma, tone]

allSpecies = np.array(os.listdir("/media/hdd/split-birdsong/birdsong"))
allSpecies = np.array_split(allSpecies, 16)
print(allSpecies)
'''
with open('speciesFilesDict.pkl', 'rb') as handle:
    dirs = pickle.load(handle)
    print(len(dirs))
a = {}
b = {}
c = {}
d = {}
e = {}
f = {}
g = {}
h = {}
i = {}
j = {}
k = {}
l = {}
m = {}
n = {}
o = {}
p = {}
dicts = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
print(len(dicts))
x = 0
for key in dirs:
    if x == 16:
        x = 0
    dicts[x].update({key: dirs[key]})
    x += 1
length = 0
for x in dicts:
    length += len(x)
assert length == len(dirs)

for x in dicts:
    print('background' in x)
#print(dicts)
print(dicts[14]['background'])
'''
workers = [loaddata.remote('/media/hdd/split-birdsong/birdsong', allSpecies, i) for i in range(16)]
futures = [c.convertData.remote() for c in workers]
out = ray.get(futures)

Y_train = []
mel = []
mfcc = []
chroma = []
tone = []

for x in out:
    for y in x[0]:
        Y_train.append(y)
    for m in x[1]:
        mel.append(m[0])
    for mf in x[2]:
        mfcc.append(mf[0])
    for ch in x[3]:
        chroma.append(ch[0])
    for t in x[4]:
        tone.append(t[0])

#np.savez_compressed("/media/hdd/Kaggle/converted_background_2D", Y_train=Y_train, mel=mel, mfcc=mfcc, chroma=chroma, tone=tone)
'''
    if i == 'y':
        for x in out:
            for y in x[0]:
                Y_train.append(y)
        np.savez_compressed("/media/hdd/Kaggle/converted_background_2D_Ytrain", Y_train=Y_train)
    if i == 'mel':
        for x in out:
            for m in x[1]:
                mel.append(m)
        np.savez_compressed("/media/hdd/Kaggle/converted_background_2D_mel", mel=mel)
    if i == 'mfcc':
        for x in out:
            for mf in x[2]:
                mfcc.append(mf)
        np.savez_compressed("/media/hdd/Kaggle/converted_background_2D_mfcc", mfcc=mfcc)
    if i == 'chroma':
        for x in out:
            for ch in x[3]:
                chroma.append(ch)
        np.savez_compressed("/media/hdd/Kaggle/converted_background_2D_chroma", chroma=chroma)
    if i == 'tone':
        for x in out:
            for t in x[4]:
                tone.append(t)
        np.savez_compressed("/media/hdd/Kaggle/converted_background_2D_tone", tone=tone)
    out = []
'''
