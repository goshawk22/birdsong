import numpy as np
import librosa
import csv
import os
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

class submission():
    def __init__(self, audio):
        self.path = 'test_audio'
        self.files = os.listdir(self.path)
        self.model = load_model('model.188-0.77.hdf5')

    def spectogram(self, split):
        #Converts a section (split) of the file into various interpretations.
        sr = self.sr
        mel = []
        db = []
        mfcc = []
        chroma = []
        tone = []
        S = librosa.feature.melspectrogram(y=split,sr=sr)
        mel.append(S)
        db.append(librosa.amplitude_to_db(np.abs(S), ref=np.max))
        #short term fourier transform
        stft = np.abs(librosa.stft(split))
        #mfcc
        mfcc.append(librosa.feature.mfcc(y=split, sr=sr, n_mfcc=40))
        #chroma
        chroma.append(librosa.feature.chroma_stft(S=stft, sr=sr))
        #tonetz
        tone.append(librosa.feature.tonnetz(y=librosa.effects.harmonic(split), sr=sr))

        return np.array(mel), np.array(db), np.array(mfcc), np.array(chroma), np.array(tone)
    
    def split(self, path, split_size=5):
        audio, self.sr = librosa.load(path)
        dur = librosa.get_duration(y=audio, sr=self.sr)
        num_splits = np.floor(int(dur)/split_size)
        files = []
        split = np.split(audio, num_splits)
        for i in range(int(num_splits)):
            files.append(split[i])

        return files

    def loadCSV(self):
        self.train = pd.read_csv('../input/birdsong-recognition/train.csv')
        self.birds = self.train['ebird_code'].unique()

        self.TEST_FOLDER = '../input/birdsong-recognition/test_audio/'
        self.test_info = pd.read_csv('../input/birdsong-recognition/test.csv')
    
    def load_test_clip(self, path, start_time, duration=5):
        return librosa.load(path, offset=start_time, duration=duration)

    def getSubmission(self):
        preds = []
        for index, row in self.test_info.iterrows():
            # Get test row information
            site = row['site']
            start_time = row['seconds'] - 5
            row_id = row['row_id']
            audio_id = row['audio_id']

            if site == 'site_1' or site == 'site_2':
                sound_clip = self.load_test_clip(self.TEST_FOLDER + audio_id + '.mp3', start_time)
            else:
                sound_clip = self.load_test_clip(self.TEST_FOLDER + audio_id + '.mp3', 0, duration=None)
            
            pred = self.make_prediction(self.spectogram(sound_clip))
            preds.append([row_id, pred])
        
        preds = pd.DataFrame(preds, columns=['row_id', 'birds'])

        preds.to_csv('submission.csv', index=False)
    
    def make_prediction(self, data):
        #Normalize Data
        melP, DBP, mfccP, chromaP, toneP = self.spectogram(data)
        melP = melP.astype("float32") * 100
        DBP = DBP.astype("float32") * 1000
        mfccP = mfccP.astype("float32") * -1/10
        chromaP = chromaP.astype("float32") * -1/30
        toneP = toneP.astype("float32") * -1/30

        return self.model.predict([melP, DBP, mfccP, chromaP, toneP])
    
    def make_prediction_site_3(self, data):
        preds = []
        for x in data:
            #Normalize Data
            melP, DBP, mfccP, chromaP, toneP = self.spectogram(data)
            melP = melP.astype("float32") * 100
            dbP = DBP.astype("float32") * 1000
            mfccP = mfccP.astype("float32") * -1/10
            chromaP = chromaP.astype("float32") * -1/30
            toneP = toneP.astype("float32") * -1/30
            # Add dimension in place of batch size to avoid error
            melP = melP[[np.newaxis,...]]
            dbP = dbP[[np.newaxis,...]]
            mfccP = mfccP[[np.newaxis,...]]
            chromaP = chromaP[[np.newaxis,...]]
            toneP = toneP[[np.newaxis,...]]

            preds.append(self.model.predict([melP, DBP, mfccP, chromaP, toneP]))
        return preds