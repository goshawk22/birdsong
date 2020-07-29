        S = librosa.feature.melspectrogram(y=data,sr=sr)

        melspectrogram = np.mean(S, axis=0)

        S_db = np.mean(librosa.amplitude_to_db(np.abs(S), ref=np.max), axis=0)

        #short term fourier transform
        stft = np.abs(librosa.stft(data))

        #mfcc
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40), axis=0)

        #chroma
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr))

        #spectral contrast
        #contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr), axis=0)

        #tonetz
        tonetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr), axis=0)


        newData.append(np.hstack((melspectrogram, S_db, mfcc, chroma, tonetz)))