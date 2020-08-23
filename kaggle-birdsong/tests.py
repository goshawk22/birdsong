import librosa
audio, sr = librosa.load("test.mp3", duration=5, sr=11025)

mel = librosa.feature.melspectrogram(y=audio,sr=sr)
print(mel.shape)

stft = librosa.stft(audio)
print(stft.shape)

chroma = librosa.feature.chroma_stft(S=stft, n_chroma=128)
print(chroma.shape)

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
print(mfcc.shape)

tone = librosa.feature.tonnetz(y=audio, sr=sr)
print(tone.shape)