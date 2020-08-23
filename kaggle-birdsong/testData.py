import numpy as np
import pickle

with open('speciesIDS.pkl', 'rb') as handle:
    speciesIDS = pickle.load(handle)
print(speciesIDS)


with np.load('/media/hdd/Kaggle/converted_background_2D.npz') as data:
    Y_train = data['Y_train']
    mel = data['mel']
    mfcc = data['mfcc']
    chroma = data['chroma']
    tone = data['tone']
print(np.amin(Y_train))
count = 0
vals = []
for x in Y_train:
    if not x in vals:
        vals.append(x)
        count += 1
vals.sort(reverse=False)
print(vals)
print(count)
print(0 in Y_train)


print(tone.shape)
print(Y_train.shape)
print(tone[0].shape)
print(0 in Y_train)
#print(np.amax(Y_train))
'''
species = []
for x in Y_train:
    if not x in species:
        species.append(x)
with open('species.pkl', 'wb') as f:
    pickle.dump(species, f)
print(species)
speciesIDS = {}
y = 1
for x in species:
    speciesIDS.update({x:y})
    y += 1
speciesIDS.update({'background':0})
with open('speciesIDS.pkl', 'wb') as f:
    pickle.dump(speciesIDS, f)

print(speciesIDS)
'''
'''
newMel = np.zeros(tone.shape[0]* 108 * 128).reshape(tone.shape[0], 128, 108)
newMFCC = np.zeros(tone.shape[0]* 108 * 128).reshape(tone.shape[0], 128, 108)
newChroma = np.zeros(tone.shape[0]* 108 * 128).reshape(tone.shape[0], 128, 108)
newTone = np.zeros(tone.shape[0]* 108*6).reshape(tone.shape[0], 6, 108)

for x in range(tone.shape[0]):
    for y in range(6):
        for z in range(108):
            newTone[x,y,z] = tone[x][0][y][z]

    for y in range(128):
        for z in range(108):
            newMel[x,y,z] = mel[x][0][y][z]

    for y in range(40):
        for z in range(108):
            newMFCC[x,y,z] = mfcc[x][0][y][z]

    for y in range(12):
        for z in range(108):
            newChroma[x,y,z] = chroma[x][0][y][z]

newMel = np.array(newMel)
newMFCC = np.array(newMFCC)
newChroma = np.array(newChroma)
newTone = np.array(newTone)

np.savez_compressed("/media/hdd/Kaggle/converted_2D", mel=newMel, mfcc=newMFCC, chroma=newChroma, tone=newTone, Y_train=Y_train)
'''