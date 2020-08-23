import numpy as np
import os
import pickle

files = []
speciesFiles = {}
allSpecies = os.listdir("/media/hdd/split-birdsong/birdsong")
for spec in allSpecies:
    speciesFiles.update({spec: os.listdir('/media/hdd/split-birdsong/birdsong/' + spec)})

with open('speciesFilesDict.pkl', 'wb') as handle:
    pickle.dump(speciesFiles, handle)

print(speciesFiles)