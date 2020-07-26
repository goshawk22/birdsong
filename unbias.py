import numpy as np
import json
import csv
import pickle

a = 0

with open('renamedIds.json', 'r') as handle:
    labels = json.load(handle)
print(len(labels))
with open ('allSpecies', 'rb') as fp:
    allSpecies = pickle.load(fp)

print(len(allSpecies))

newIds = {}

for species in allSpecies:
    a = 0
    for key in labels:
        #print(key)
        if labels[key] == species:
            newIds.update({key:species})
            a += 1
        if a == 100:
            a = 0
            break

print(len(newIds))
speciesCount = {}
for species in allSpecies:
    speciesCount.update({species:0})
    for key in labels:
        if labels[key] == species:
            speciesCount.update({species:speciesCount[species]+1})


with open('recordingsCOunt.json', 'w+') as f:
    json.dump(speciesCount, f)

with open('unbiasedIds.pkl', 'wb') as fp:
    pickle.dump(newIds, fp)

for key in speciesCount:
    if speciesCount[key] <= 100:
        print(key)