import numpy as np
import json
import csv
import pickle
import os

a = 0
myBirds = []
with open('renamedIds.json', 'r') as handle:
    labels = json.load(handle)
with open('myBirds.txt', 'r') as handle:
    reader = csv.reader(handle)
    for row in reader:
        myBirds.append(row)
print(len(labels))
print(myBirds[0])
myBirds = myBirds[0]
with open ('cleanedAllSpecies.pkl', 'rb') as fp:
    allSpecies = pickle.load(fp)

print(len(allSpecies))

newIds = {}

for species in allSpecies:
    print(species)
    if species in myBirds:
        print(species)
        a = 0
        for key in labels:
            #print(key)
            if labels[key] == species:
                newIds.update({key:species})
                a += 1

print(len(newIds))
speciesCount = {}
for species in allSpecies:
    if species in myBirds:
        print(species, ": True")
        speciesCount.update({species:0})
        for key in labels:
            if labels[key] == species:
                speciesCount.update({species:speciesCount[species]+1})


with open('recordingsCount.json', 'w+') as f:
    json.dump(speciesCount, f)

with open('ids.pkl', 'wb') as fp:
    pickle.dump(newIds, fp)
