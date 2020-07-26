import json
import os

with open('ids2.json', 'r') as handle:
    labels = json.load(handle)
    lists = []
    a = 0
    newLabels = {}

for key in labels:
    newKey = key[0:8] + ".mp3"
    print(newKey)
    os.system('mv /media/hdd/birdsong/"' + key + '" /media/hdd/birdsong/"' + newKey + '"')
    newLabels.update({newKey:labels[key]})

with open("renamedIds.json","w") as f:
    json.dump(newLabels,f)