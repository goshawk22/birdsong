import csv
import numpy as np
import os

with open("labels.csv", mode='r') as birdLabels:
    reader = csv.reader(birdLabels)
    labels = {rows[1] for rows in reader}

for bird in labels:
    print(bird)
    os.system('wget -O json/' + bird.replace(" ", "") + '-query.json https://www.xeno-canto.org/api/2/recordings?query=' + bird.replace(" ", "%20"))