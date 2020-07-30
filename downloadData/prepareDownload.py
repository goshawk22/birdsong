# -*- coding: utf-8 -*-
import json
import pandas as pd
import csv
import pickle
id = {}

with open("labels.csv", mode='r') as birdLabels:
    reader = csv.reader(birdLabels)
    labels = {rows[1] for rows in reader}

for bird in labels:
    # Get the json entries from your downloaded json
    jsonFile = open('json/' + bird.replace(" ", "") + '-query.json', 'r')
    values = json.load(jsonFile)
    jsonFile.close()

    # Create a pandas dataframe of records & convert to .csv file
    record_df = pd.DataFrame(values['recordings'])
    record_df.to_csv('data/' + bird.replace(" ", "") + '-noca.csv', index=False)

    # Make wget input file
    url_list = []
    for file in record_df['file'].tolist():
        url_list.append('https:{}'.format(file))
    with open('xc-noca-urls.txt', 'a') as f:
        for item in url_list:
            f.write("{}\n".format(item))
    
    for fileName in record_df['file-name'].tolist():
        for en in record_df['en']:
           id.update({fileName:en})
    
    with open("ids2.json","w") as f:
        json.dump(id,f)

    audioList = open("audioList.txt", 'a', encoding='utf8')
    for fileName in record_df['file-name'].tolist():
        audioList.write(fileName)
        audioList.write('\n')
    
    audioList.close()