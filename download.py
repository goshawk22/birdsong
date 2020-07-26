import ray
import os
import csv
import numpy as np
ray.init()

@ray.remote
def download(i):
    os.system("wget -P /media/hdd/birdsong/ --trust-server-names -i /home/adam/GitHub/birdsong/urls/list" + str(i) + ".txt")
    print("Done")

'''
urls = open("xc-noca-urls.txt")
reader = csv.reader(urls)
allRows = [row for row in reader]

allRows = np.array(allRows)
allRows = np.array_split(allRows, 16)

for file in range(len(allRows)):    
    urls = list(allRows[file])
    with open("urls/list" + str(file) + ".txt", 'w+') as f:
        for url in urls:
            a = url[0]
            f.write("{}\n".format(a))
'''

[download.remote(i) for i in range(16)]