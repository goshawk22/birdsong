import librosa
import os
import ray
import numpy as np
import time
ray.init()

@ray.remote
def clean(files):
    for file in files:
        try:
            data, sr = librosa.load("/media/hdd/birdsong/" + file, sr=11025)
            length = librosa.get_duration(y = data, sr = sr)
        except ZeroDivisionError:
            os.system('rm /media/hdd/birdsong/"' + file + '"')
        if length < 5:
            os.system('rm /media/hdd/birdsong/"' + file + '"')
        
        #time.sleep(0.2)



files = os.listdir("/media/hdd/birdsong")
files = np.array(files)
files = np.array_split(files, 16)
files = list(files)
futures = [clean.remote(file) for file in files]
print(ray.get(futures))