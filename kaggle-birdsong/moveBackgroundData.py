import os

for x in os.listdir('/media/hdd/split-birdsong/background/'):
    os.system('cp /media/hdd/split-birdsong/background/' + x + '/* /media/hdd/split-birdsong/birdsong/background')