import csv

import csv

with open('labels.csv', mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]:rows[1] for rows in reader}

print(mydict["148426"])