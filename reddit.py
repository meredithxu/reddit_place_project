import csv
import json
subData = []
with open('tile_placements.csv','r') as file:
    reader = csv.reader(file)
    lineCount = 0
    for r in reader:
        if lineCount<=1000:
            subData.append(r)
        if lineCount == 0:
            print(r)
            lineCount = lineCount+1
        else:
            print(r[0]+" "+r[1]+" "+r[2]+" "+r[3]+" "+r[4])
            lineCount = lineCount+1
    print(lineCount)

#extract the first 1000 records in the original dataset as a subset. Stored in 'tile_placements_sub.csv'.
with open('tile_placements_sub.csv','w') as fileOut:
    writer = csv.writer(fileOut)
    writer.writerows(subData)


