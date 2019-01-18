import csv
import re
from line import *
from point import *
from path import *

# a helper function that organizes coordinates retrieved from JSON file as list of tuples given an ID
def coordHelper(picId,locations):
    coordinates=[]
    for i in range(0,len(locations.get(picId))-1,2):
        coord=(locations.get(picId)[i],locations.get(picId)[i+1])
        coordinates.append(coord)
    return coordinates




def store_locations():
    # extract path values from JSON file and store them in a dictionary whose key is the picture ID(string) and value is a list of coordinates(string) indicating the location of a picture
    locations = dict()
    read = False
    numOfPic = 0
    picId = -1
    with open('atlasTest2.js') as atlasJS:
         for line in atlasJS:
             if '"id":' in line:
                 picId = re.findall('\d+',line)[0]
                 #print(picId)
             if '"path"' in line:
                 read = True
                 numOfPic = numOfPic + 1
                 #print(numOfPic)
             if read:
                 if len(locations) < numOfPic:
                    l = []
                    locations[picId] = l
                    coord = re.findall(r"[-+]?[0-9]*\.?[0-9]+",line)
                    for co in coord:
                        locations.get(picId).append(co)
                 else:
                    coord = re.findall(r"[-+]?[0-9]*\.?[0-9]+",line)
                    #print(coord)
                    if picId not in locations:
                        print("ERROR!!!!!!!!!!")
                        print(picId)
                    for co in coord:
                        locations.get(picId).append(co)
                 if "}" in line:
                     read = False

    print(locations)
    print(len(locations))
    print(numOfPic)
    return locations


# Given the vertices of a rectangle, return data points inside that rectangle
def spatialData(all_data, lo_left_v, lo_right_v, up_left_v, up_right_v):
    left_line = Line(lo_left_v,up_left_v)
    right_line = Line(lo_right_v,up_right_v)
    up_line = Line(up_left_v,up_right_v)
    lo_line = Line(lo_left_v,lo_right_v)
    rectangle = Path(-1)
    rectangle.add_line(left_line)
    rectangle.add_line(right_line)
    rectangle.add_line(up_line)
    rectangle.add_line(lo_line)
    spatial_sub = []

    # iterate through the dictionary. For each picture, check if every coordinate of that picture is inside the rectangle. If so, add picture ID to the subdata
    for key, value in all_data.items():
        coords = coordHelper(key,all_data)
        in_image = True
        for co in coords:
            p = Point(float(co[0]),float(co[1]))
            if not rectangle.pixel_is_in_image(p):
                in_image = False
                break
        if in_image:
            spatial_sub.append(key)
    return spatial_sub


if __name__ == "__main__":

    subData = []
    with open('tile_placements.csv','r') as file:
        reader = csv.reader(file)
        lineCount = 0
        for r in reader:
            if lineCount<=1000:
                subData.append(r)
            if lineCount == 0:
                #print(r)
                lineCount = lineCount+1
            else:
                #print(r[0]+" "+r[1]+" "+r[2]+" "+r[3]+" "+r[4])
                lineCount = lineCount+1
        print(lineCount)

    # extract the first 1000 records in the original dataset as a subset. Stored in 'tile_placements_sub.csv'.
    with open('tile_placements_sub.csv','w') as fileOut:
        writer = csv.writer(fileOut)
        writer.writerows(subData)


    locations = store_locations()
    

    # test code for the helper function
    for pId in range(21):
        coordinates=coordHelper(str(pId),locations)
        print(coordinates)
    print("TEST SUB DATA!!!!!!!!!!!!!!!!")
    print(spatialData(locations,Point(500,0),Point(500,500),Point(0,0),Point(0,500)))


