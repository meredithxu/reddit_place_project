import csv
import re
import json
from line import *
from point import *
from path import *


def store_locations(js_filename):
    locations = dict()
    with open(js_filename) as f:
        data = json.load(f)

    for element in data["atlas"]:
        pic_id = element["id"]
        path = Path(pic_id)
        points = element["path"]

        if len(points) > 0:
            # The first point in points is also the ending point, so add a copy of it to the end
            first_element = points[0]
            points.append(first_element)
            for i in range(len(points) - 1):
                start_x = points[i][0]
                start_y = points[i][1]
                end_x = points[i+1][0]
                end_y = points[i+1][1]

                point1 = Point(start_x, start_y)
                point2 = Point(end_x, end_y)
                line = Line(point1, point2)
                path.add_line(line)

        locations[pic_id] = path

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


    locations = store_locations('atlas.json')
    

    print("TEST SUB DATA!!!!!!!!!!!!!!!!")
    print(spatialData(locations,Point(500,0),Point(500,500),Point(0,0),Point(0,500)))


