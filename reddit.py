import csv
import re

def pixel_is_in_image(x_coord, y_coord, path):
    """
        Returns true if pixel is in image cooresponding to pic_id and false if not

        path is a list of the path coordinates of the edge

    """
    #Separate out path, which is a list of floats, into a list of (x,y) tuples
    points = dict()
    while i < len(path):
        
        i += 2

    # Do horizontal line test
    # If line passes through odd number of edges, it is inside the image
    # If line passes through even number of edges, it is outside the image
    num_intersections = 0
    i = 0
    while i < len(path):
        start_x = float(path[i])
        start_y = float(path[i + 1])
        end_x = float(path[0])
        end_y = float(path[1]) 
        if i != len(path) - 2:
            end_x = float(path[i + 2])
            end_y = float(path[i + 3])

        # Check if the point is on the border. If it is, return true
        if x_coord == start_x or x_coord == end_x or y_coord == start_y or y_coord == end_y:
            return True

    
        if (y_coord <= end_y and y_coord >= start_y) or (y_coord >= end_y and y_coord <= start_y):
            
            if (x_coord >= start_x or x_coord >= end_x):
            
                num_intersections += 1
        # else:
        #     print("x,y:" + str(x_coord) + ", " + str(y_coord))
        #     print("start x: " + str(start_x))
        #     print("end x: " + str(end_x))
        #     print("start y: " + str(start_y))
        #     print("end y: " + str(end_y))

        i += 4

    return (num_intersections % 2 == 1)


def store_locations():
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

    return locations

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

    #extract the first 1000 records in the original dataset as a subset. Stored in 'tile_placements_sub.csv'.
    with open('tile_placements_sub.csv','w') as fileOut:
        writer = csv.writer(fileOut)
        writer.writerows(subData)

    locations = store_locations()

    print(locations)
    print(len(locations))
    # print(numOfPic)



