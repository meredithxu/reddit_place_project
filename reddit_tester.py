from line import *
from point import *
from path import *
import csv
import re
from reddit import *

# extract path values from JSON file and store them in a dictionary whose key is the picture ID(string) and value is a list of coordinates(string) indicating the location of a picture

locations = store_locations('atlasTest2.js')


# Every pixel within square (838,415), (838,483), (886, 486), and (886,415) should be a part of the image 
first_image_path = locations.get('0')
# print(first_image_path)
for x in range(839, 887):
  for y in range(416, 484):
    test_point = Point(x,y)
    if not first_image_path.pixel_is_in_image(test_point):
      print("FAILED " + str(x) + "," + str(y))

#Every pixel not in this square should not be a part of the image

for x in range(0, 839):
  for y in range(0, 416):
    test_point = Point(x,y)
    if first_image_path.pixel_is_in_image(test_point):
      print("FAILED " + str(x) + "," + str(y))

for x in range(887, 1000):
  for y in range(484, 1000):
    test_point = Point(x,y)
    if first_image_path.pixel_is_in_image(test_point):
      print("FAILED " + str(x) + "," + str(y))    




# triangle = locations.get('triangle')
# #Test that pixel_is_in_image correctly returns true for points within across a diagonal
# if triangle.pixel_is_in_image(Point(1, 2)):
#   print("FAILED 1, 2")
# if not triangle.pixel_is_in_image(Point(2, 1)):
#   print("FAILED 2, 1")


italyFlag = locations.get('13')

# Test that pixel_is_in_image correctly returns false when given coordinates within a hole in the middle of the polygon
if italyFlag.pixel_is_in_image(Point(135, 359)):
  print("FAILED 135, 359")

if italyFlag.pixel_is_in_image(Point(130, 355)):
  print("FAILED 130,355")

if italyFlag.pixel_is_in_image(Point(120, 360)):
  print("FAILED 120,360")


print("SUCCESS")
