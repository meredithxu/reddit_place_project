from reddit import *


# Every pixel within square (838,415), (838,483), (886, 486), and (886,415) should be a part of the image 
locations = store_locations()
first_image_path = locations.get('0')
print(first_image_path)
for x in range(839, 887):
  for y in range(416, 484):
    if not pixel_is_in_image(x,y, first_image_path):
      print("FAILED " + str(x) + "," + str(y))

#Every pixel not in this square should not be a part of the image

for x in range(0, 839):
  for y in range(0, 416):
    if pixel_is_in_image(x,y, first_image_path):
      print("FAILED " + str(x) + "," + str(y))

for x in range(887, 1000):
  for y in range(484, 1000):
    if pixel_is_in_image(x,y, first_image_path):
      print("FAILED " + str(x) + "," + str(y))    



triangle = ['0','0','3','3','3','0']

#Test that pixel_is_in_image correctly returns true for points within across a diagonal
if pixel_is_in_image(1, 2, triangle):
  print("FAILED 1, 2")
if not pixel_is_in_image(2, 1, triangle):
  print("FAILED 2, 1")


italyFlag = locations.get('13')

# Test that pixel_is_in_image correctly returns false when given coordinates within a hole in the middle of the polygon
if pixel_is_in_image(135, 359, italyFlag):
  print("FAILED 135, 359")

if pixel_is_in_image(130, 355, italyFlag):
  print("FAILED 130,355")

if pixel_is_in_image(120, 360, italyFlag):
  print("FAILED 120,360")



print("SUCCESS")