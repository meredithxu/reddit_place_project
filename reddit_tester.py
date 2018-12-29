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
print("SUCCESS")



#Test something that is not a square
italyFlag = locations.get('13')