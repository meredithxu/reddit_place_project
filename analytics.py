import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *

def write_users_per_project(locations, filename, write_to_file = False, write_file = "users_per_project.csv"):

  # Project count will count the number of users assigned to each project
  # project_count is a dictionary of the following format:
  #  The index is the number of users working on one project
  #  The value is the number of projects that have the indexed number of users
  # EX: If there are 6 projects that have 10 users contributing, then one element in project_count is: project_count.get(10) = 6
  user_count = {}

  for pic_id in locations:
    path = locations.get(pic_id)
    users = set()

    with open(filename,'r') as file:
      # Skip first line (header row)
      next(file, None)

      reader = csv.reader(file)
      
      for r in reader:
        user = r[1]
        x = float(r[2])
        y = float(r[3])

        # If this pixel is inside the image, then this user has contributed to the image
        if ( path.pixel_is_in_image(Point(x,y))):
          users.add(user)

    num_users = len(users)

    if (user_count.get(num_users) is None):
      user_count[num_users] = 1
    else:
      user_count[num_users] += 1

  if write_to_file:
    # Write these results in a CSV file
    with open(write_file,'w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Number of contributing users x", "Number of projects with x many users"])
      for num in user_count:
        writer.writerow([num, user_count.get(num)])


  return user_count


def get_time_per_project(locations, filename, write_to_file = False, write_file = "time_per_project.csv"):
  # Find the oldest and the newest point of every project and return them as a tuple
  times_count = {}

  for pic_id in locations:
    path = locations.get(pic_id)
    min_time = sys.maxsize
    max_time = -sys.maxsize
    with open(filename,'r') as file:
      # Skip first line (header row)
      next(file, None)

      reader = csv.reader(file)
      
      for r in reader:
        time = int(r[0])
        x = float(r[2])
        y = float(r[3])

        # If this pixel is inside the image, then this user has contributed to the image
        if ( path.pixel_is_in_image(Point(x,y)) ):
          if time > max_time:
            max_time = time
          if time < min_time:
            min_time = time


    time_alive = max_time - min_time

    if min_time > max_time:

      times_count[pic_id] = ("NA", "NA", 0)
    else:
      times_count[pic_id] = (min_time, max_time, time_alive)


  if write_to_file:
    # Write these results in a CSV file
    with open(write_file,'w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Pic ID", "Start Time", "End Time", "Duration"])
      for pic_id in times_count:
        writer.writerow([pic_id, times_count.get(pic_id)[0], times_count.get(pic_id)[1], times_count.get(pic_id)[2]])


  return times_count

def get_color_per_project(locations, filename, write_to_file = False, write_file = "color_per_project.csv"):
  
  # Calculate the entropy of the colors in every project
  color_count = {}

  for pic_id in locations:
    path = locations.get(pic_id)
    color_count[pic_id] = {}
    total_pixels = 0
    with open(filename,'r') as file:
      # Skip first line (header row)
      next(file, None)

      reader = csv.reader(file)
      
      for r in reader:
        color = r[4]
        x = float(r[2])
        y = float(r[3])

        # color_count keeps track of how many pixels of each color are used in a project
        # If this pixel is inside the image, then update the counter
        if ( path.pixel_is_in_image(Point(x,y))):
          if color_count.get(pic_id).get(color) is None:
            color_count[pic_id][color] = 1
          else:
            color_count[pic_id][color] += 1

          total_pixels += 1

    if total_pixels > 0:
      color_count[pic_id]['pixel_count'] = total_pixels
    else:
      color_count.pop(pic_id, None)


  for pic_id in color_count: 
    color_probabilities = []

    for color in color_count.get(pic_id):
      color_prob = color_count.get(pic_id).get(color) / color_count.get(pic_id).get('pixel_count')
      color_probabilities.append(color_prob)

    entropy = 0
    for i in range(len(color_probabilities)):
      entropy += color_probabilities[i] * math.log(color_probabilities[i])

    entropy = -entropy
    color_count[pic_id]['entropy'] = entropy

  if write_to_file:
    # Write these results in a CSV file
    with open(write_file,'w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Pic Id", "Entropy", "Color 0", "Color 1", "Color 2", "Color 3", "Color 4", "Color 5", "Color 6", "Color 7", "Color 8", "Color 9", "Color 10", "Color 11", "Color 12", "Color 13", "Color 14", "Color 15"])
      for pic_id in color_count:
        writer.writerow([pic_id, color_count.get(pic_id).get('entropy'), color_count.get(pic_id).get('0'), color_count.get(pic_id).get('1'), color_count.get(pic_id).get('2'), color_count.get(pic_id).get('3'), color_count.get(pic_id).get('4'), color_count.get(pic_id).get('5'), color_count.get(pic_id).get('6'), color_count.get(pic_id).get('7'), color_count.get(pic_id).get('8'), color_count.get(pic_id).get('9'), color_count.get(pic_id).get('10'), color_count.get(pic_id).get('11'), color_count.get(pic_id).get('12'), color_count.get(pic_id).get('13'), color_count.get(pic_id).get('14'), color_count.get(pic_id).get('15') ])


  return color_count


if __name__ == "__main__":
  locations = store_locations()
  print(locations)
  write_users_per_project(locations, "tile_placements_sub.csv", True)
  get_time_per_project(locations, "tile_placements_sub.csv", True)
  get_color_per_project(locations, "tile_placements_sub.csv", True)






        
        
