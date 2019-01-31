import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *

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


def write_users_per_project(locations, filename, write_to_file = False):

    # projects_users will keep track of every user that has contributed to a project. 
    # The Keys are the Picture Id and the value is a set of usernames
    projects_users = {}


    with open(filename,'r') as file:
        # Skip first line (header row)
        next(file, None)

        reader = csv.reader(file)

        for r in reader:
            user = r[1]
            x = float(r[2])
            y = float(r[3])

            boundary_list = create_sorted_lists(locations)
            filter_lists(Point(x,y), boundary_list)
            for boundary in boundary_list:
                pic_id = boundary[0]
                path = locations.get(pic_id)

                if projects_users.get(pic_id) is None:
                    projects_users[pic_id] = set()
                  # If this pixel is inside the image, then this user has contributed to the image
                if ( path.pixel_is_in_image(Point(x,y))):
                    projects_users[pic_id].add(user)


    if write_to_file:


        with open("users_per_project.csv", "w") as fileOut:
            writer = csv.writer(fileOut, delimiter = ",")
            writer.writerow(["Pic Id", "Num Users", "Users"])
            for pic_id in projects_users:
                writer.writerow([pic_id, len(projects_users.get(pic_id))] + list(projects_users.get(pic_id)))


    return projects_users


def create_sorted_lists(locations):
	boundary_list = []

	for pic_id in locations:
		path = locations[pic_id]
		boundary_list.append((pic_id, path.get_top(), path.get_left(), path.get_right(), path.get_bottom()))

		# if((path.get_top() == None) or (path.get_left() == None) or (path.get_right() == None) or (path.get_bottom() == None)): 
		# 	print(pic_id)

	return boundary_list


def filter_lists(test_point, boundary_list):

	# Sort by top
	boundary_list.sort(key=lambda x: x[1])
	# Discard all figures whose top boundary does not include test_point
	for idx, b in enumerate (boundary_list):
		if test_point.get_y() <= b[1]:
			del boundary_list[:idx]
			break;


	# Sort by left
	boundary_list.sort(key=lambda x: x[2])
    # Discard all figures whose left boundary does not include test_point
	for idx, b in enumerate (boundary_list):
		if test_point.get_x() < b[2]:
			del boundary_list[idx:]
			break;

	# Sort by right
	boundary_list.sort(key=lambda x: x[3])
    # Discard all figures whose right boundary does not include test_point
	for idx, b in enumerate (boundary_list):
		if test_point.get_x() < b[3]:
			del boundary_list[:idx]
			break;

	# Sort by bottom
	boundary_list.sort(key=lambda x: x[4])
    # Discard all figures whose bottom boundary does not include test_point
	for idx, b in enumerate (boundary_list):
		if test_point.get_y() < b[4]:
			del boundary_list[idx:]
			break;

	return boundary_list

if __name__ == "__main__":

	locations = store_locations("atlas.json")
	test_point1 = Point(850,460)
	boundary_list = create_sorted_lists(locations)
	filter_lists(test_point1, boundary_list)	
	# print(boundary_list,'\n')


	write_users_per_project(locations, "tile_placements_sub.csv", True)
