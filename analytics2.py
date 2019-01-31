import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *


# def write_users_per_project(locations, filename, write_to_file = False):

#   # user_count will count the number of users assigned to each project
#   # user_count is a dictionary of the following format:
#   #  The index is the number of users working on one project
#   #  The value is the number of projects that have the indexed number of users
#   # EX: If there are 6 projects that have 10 users contributing, then one element in user_count is: user_count.get(10) = 6
#   user_count = {}

#   # projects_users will keep track of every user that has contributed to a project. 
#   # The Keys are the Picture Id and the value is a set of usernames
#   projects_users = {}

#   for pic_id in locations:
#     path = locations.get(pic_id)
#     projects_users[pic_id] = set()

#     with open(filename,'r') as file:
#       # Skip first line (header row)
#       next(file, None)

#       reader = csv.reader(file)
      
#       for r in reader:
#         user = r[1]
#         x = float(r[2])
#         y = float(r[3])

#         # If this pixel is inside the image, then this user has contributed to the image
#         if ( path.pixel_is_in_image(Point(x,y))):
#           projects_users[pic_id].add(user)

#     num_users = len(projects_users[pic_id])

#     if (user_count.get(num_users) is None):
#       user_count[num_users] = 1
#     else:
#       user_count[num_users] += 1

#   if write_to_file:
#     # Write these results in a CSV file
#     with open("num_users_per_project.csv",'w') as fileOut:
#       writer = csv.writer(fileOut, delimiter = ",")
#       writer.writerow(["Number of contributing users x", "Number of projects with x many users"])
#       for num in user_count:
#         writer.writerow([num, user_count.get(num)])

#     with open("users_per_project.csv", "w") as fileOut:
#       writer = csv.writer(fileOut, delimiter = ",")
#       writer.writerow(["Pic Id", "Num Users", "Users"])
#       for pic_id in projects_users:
#         writer.writerow([pic_id, len(projects_users.get(pic_id))] + list(projects_users.get(pic_id)))
        

#   return user_count


def create_sorted_lists(locations):
	boundary_list = []

	for pic_id in locations:
		path = locations[pic_id]
		boundary_list.append((pic_id, path.get_top(), path.get_left(), path.get_right(), path.get_bottom()))

	return boundary_list


def filter_lists(test_point, boundary_list):
	boundary_list.sort(key=lambda x: x[1])
	for idx, b in enumerate (boundary_list):
		if test_point.get_y() <= b[1]:
			del boundary_list[:idx]
			break;
	boundary_list.sort(key=lambda x: x[2])
	for idx, b in enumerate (boundary_list):
		if test_point.get_x() < b[2]:
			del boundary_list[idx:]
			break;
	boundary_list.sort(key=lambda x: x[3])
	for idx, b in enumerate (boundary_list):
		if test_point.get_x() >= b[3]:
			del boundary_list[:idx]
			break;
	boundary_list.sort(key=lambda x: x[4])
	for idx, b in enumerate (boundary_list):
		if test_point.get_y() < b[4]:
			del boundary_list[idx:]
			break;

	return boundary_list

if __name__ == "__main__":

	locations = store_locations("atlasTest2.json")
	test_point1 = Point(850,460)
	boundary_list = create_sorted_lists(locations)
	print(boundary_list,'/n')
	filter_lists(test_point1, boundary_list)
	print(boundary_list,'/n')


	#write_users_per_project(locations, "tile_placements_sub.csv", True)
