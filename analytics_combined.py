import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *


def write_analytics(locations, filename):

    # projects_users will keep track of every user that has contributed to a project. 
    # The Keys are the Picture Id and the value is a set of usernames
    projects_users = dict()

     # Calculate the entropy of the colors in every project
    color_count = dict()


    # Find the number of updates for every project
    # project_updates is a dictionary with the project ID as key and number of updates as value
    project_updates = dict()

    # Find all the projects that each user has contributed to 
    # user_projects is a dictionary with the user ID as key and list of projects that a particular user has contributed to as value
    user_projects = dict()

    # Find the oldest and the newest point of every project and return them as a tuple
    times_count = {}


    with open(filename,'r') as file:
        # Skip first line (header row)
        next(file, None)

        reader = csv.reader(file)

        for r in reader:
            time = int(r[0])
            user = r[1]
            x = float(r[2])
            y = float(r[3])
            color = r[4]

            boundary_list = create_sorted_lists(locations)
            filter_lists(Point(x,y), boundary_list)
            for boundary in boundary_list:
                pic_id = boundary[0]
                path = locations.get(pic_id)

                if projects_users.get(pic_id) is None:
                    projects_users[pic_id] = set()
                if color_count.get(pic_id) is None:
                    color_count[pic_id] = { 'pixel_count': 0 }

                if project_updates.get(pic_id) is None:
                    project_updates[pic_id] = 0 

                if times_count.get(pic_id) is None:
                    times_count[pic_id] = { "times": [], "starting_time": "NA", "ending_time:": "NA", "duration": "NA" }
                  # If this pixel is inside the image, then this user has contributed to the image
                if ( path.pixel_is_in_image(Point(x,y))):
                    projects_users[pic_id].add(user)

                    project_updates[pic_id] += 1

                    times_count[pic_id]["times"].append(time)

                    if user in user_projects:
                        if pic_id not in user_projects[user]:
                            user_projects[user].add(pic_id)
                    else:
                        user_projects[user] = set()
                        user_projects[user].add(pic_id)

                    color_count[pic_id]['pixel_count'] += 1
                    if color_count.get(pic_id).get(color) is None:
                        color_count[pic_id][color] = 1
                    else:
                        color_count[pic_id][color] += 1

    for pic_id in color_count:
        if color_count[pic_id]['pixel_count'] > 0:
            color_count[pic_id]['entropy'] = calculate_color_entropy(color_count.get(pic_id))
        else:
            color_count[pic_id]['entropy'] = "NA"

    for pic_id in times_count:
        if len(times_count.get(pic_id).get("times")) > 0:
            times_count[pic_id]["times"].sort()
            starting_time = times_count.get(pic_id).get("times")[0]
            ending_time = times_count.get(pic_id).get("times")[-1]
            times_count[pic_id]["starting_time"] = starting_time
            times_count[pic_id]["ending_time"] = ending_time
            times_count[pic_id]["duration"] = ending_time - starting_time



    with open("users_per_project.csv", "w") as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Pic Id", "Num Users", "Users"])
        for pic_id in projects_users:
            writer.writerow([pic_id, len(projects_users.get(pic_id))] + list(projects_users.get(pic_id)))

    # Write these results in a CSV file
    with open("color_per_picture_entropy.csv",'w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Pic Id", "Entropy", "Color 0", "Color 1", "Color 2", "Color 3", "Color 4", "Color 5", "Color 6", "Color 7", "Color 8", "Color 9", "Color 10", "Color 11", "Color 12", "Color 13", "Color 14", "Color 15"])
        for pic_id in color_count:
            writer.writerow([pic_id, color_count.get(pic_id).get('entropy'), color_count.get(pic_id).get('0'), color_count.get(pic_id).get('1'), color_count.get(pic_id).get('2'), color_count.get(pic_id).get('3'), color_count.get(pic_id).get('4'), color_count.get(pic_id).get('5'), color_count.get(pic_id).get('6'), color_count.get(pic_id).get('7'), color_count.get(pic_id).get('8'), color_count.get(pic_id).get('9'), color_count.get(pic_id).get('10'), color_count.get(pic_id).get('11'), color_count.get(pic_id).get('12'), color_count.get(pic_id).get('13'), color_count.get(pic_id).get('14'), color_count.get(pic_id).get('15') ])


    with open('updates_per_project.csv','w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Project ID", "Number of updates"])
        for proj in project_updates:
            writer.writerow([proj, project_updates.get(proj)])

    with open('projects_per_user.csv','w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["user ID", "list of projects", "number of projects"])
        for u in user_projects:
            writer.writerow([u, user_projects.get(u), len(user_projects.get(u))])

    with open("times_per_project.csv",'w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Pic ID", "Start Time", "End Time", "Duration", "All Times"])
        for pic_id in times_count:
            starting_time = times_count.get(pic_id).get("starting_time")
            ending_time = times_count.get(pic_id).get("ending_time")
            duration = times_count.get(pic_id).get("duration")
            times = times_count.get(pic_id).get("times")
            writer.writerow([pic_id, starting_time, ending_time, duration, times])


    None

def write_pixels_per_project(locations):
    # Find the number of pixels in each image
    # project_pixels is a dictionary with the project ID as key and number of pixels in that project as value
    project_pixels = dict()

    boundary_list = create_sorted_lists(locations)
    for boundaries in boundary_list:
        pic_id = boundaries[0]
        top = int(boundaries[1])
        left = int(boundaries[2])
        right = int(boundaries[3])
        bottom = int(boundaries[4])
        path = locations.get(pic_id)
        project_pixels[pic_id] = 0
        for y in range(bottom, top + 1):
            for x in range(left, right + 1):
                test_point = Point(x,y)
                if ( path.pixel_is_in_image(test_point) ):
                    project_pixels[pic_id] += 1

    with open('pixels_per_project.csv','w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Project", "Number of pixels"])
        for proj in project_pixels:
            writer.writerow([proj, project_pixels.get(proj)])
    return project_pixels

def create_sorted_lists(locations):
  boundary_list = []

  for pic_id in locations:
    path = locations[pic_id]
    boundary_list.append((pic_id, path.get_top(), path.get_left(), path.get_right(), path.get_bottom()))

    # if((path.get_top() == None) or (path.get_left() == None) or (path.get_right() == None) or (path.get_bottom() == None)): 
    #   print(pic_id)

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


def calculate_color_entropy(color_count):
    # Caluculate the entropy of all the colors
    # Equation: Sum of P(color) * log P(color) for all colors
    color_probabilities = []

    for color in color_count:
      color_prob = color_count.get(color) / color_count.get('pixel_count')
      color_probabilities.append(color_prob)

    entropy = 0
    for i in range(len(color_probabilities)):
      entropy += color_probabilities[i] * math.log(color_probabilities[i])

    entropy = -1 * entropy

    return entropy

if __name__ == "__main__":

  locations = store_locations("atlas.json")
  test_point1 = Point(850,460)
  boundary_list = create_sorted_lists(locations)
  filter_lists(test_point1, boundary_list)  
  # print(boundary_list,'\n')

  write_pixels_per_project(locations)
  write_analytics(locations, "tile_placements_sub.csv")
