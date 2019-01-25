import csv
import sys
from reddit import *
from line import *
from point import *
from path import *

def get_canvas_region(locations):

  # Returns a canvas that covers all the projects in the dataset

  min_x = sys.maxsize
  max_x = -sys.maxsize
  min_y = sys.maxsize
  max_y = -sys.maxsize
  for pic_id in locations:
    path = locations.get(pic_id)
    for l in path.get_all_lines():
      if l.get_point1().get_x() < min_x:
        min_x = l.get_point1().get_x()
      if l.get_point1().get_x() > max_x:
        max_x = l.get_point1().get_x()
      if l.get_point2().get_x() < min_x:
        min_x = l.get_point2().get_x()
      if l.get_point2().get_x() > max_x:
        max_x = l.get_point2().get_x()
      if l.get_point1().get_y() < min_y:
        min_y = l.get_point1().get_y()
      if l.get_point1().get_y() > max_y:
        max_y = l.get_point1().get_y()
      if l.get_point2().get_y() < min_y:
        min_y = l.get_point2().get_y()
      if l.get_point2().get_y() > max_y:
        max_y = l.get_point2().get_y()
  region = [min_x, max_x, min_y, max_y]
  #vertices = [(min_x,min_y), (min_x,max_y), (max_x,min_y), (max_x,max_y)]
  return region


def write_users_per_project(locations, filename, write_to_file = False):

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
    with open('users_per_image.csv','w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Number of contributing users x", "Number of projects with x many users"])
      for num in user_count:
        writer.writerow([num, user_count.get(num)])


  return user_count


def get_time_per_project(locations, filename, write_to_file = True):
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
    with open('time_per_project.csv','w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Pic ID", "Start Time", "End Time", "Duration"])
      for pic_id in times_count:
        writer.writerow([pic_id, times_count.get(pic_id)[0], times_count.get(pic_id)[1], times_count.get(pic_id)[2]])


  return times_count

def write_updates_per_project(locations,filename,write_to_file):
  # Find the number of updates for every project
  # project_updates is a dictionary with the project ID as key and number of updates as value
  project_updates = dict()
  for pic_id in locations:
    project_updates[pic_id] = 0
    path = locations.get(pic_id)

    with open(filename,'r') as file:
      next(file, None)
      reader = csv.reader(file)
      
      for r in reader:
        x = float(r[2])
        y = float(r[3])
        if ( path.pixel_is_in_image(Point(x,y))):
          project_updates[pic_id] += 1

  if write_to_file:
    with open('updates_per_project.csv','w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Project ID", "Number of updates"])
      for proj in project_updates:
        writer.writerow([proj, project_updates.get(proj)])
  return project_updates

def write_projects_per_user(locations,filename,write_to_file):
  # Find all the projects that each user has contributed to 
  # user_projects is a dictionary with the user ID as key and list of projects that a particular user has contributed to as value
  user_projects = dict()
  with open(filename,'r') as file:
    next(file, None)
    reader = csv.reader(file)
      
    for r in reader:
      user = r[1]
      x = float(r[2])
      y = float(r[3])
      for pic_id in locations:
        path = locations.get(pic_id)
        if ( path.pixel_is_in_image(Point(x,y))):
          if user in user_projects:
            if pic_id not in user_projects[user]:
              user_projects[user].add(pic_id)
          else:
            user_projects[user] = set()
            user_projects[user].add(pic_id)

  if write_to_file:
    with open('prjects_per_user.csv','w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["user ID", "list of projects", "number of projects"])
      for u in user_projects:
        writer.writerow([u, user_projects.get(u), len(user_projects.get(u))])
  return user_projects

def write_pixels_per_project(locations,write_to_file):
  # Find the number of pixels in each image
  # project_pixels is a dictionary with the project ID as key and number of pixels in that project as value
  project_pixels = dict()
  region = get_canvas_region(locations)
  min_x = int(region[0])
  max_x = int(region[1]+0.5)
  min_y = int(region[2])
  max_y = int(region[3]+0.5)
  for pic_id in locations:
    path = locations.get(pic_id)
    project_pixels[pic_id] = 0
    for x in [float(j) / 2 for j in range(min_x, max_x, 1)]:
      for y in [float(i) / 2 for i in range(min_y, max_y, 1)]:
        if ( path.pixel_is_in_image(Point(x,y))):
          project_pixels[pic_id] += 1
  if write_to_file:
    with open('pixels_per_project.csv','w') as fileOut:
      writer = csv.writer(fileOut, delimiter = ",")
      writer.writerow(["Project", "Number of pixels"])
      for proj in project_pixels:
        writer.writerow([proj, project_pixels.get(proj)])
  return project_pixels



if __name__ == "__main__":
  locations = store_locations()
  print(locations)
  print(get_canvas_region(locations))
  write_users_per_project(locations, "tile_placements_sub.csv", True)
  get_time_per_project(locations, "tile_placements_sub.csv", True)
  write_updates_per_project(locations,"tile_placements_sub.csv", True)
  write_projects_per_user(locations,"tile_placements_sub.csv", True)
  write_pixels_per_project(locations, True)






        
        
