import csv
from reddit import *
from line import *
from point import *
from path import *

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



if __name__ == "__main__":
  locations = store_locations()
  write_users_per_project(locations, "tile_placements_sub.csv", True)






        
        
