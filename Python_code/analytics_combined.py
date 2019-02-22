import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *

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




def add_atlas_data_to_tile_placements(locations, filename, output_filename):
    """
        Takes the tile_placements and atlas data and creates a csv where each line is a tuple of the following elements:

        time, user, x, y, color, project_id, project name, final
        
        If a pixel is a part of two different projects, it will be listed multiple times. 
    """
    data = dict()
    latest_times = dict()

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

                if ( path.pixel_is_in_image(Point(x,y))):
                    # Add the point
                    data[pic_id] = {
                        "time": time,
                        "user": user,
                        "x": x,
                        "y": y,
                        "color": color,
                        "pic_id": pic_id,
                        "name": path.get_name(),
                        "final": 0
                    }
                    
                    if latest_times.get((x,y,pic_id)) is None or latest_times[(x,y,pic_id)] < time:
                        latest_times[(x,y,pic_id)] = time
                        data[pic_id]["final"] = 1


    with open(output_filename, 'w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["ts", "user" ,"x_coordinate" ,"y_coordinate" ,"color", "pic_id", "name", "final"])
        for item in data:
            writer.writerow([data.get(item).get("time"), data.get(item).get("user"), data.get(item).get("x"), data.get(item).get("y"), data.get(item).get("color"), data.get(item).get("pic_id"), data.get(item).get("name"), data.get(item).get("final") ])


if __name__ == "__main__":
    locations = store_locations("../data/atlas_filtered.json")
    add_atlas_data_to_tile_placements(locations, "tile_placements.csv", "tile_placements_2.csv")
  