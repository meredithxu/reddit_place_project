import csv
import sys
import math
import numpy as np
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
  """
     Filters candidate pic_ids for a pixel based on boundaries of the project (bottom, top, left, right)
  """
  filtered_lists = []

  for b in boundary_list:
    # Discard all figures whose top boundary does not include test_point
    if test_point.get_y() <= b[1]:
      # Discard all figures whose left boundary does not include test_point
      if test_point.get_x() >= b[2]:
        # Discard all figures whose right boundary does not include test_point
        if test_point.get_x() <= b[3]:
          # Discard all figures whose bottom boundary does not include test_point
          if test_point.get_y() >= b[4]:
            filtered_lists.append(b)

  return filtered_lists

def pixel_assignments(locations):
  """
  	Creates matrix with pic_id associated with each pixel (shifted by 1, pic_id 0 is unknown/null)
  """
  boundary_list = create_sorted_lists(locations)
  pixel_assign = {}

  for y in range(1000):
    for x in range(1000):
      point = Point(x+.5,y+.5)
      filtered_list = filter_lists(point, boundary_list)
      
      for boundary in filtered_list:
        pic_id = boundary[0]
        path = locations.get(pic_id)
        if path.pixel_is_in_image(Point(x+.5,y+.5)):
          if (x,y) not in pixel_assign:
            pixel_assign[(x,y)] = [pic_id]
          else:
            pixel_assign[(x,y)].append(pic_id)
	
  return pixel_assign

def final_update_time_and_color(filename):
  """
  	Creates matrices with final update time and final color for each pixel in the canvas
  """
  final_time = np.uint64(np.zeros((1001,1001)))
  final_color = np.uint8(np.zeros((1001,1001)))

  with open(filename,'r') as file:
    # Skip first line (header row)
    next(file, None)

    reader = csv.reader(file)

    for r in reader:
      time = int(r[0])
      x = int(r[2])
      y = int(r[3])
      color = int(r[4])

      if final_time[y][x] < time:
        final_time[y][x] = time
        final_color[y][x] = color 

  return final_time, final_color

def add_atlas_data_to_tile_placements(locations, input_filename, output_filename):
  """
    Takes the tile_placements and atlas data and creates a csv where each line is a tuple of the following elements:
    time, user, x, y, color, project_id, pixel, pixel_color.
  """  
  #Creates matrix with pic_id associated with each pixel
  proj_per_pixel = pixel_assignments(locations)

  #Creates matrices with final update time and final color for each pixel in the canvas
  final_up_time, final_up_color = final_update_time_and_color(input_filename)
    
  with open(input_filename,'r') as file_in:
    with open(output_filename, 'w') as file_out:
      writer = csv.writer(file_out, delimiter = ",")
      writer.writerow(["ts", "user" ,"x_coordinate" ,"y_coordinate" ,"color", "pic_id", "pixel", "pixel_color"])
      
      # Skip first line (header row)
      next(file_in, None)

      reader = csv.reader(file_in)

      for r in reader:
        time = int(r[0])
        user = r[1]
        x = int(r[2])
        y = int(r[3])
        color = int(r[4])

        
        if (x,y) in proj_per_pixel:
          pic_ids = proj_per_pixel[(x,y)]

          if len(pic_ids) > 0:
            if time >= final_up_time[y][x]:
              pixel = 1
              final_up_time[y][x] = sys.maxsize
            else:
              pixel = 0

            if color == final_up_color[y][x]:
              pixel_color = 1
            else:
              pixel_color = 0

            for pic_id in pic_ids:
              writer.writerow([time, user, x, y, color, pic_id, pixel, pixel_color])

if __name__ == "__main__":
    locations = store_locations("../data/atlas_filtered.json")
    
    #add_atlas_data_to_tile_placements(locations, "../data/sorted_tile_placements.csv", "../data/sorted_tile_placements_proj.csv")
    #add_atlas_data_to_tile_placements(locations, "../data/sorted_tile_placements_denoised_freq.csv",\
    #	"../data/tile_placements_denoised_freq_proj.csv")
    add_atlas_data_to_tile_placements(locations, "../data/sorted_tile_placements_denoised_users.csv",\
    	"../data/tile_placements_denoised_users_proj.csv")

