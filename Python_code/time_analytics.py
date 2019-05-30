import csv
import sys
import math
from reddit import *
from line import *
from point import *
from path import *
from analytics_combined import *

def time_analysis(filename):
    # locations = store_locations("atlas.json")
    locations = store_locations("atlas_complete.json")
    project_time = dict()
    with open(filename,'r') as file:
        # Skip first line (header row)
        next(file, None)
        
        reader = csv.reader(file)

        for r in reader:
            time = int(r[0])
   
            x = float(r[2])
            y = float(r[3])

            boundary_list = create_sorted_lists(locations)
            filter_lists(Point(x,y), boundary_list)
            for boundary in boundary_list:
                pic_id = boundary[0]
                path = locations.get(pic_id)
                if ( path.pixel_is_in_image(Point(x,y))):
                    if project_time.get(pic_id) is None:
                        project_time[pic_id] = {"oldest": time, "newest": time}
                    else:
                        if time < project_time.get(pic_id)["oldest"]:
                            project_time[pic_id]["oldest"] = time
                        if time > project_time.get(pic_id)["newest"]:
                            project_time[pic_id]["newest"] = time

    with open("times_per_project.csv",'w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["Pic ID", "Newest Time", "Oldest Time", "Duration"])
        for pic_id in project_time:
            newest_time = project_time.get(pic_id).get("newest")
            oldest_time = project_time.get(pic_id).get("oldest")
            duration = newest_time - oldest_time
            writer.writerow([pic_id, newest_time, oldest_time, duration])


time_analysis("final_canvas_tile_placements.csv")
