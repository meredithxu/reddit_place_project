import csv
import sys
import math
import numpy as np
from reddit import *
from line import *
from point import *
from path import *
from analytics_combined import *

def create_user_vectors(filename, locations):
    
    project_users_count = dict()
    with open(filename,'r') as file:
        # Skip first line (header row)
        next(file, None)
        
        reader = csv.reader(file)

        for r in reader:
            time = int(r[0])
            user = r[1]
            x = float(r[2])
            y = float(r[3])

            boundary_list = create_sorted_lists(locations)
            filter_lists(Point(x,y), boundary_list)
            for boundary in boundary_list:
                pic_id = boundary[0]
                path = locations.get(pic_id)
                if ( path.pixel_is_in_image(Point(x,y))):
                    if project_users_count.get(user) is None:
                        project_users_count[user] = dict()
                        project_users_count[user][pic_id] = 1
                    
                    elif project_users_count.get(user).get(pic_id) is None:
                        project_users_count[user][pic_id] = 1
                    else:
                        project_users_count[user][pic_id] += 1
                        
    with open("final_canvas_users.csv",'w') as fileOut:
        writer = csv.writer(fileOut, delimiter = ",")
        writer.writerow(["User", "Project ID", "# Updates"])
        for user in project_users_count:
            row = [user]
            for pic_id in project_users_count.get(user):
                
                row.append( str(pic_id) )
                row.append( str(project_users_count.get(user).get(pic_id)))
            writer.writerow(row)
    return project_users_count


if __name__ == "__main__":
    locations = store_locations("atlas.json")
    project_users_count = create_user_vectors("final_canvas_tile_placements.csv", locations)
    num_users = len(project_users_count)
    num_projects = len(locations)

    users = list(project_users_count.keys())
    projects = list(locations.keys())
    user_project_matrix = np.zeros((num_users, num_projects))

    for user in project_users_count:
        for pic_id in project_users_count.get(user):
            user_index = users.index(user)
            project_index = projects.index(pic_id)
            user_project_matrix[user_index][project_index] = project_users_count.get(user).get(pic_id)

    np.save("user_project_matrix.npy", user_project_matrix)
    np.save("users_array.npy", users)
    np.save("project_ids_array.npy", projects)