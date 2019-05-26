from reddit import *
import path
import point
import line
import csv

def get_list_of_overlapping_proj(output_filename):
    '''
        Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color), return projects on the final canvas that overlap each other by at least 90%.
    '''

    projects_to_remove = set()
    project_pixels = dict()

    locations = store_locations("../data/atlas_filtered.json")
    # locations = store_locations("../data/test_atlas.json")

    for pic_id1 in locations:
        for pic_id2 in locations:
            if (pic_id1 != pic_id2):


                if ( int(locations.get(pic_id1).left) > int(locations.get(pic_id2).right) ):
                    continue

                if ( int(locations.get(pic_id1).right) < int(locations.get(pic_id2).left) ):
                    continue

                if ( int(locations.get(pic_id1).bottom) > int(locations.get(pic_id2).top) ):
                    continue

                if ( int(locations.get(pic_id1).top) < int(locations.get(pic_id2).bottom) ):
                    continue


                if project_pixels.get(pic_id1) == None:
                    pixels = set()
                    # Run through bounding box of pic1
                    for i in range( int(locations.get(pic_id1).left), int(locations.get(pic_id1).right) + 1  ):
                        for j in range( int(locations.get(pic_id1).bottom), int(locations.get(pic_id1).top) + 1 ):
                            if( locations.get(pic_id1).pixel_is_in_image( Point(i, j) ) ):
                                pixels.add((i, j))

                    project_pixels[pic_id1] = pixels

                if project_pixels.get(pic_id2) == None:
                    pixels = set()
                    for i in range( int(locations.get(pic_id2).left), int(locations.get(pic_id2).right) + 1  ):
                        for j in range( int(locations.get(pic_id2).bottom), int(locations.get(pic_id2).top) + 1 ):
                        
                            if( locations.get(pic_id2).pixel_is_in_image( Point(i, j) ) ):
                                pixels.add((i,j))

                    project_pixels[pic_id2] = pixels

                pic1_pixel_count = len(project_pixels[pic_id1])     
                pic2_pixel_count = len(project_pixels[pic_id2])  
                overlapping_pixels =  len(project_pixels[pic_id1] & project_pixels[pic_id2])

                if pic1_pixel_count > 0:
                    overlapping_area1 = overlapping_pixels / pic1_pixel_count

                if pic2_pixel_count > 0:
                    overlapping_area2 = overlapping_pixels / pic2_pixel_count

                if (overlapping_area1 >= 0.9 and overlapping_area2 >= 0.9):
                    if pic_id1 not in projects_to_remove and pic_id2 not in projects_to_remove:
                        projects_to_remove.add(pic_id2)

    with open(output_filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(projects_to_remove)

    return projects_to_remove

if __name__ == "__main__":
    get_list_of_overlapping_proj("../data/proj_to_remove.txt")