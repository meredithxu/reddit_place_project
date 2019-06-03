from reddit import *
import path
import point
import line
import csv

def get_list_of_removed_proj(output_filename, writeto_file = False):
    '''
        Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color), return projects on the final canvas that overlap each other by at least 80% and should be removed

        If writeto_file is True, then a text file with the projects will be produced.
        Otherwise output_filename is ignored.

        There are also some projects that will confuse our CNN, so we maually remove them:
   
        1913: Homelab invasion, intersects 1616 twich logo
        1649: beaver, 1707: warriors
        1849: Faeria Yak (only has border)
        1319: very incomplete
        1824 (climber's head, too small)
        1383, 1493, 1823, 1818, 645, 1640 (Very small)
        1240, 1516 (1 pixel)
        1738: Auburn University, Includes extra border pixels but still overlaps 1705

    '''
    projects_to_remove = {'1738', '1763', '1913', '1616', '1649', '1707', '1849', '1319', '1824', '1240', '1516'}

    project_pixels = dict()
    print_rows = []

    locations = store_locations("../data/atlas_complete.json")
    # locations = store_locations("../data/test_atlas.json")
    if writeto_file:
        with open(output_filename, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["Kept Projects", "Removed Projects"])

    for pic_id1 in locations:
        if "(covered)" in locations.get(pic_id1).get_name().lower() or "(former)" in locations.get(pic_id1).get_name().lower():
            projects_to_remove.add(pic_id1)
            
            if writeto_file:
                with open(output_filename, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([pic_id1, locations.get(pic_id1).get_name()])

            continue

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
                
                # Remove the projects that cover the entire canvas
                if pic1_pixel_count > 9000 or pic2_pixel_count > 9000:

                    if pic1_pixel_count > 9000:
                        if pic_id1 not in projects_to_remove:
                            projects_to_remove.add(pic_id1)

                    if pic2_pixel_count > 9000:
                        if pic_id2 not in projects_to_remove:
                            projects_to_remove.add(pic_id2)

                    continue

                if pic1_pixel_count > 0:
                    overlapping_area1 = overlapping_pixels / pic1_pixel_count

                if pic2_pixel_count > 0:
                    overlapping_area2 = overlapping_pixels / pic2_pixel_count

               
                if (overlapping_area1 >= 0.8 and overlapping_area2 >= 0.8):
                    if pic_id1 not in projects_to_remove and pic_id2 not in projects_to_remove:
                        

                        # # manually remove 1763 University of Kentucky and keep 1012 Rutgers University
                        # if pic_id1 == '1763' or pic_id2 == '1763':
                        #     projects_to_remove.add('1763')
                        #     continue

                        # Keep the larger one    
                        if pic1_pixel_count >= pic2_pixel_count:
                            projects_to_remove.add(pic_id2)
                            print_rows.append([str(pic_id1) + " " + locations.get(pic_id1).get_name(), str(pic_id2) + " " + locations.get(pic_id2).get_name() ] )

                        else:
                            projects_to_remove.add(pic_id1)
                            print_rows.append([ str(pic_id2) + " " + locations.get(pic_id2).get_name(), str(pic_id1) + " " + locations.get(pic_id1).get_name()  ])



    if writeto_file:
        with open(output_filename, 'a') as file:
            writer = csv.writer(file)
            
            for row in print_rows:
                writer.writerow(row)

            writer.writerow(["1012 Rutgers University", "1763 University of Kentucky"])
            writer.writerow(projects_to_remove)
            
    return projects_to_remove

if __name__ == "__main__":
    filename = "../data/proj_to_remove.txt"
    set2 = get_list_of_removed_proj(filename, writeto_file = True)    

    