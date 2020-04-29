from reddit import *
import path
import point
import line
import csv


def get_list_of_removed_proj(output_filename, writeto_file=False):
    '''
        Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color), return projects on the final canvas that overlap each other by at least 80% and should be removed

        If writeto_file is True, then a text file with the projects will be produced.
        Otherwise output_filename is ignored.
    '''
    projects_to_remove = {'1738', '1616', '1849', '1824', '1240', '1516', '1289', '129', '720', '1915', '757', '867', '72', '1758', '1823', '1900', '426', '607', '1349', '1483', '1549', '1707', '1823', '1900', '1308', '520', '527', '574', '581', '1790', '651', '680', '697', '704', '939', '955', '1092', '1105', '1155', '1159', '1165', '1166', '1168', '1174', '1180', '1181', '1200', '1202', '1263', '1264', '1265', '1272', '1274', '1275', '1282', '1284', '1286', '1302', '1319', '1329', '1378', '1380', '1382', '1395', '1396', '1412',
                          '1418', '1423', '1426', '1428', '1446', '1482', '1493', '1494', '1502', '1508', '1509', '1512', '1524', '1529', '1540', '1548', '1550', '1573', '1575', '1595', '1624', '1635', '1637', '1638', '1640', '1641', '1643', '1649', '1674', '1682', '1684', '1702', '1707', '1714', '1716', '1720', '1721', '1735', '1747', '1749', '1756', '1761', '1763', '1768', '1788', '1795', '1798', '1800', '1811', '1812', '1818', '1820', '1825', '1832', '1834', '1844', '1853', '1865', '1887', '1913', '1989', '1999', '2007', '2017', '2025', '1926'}

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
                    writer.writerow(
                        [pic_id1, locations.get(pic_id1).get_name()])

            continue

        for pic_id2 in locations:
            if (pic_id1 != pic_id2):
                if (int(locations.get(pic_id1).left) > int(locations.get(pic_id2).right)):
                    continue

                if (int(locations.get(pic_id1).right) < int(locations.get(pic_id2).left)):
                    continue

                if (int(locations.get(pic_id1).bottom) > int(locations.get(pic_id2).top)):
                    continue

                if (int(locations.get(pic_id1).top) < int(locations.get(pic_id2).bottom)):
                    continue

                if project_pixels.get(pic_id1) == None:
                    pixels = set()
                    # Run through bounding box of pic1
                    for i in range(int(locations.get(pic_id1).left), int(locations.get(pic_id1).right) + 1):
                        for j in range(int(locations.get(pic_id1).bottom), int(locations.get(pic_id1).top) + 1):
                            if(locations.get(pic_id1).pixel_is_in_image(Point(i, j))):
                                pixels.add((i, j))

                    project_pixels[pic_id1] = pixels

                if project_pixels.get(pic_id2) == None:
                    pixels = set()
                    for i in range(int(locations.get(pic_id2).left), int(locations.get(pic_id2).right) + 1):
                        for j in range(int(locations.get(pic_id2).bottom), int(locations.get(pic_id2).top) + 1):

                            if(locations.get(pic_id2).pixel_is_in_image(Point(i, j))):
                                pixels.add((i, j))

                    project_pixels[pic_id2] = pixels

                pic1_pixel_count = len(project_pixels[pic_id1])
                pic2_pixel_count = len(project_pixels[pic_id2])
                overlapping_pixels = len(
                    project_pixels[pic_id1] & project_pixels[pic_id2])

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
                            print_rows.append([str(pic_id1) + " " + locations.get(pic_id1).get_name(), str(
                                pic_id2) + " " + locations.get(pic_id2).get_name()])

                        else:
                            projects_to_remove.add(pic_id1)
                            print_rows.append([str(pic_id2) + " " + locations.get(pic_id2).get_name(), str(
                                pic_id1) + " " + locations.get(pic_id1).get_name()])


    if writeto_file:
        with open(output_filename, 'a') as file:
            writer = csv.writer(file)

            for row in print_rows:
                writer.writerow(row)

            writer.writerow(["1012 Rutgers University",
                             "1763 University of Kentucky"])
            writer.writerow(projects_to_remove)

    return projects_to_remove


if __name__ == "__main__":
    filename = "../data/proj_to_remove.txt"
    set2 = get_list_of_removed_proj(filename, writeto_file=True)
