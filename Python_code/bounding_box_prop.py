from line import *
from point import *
from path import *
from reddit import *
import numpy as np

project_updates = dict()
projects_to_remove = {'777', '1921', '1169', '42', '1066', '1757', '1824', '320', '998', '1870', '1811',\
                     '1925', '1927', '704', '1085', '1308', '1378', '1412', '1418', '1428', '1455', '1482',\
                      '1512', '1548', '1589', '1614', '1790', '1319', '939', '1263', '1383', '1155', '1761', 
                     '1524', '351', '129', '1046', '1073', '1595', '1254', '1528', '1529', '1578', '1616',\
                     '1721'}
with open("../data/tile_placements_proj.csv",'r') as file:
# Skip first line (header row)
	next(file, None)

	reader = csv.reader(file)

	for r in reader:
		final_pixel = int(r[-2])
		pic_id = str(r[-3])

		if pic_id not in projects_to_remove:
			if final_pixel == 1:
				if pic_id not in project_updates:
					project_updates[pic_id] = 1

				else:
					project_updates[pic_id] += 1


print(len(project_updates))
locations = store_locations("../data/atlas_filtered.json")
area_prop = np.zeros(len(project_updates))
pic_id_list = np.zeros(len(project_updates))
count = 0
for pic_id in locations:
	if pic_id not in projects_to_remove:
		if pic_id in project_updates:

			path = locations.get(pic_id)

			top = path.get_top()
			bottom = path.get_bottom()
			left = path.get_left()
			right = path.get_right()

			area = (top-bottom) * (right-left)
			num_updates = project_updates[pic_id]

			pic_id_list[count] = pic_id
			area_prop[count] = num_updates/area
			count += 1

print(area_prop.size)
print(pic_id_list.size)
np.save('../data/area_prop', area_prop)
np.save('../data/pic_id_list', pic_id_list)
