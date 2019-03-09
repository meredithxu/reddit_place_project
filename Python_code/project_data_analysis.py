import csv
import numpy as np
import scipy

def updates_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of updates per project (dictionary). Some projects might be removed from the analysis. 
		It also returns the total number of updates.
	'''
	updates_per_proj = dict()
	total_updates = 0

	with open(input_file_proj,'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)
		
		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			
			if proj not in projects_to_remove:
				if proj in updates_per_proj:
					updates_per_proj[proj] = updates_per_proj[proj] + 1
				else:
					updates_per_proj[proj] = 1
				total_updates = total_updates + 1

	return updates_per_proj, total_updates


def colors_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of colors per project (dictionary). Some projects might be removed from the analysis.
		For each project, the output contains how often a given color (out of 16 possible in the canvas) was used.
		Important: Only pixels (final canvas) are considered here.
	'''
	colors_per_proj = {}
	num_colors = 16

	with open(input_file_proj,'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)

		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			color = int(r[4])
			pixel = int(r[6])

			if proj not in projects_to_remove and pixel == 1:
				if proj in colors_per_proj:
					colors_per_proj[proj][color] = colors_per_proj[proj][color] + 1.
				else:
					colors_per_proj[proj] = np.zeros(num_colors)
					colors_per_proj[proj][color] = 1.

	return colors_per_proj

def entropy_per_project(colors_per_project):
	'''
		Given a dict with colors per project, computes the entropy of each project (dictionary).
	'''
	entropy_per_proj = {}
	num_colors = 16

	for proj in colors_per_project:
		ent = 0.
    
		colors_per_project[proj] = colors_per_project[proj] / np.sum(colors_per_project[proj])
    
		for c in range(num_colors):
			p = colors_per_project[proj][c]
			
			if p > 0:
				ent = ent - p * np.log(p)
            
		entropy_per_proj[proj] = ent

	return entropy_per_proj

def pixels_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of pixels per project. Some projects might be removed from the analysis.
	'''
	pixels_per_proj = {}

	with open(input_file_proj,'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)

		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			pixel = int(r[6])

			if proj not in projects_to_remove and pixel == 1:
				if proj in pixels_per_proj:
					pixels_per_proj[proj] = pixels_per_proj[proj] + 1
				else:
					pixels_per_proj[proj] = 1

	return pixels_per_proj

def projects_per_user(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of projects per user. Some projects might be removed from the analysis.
		Important: Only updates that agree with the final color of the tile in the canvas are considered.
	'''
	proj_per_user_lst = {}
	
	with open(input_file_proj, 'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)
		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			user = r[1]

			pixel_color = int(r[7])

			if proj not in projects_to_remove and pixel_color == 1:
				if user in proj_per_user_lst:
					if proj not in proj_per_user_lst[user]:
						proj_per_user_lst[user][proj] = 1
				else:   
					proj_per_user_lst[user] = {}
					proj_per_user_lst[user][proj] = 1

	proj_per_user = {}

	for user in proj_per_user_lst:
		proj_per_user[user] = len(proj_per_user_lst[user])

	return proj_per_user

def users_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of users per project. Some projects might be removed from the analysis.
		Important: Only updates that agree with the final color of the tile in the canvas are considered.
	'''
	users_per_proj_lst = {}

	with open(input_file_proj, 'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)

		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			user = r[1]
                         
			pixel_color = int(r[7])

			if proj not in projects_to_remove and pixel_color == 1:
				if proj in users_per_proj_lst:
					if user not in users_per_proj_lst[proj]:
						users_per_proj_lst[proj][user] = 1
				else:   
					users_per_proj_lst[proj] = {}
					users_per_proj_lst[proj][user] = 1

	users_per_proj = {}

	for proj in users_per_proj_lst:
		users_per_proj[proj] = len(users_per_proj_lst[proj])
	
	return users_per_proj


def times_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the total time (in miliseconds) per project. Some projects might be removed from the analysis.
		Important: Only updates that agree with the final color of the tile in the canvas are considered.
	'''
	min_times_per_proj = {}
	max_times_per_proj = {}

	with open(input_file_proj, 'r') as file:
		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)
    
		for r in reader:
			#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
			proj = r[5]
			t = int(r[0])
 
			pixel_color = int(r[7])

			if proj not in projects_to_remove and pixel_color == 1:
				if proj in min_times_per_proj:
					if min_times_per_proj[proj] > t:
						min_times_per_proj[proj] = t
                
					if max_times_per_proj[proj] < t:
						max_times_per_proj[proj] = t
				else:
					min_times_per_proj[proj] = t
					max_times_per_proj[proj] = t
                
	times_per_proj = {}

	for proj in min_times_per_proj:
		times_per_proj[proj] = max_times_per_proj[proj] - min_times_per_proj[proj]

	return times_per_proj

def icdf(dct):
	'''
		Simple function to compute an inverse cumulative density distribution from 
		a dictionary. 
	'''
	count = np.zeros(max(dct.values())+1)

	for k in dct:
		count[dct[k]] = count[dct[k]] + 1
    
	for i in reversed(range(len(count)-1)):
		count[i] = count[i] + count[i+1]
    
	count = count / count[0]

	return count

def user_project_matrix(input_file_proj, projects_to_remove, match_pixel, match_pixel_color):
	"""
		Builds different types of user matrix depending on the parameters match_pixel
		and match pixel color. If they are True, such fields are considered as a filter when
		filling the matrix.
	"""
	users_dict = dict()
	projects_dict = dict()

	projects_count = 0
	users_count = 0

	with open(input_file_proj,'r') as file:
    		# Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)
		#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
		for r in reader:
			user = r[1]
			proj = r[5]
			pixel = int(r[6])
			pixel_color = int(r[7])
        
			if proj not in projects_to_remove:
				if (not match_pixel or pixel == 1) and (not match_pixel_color or pixel_color == 1):
					if user not in users_dict:
						users_dict[user] = users_count
						users_count += 1
            
					if proj not in projects_dict:
						projects_dict[proj] = projects_count
						projects_count += 1

	user_proj_matrix = scipy.sparse.lil_matrix((users_count, projects_count))
    
	with open("../data/sorted_tile_placements_proj.csv", "r") as file:
		reader = csv.reader(file, delimiter = ",")
		# Skip first line (header row)
		next(file, None)
		#ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color
		for r in reader:
			user = r[1]
			proj = r[5]
			pixel = int(r[6])
			pixel_color = int(r[7])
            
			if proj not in projects_to_remove:
				if (not match_pixel or pixel == 1) and (not match_pixel_color or pixel_color == 1):
                			user_proj_matrix[users_dict[user],projects_dict[proj]] += 1

	return user_proj_matrix 

def user_project_matrix_all(input_file_proj, projects_to_remove):
	"""
		Builds (non-normalized) user-project matrix with the number of updates per user within project regions.
	"""
	return user_project_matrix(input_file_proj, projects_to_remove, False, False)

def user_project_matrix_pixel(input_file_proj, projects_to_remove):
	"""
		Builds (non-normalized) user-project matrix with the number of pixels per user in a project.
	"""
	return user_project_matrix(input_file_proj, projects_to_remove, True, True)

def user_project_matrix_pixel_color(input_file_proj, projects_to_remove):
	"""
		Builds (non-normalized) user-project matrix with the number of updates that match the final
		pixels per user in a project.
	"""
	return user_project_matrix(input_file_proj, projects_to_remove, False, True)
