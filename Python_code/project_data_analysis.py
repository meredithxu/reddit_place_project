import csv
import numpy as np
import scipy
import random
import numpy.linalg as npla
import scipy
import sklearn
from sklearn import preprocessing
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla
from scipy.spatial import distance
from reddit import *


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


#def entropy_update_per_project(updates_per_proj):
	'''
		Given a dict with updates per project, computes the entropy of each project (dictionary).
	'''

#	for proj in updates_per_proj:
#		ent = 0
#		updates_per_proj[proj] = updates_per_proj[proj] / np.sum(np.array(list(updates_per_proj.values())))
#		p = updates_per_proj[proj]
#		if p > 0:
#			ent = ent - p * np.log(p)

#	return ent


def colors_per_project(input_file_proj, projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the number of colors per project (dictionary). Some projects might be removed from the analysis.
		For each project, the output contains how often a given color (out of 16 possible in the canvas) was used.
		Important: Only pixels (final canvas) are considered here.
	'''
	colors_per_proj = {}
	num_colors = 18

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


def update_category_per_project():
#Parse data to separate update types for each project
# Computing the different types of updates for each project
	min_time = 1490918688000
	max_time = 1491238734000
	n_vals = int((max_time-min_time) / (1000 * 60 * 60))
	tile_updates = dict()
	total_tile_updates={"final_updates":np.zeros(n_vals+1),"agreeing_updates":np.zeros(n_vals+1),"disagreeing_updates":np.zeros(n_vals+1)}

	with open("../data/sorted_tile_placements_proj.csv",'r') as file:
#Skip first line (header row)
		next(file, None)
		reader = csv.reader(file)
		for r in reader:
			picID = int(r[5])
			ts = int(r[0])        
			t = int((ts-min_time) / (1000 * 60 * 60))
			final_pixel = int(r[6])
			final_pixel_color = int(r[7])
			if picID not in tile_updates:
				tile_updates[picID] = {"final_updates":np.zeros(n_vals+1), 
                                   "agreeing_updates":np.zeros(n_vals+1), 
                                   "disagreeing_updates":np.zeros(n_vals+1)}
			if final_pixel == 1:
				tile_updates[picID]["final_updates"][t] = tile_updates[picID]["final_updates"][t] + 1
				total_tile_updates["final_updates"][t] = total_tile_updates["final_updates"][t] + 1
			if final_pixel_color == 1:
				tile_updates[picID]["agreeing_updates"][t] = tile_updates[picID]["agreeing_updates"][t] + 1
				total_tile_updates["agreeing_updates"][t] = total_tile_updates["agreeing_updates"][t] + 1
			else:
				tile_updates[picID]["disagreeing_updates"][t] = tile_updates[picID]["disagreeing_updates"][t] + 1     
				total_tile_updates["disagreeing_updates"][t] = total_tile_updates["disagreeing_updates"][t] + 1

	for picID in tile_updates:
		s = np.sum(tile_updates[picID]["final_updates"]) + np.sum(tile_updates[picID]["agreeing_updates"])\
        + np.sum(tile_updates[picID]["disagreeing_updates"])
		tile_updates[picID]["final_updates"] = tile_updates[picID]["final_updates"] / s
    
		tile_updates[picID]["agreeing_updates"] = tile_updates[picID]["agreeing_updates"] / s
    
		tile_updates[picID]["disagreeing_updates"] = tile_updates[picID]["disagreeing_updates"] / s
    
	s = np.sum(total_tile_updates["final_updates"]) + np.sum(total_tile_updates["agreeing_updates"])\
        + np.sum(total_tile_updates["disagreeing_updates"])
	total_tile_updates["final_updates"] = total_tile_updates["final_updates"] / s
	total_tile_updates["agreeing_updates"] = total_tile_updates["agreeing_updates"] / s
	total_tile_updates["disagreeing_updates"] = total_tile_updates["disagreeing_updates"] / s
    
	return tile_updates, total_tile_updates



def update_time_entropy_per_project(tile_updates):
	'''
		Given a dict with colors per project, computes the entropy of each project (dictionary).
	'''
	update_time_entropy_per_proj = {}


	for proj in tile_updates:
		ent = 0.
		agree_up = np.sum(tile_updates[proj]["agreeing_updates"])
		disagree_up = np.sum(tile_updates[proj]["disagreeing_updates"])
		sum_up= agree_up+ disagree_up
        
		for t in range(0,len(list(tile_updates[proj]["agreeing_updates"]))):
            
			p= tile_updates[proj]["agreeing_updates"][t]/sum_up
			if p > 0:
				ent = ent - p * np.log(p)
                
			q= tile_updates[proj]["disagreeing_updates"][t]/sum_up
			if q > 0:
				ent = ent - q * np.log(q)

            
		update_time_entropy_per_proj[proj] = ent

	return update_time_entropy_per_proj





def update_entropy_per_project(tile_updates):
	'''
		Given a dict with updates per project, computes the update entropy of each project (dictionary).
	'''
	update_entropy_per_proj = {}

	for proj in tile_updates:
		ent = 0.
		agree_up = np.sum(tile_updates[proj]["agreeing_updates"])
		disagree_up = np.sum(tile_updates[proj]["disagreeing_updates"])
		sum_up= agree_up+ disagree_up      
		p1 = agree_up/sum_up
		p2 = disagree_up/sum_up
		ent= -p1* np.log(p1)
		ent=ent - p2* np.log(p2)
            
		update_entropy_per_proj[proj] = ent

	return update_entropy_per_proj









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


def projects_per_user_list(input_file_proj, projects_to_remove):
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
						proj_per_user_lst[user][proj] = proj_per_user_lst[user][proj]+1                        
				else:   
					proj_per_user_lst[user] = {}
					proj_per_user_lst[user][proj] = 1

	proj_per_user = {}

	for user in proj_per_user_lst:
		proj_per_user[user] = len(proj_per_user_lst[user])

	return proj_per_user_lst





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

def users_per_project_list(input_file_proj, projects_to_remove):
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
						users_per_proj_lst[proj][user] = users_per_proj_lst[proj][user]+1                        
				else:   
					users_per_proj_lst[proj] = {}
					users_per_proj_lst[proj][user] = 1

	users_per_proj = {}

	for proj in users_per_proj_lst:
		users_per_proj[proj] = len(users_per_proj_lst[proj])
	
	return users_per_proj_lst





def area_per_project(input_file,projects_to_remove):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the area per project. Some projects might be removed from the analysis.
		Important: The area is computed as a fraction of the area covered in a bounding rectangle.
	'''
	project_updates = dict()
    
	with open(input_file,'r') as file:
# Skip first line (header row)
		next(file, None)

		reader = csv.reader(file)
		left_most={}
		right_most={}
		top_most={}
		bottom_most={}
        
		for r in reader:
			final_pixel = int(r[-2])
			pic_id = str(r[-3])
			x = int(r[2])
			y = int(r[3]) 
            
			if pic_id not in projects_to_remove:
				if final_pixel == 1:
                    
					if pic_id not in project_updates:
						project_updates[pic_id] = 1
						left_most[pic_id]=x
						right_most[pic_id]=x
						top_most[pic_id]=y
						bottom_most[pic_id]=y  
                        
					else:
						project_updates[pic_id] += 1
						if (x<=left_most[pic_id]):
							left_most[pic_id]=x
						if (x>right_most[pic_id]):
							right_most[pic_id]=x
						if (y<=bottom_most[pic_id]):
							bottom_most[pic_id]=y
						if (y>top_most[pic_id]):
							top_most[pic_id]=y                            
                       

	#print(len(project_updates))
	locations = store_locations("../data/atlas.json")
	area_prop={}

	for pic_id in locations:
		if pic_id not in projects_to_remove:
			if pic_id in project_updates:
                    
				area = (top_most[pic_id]-bottom_most[pic_id]+1) * (right_most[pic_id]-left_most[pic_id]+1)
				num_updates = project_updates[pic_id]

				area_prop[pic_id]=num_updates/area #pic_id_list[count] = pic_id
				if (area_prop[pic_id]>1):
					print("project "+pic_id)
	return area_prop






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


#IDs = []
#for p in updates_per_proj.keys():
    #IDs.append(names[int(p)])
    #IDs.append(str(p))
        
#IDs = np.array(list(IDs), dtype=object)
def Ratio(X,Y,names):
	'''
		Simple function to compute ratio of values between two dictionaries Y and X (Y/X) and returns another dictionary. 
	'''
	ratios= {}
	IDs = []
	for p in X.keys():
		xvalue = X[p]
		yvalue = Y[p]
		ratios[p] = yvalue / xvalue
		IDs.append(names[int(p)])
        #IDs.append(str(p))
	IDs = np.array(list(IDs), dtype=object)
	return ratios,IDs
   
def Create_Array(Dict_1,Dict_2):
	'''
		Simple function to create arrays from two dictionaries with same project keys. 
	'''
	X = np.zeros(len(Dict_1))
	Y = np.zeros(len(Dict_2))
	i = 0
	for p in Dict_1.keys():
		X[i] = Dict_1[p]
		Y[i] = Dict_2[p] 
		i = i + 1   
	return X,Y   
    
    
    
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

	return user_proj_matrix, users_dict, projects_dict 

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


def distance_per_project(input_file_proj, projects_to_remove,sample_size):  
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the average distance of (a set of sampled) users per project. Some projects might be removed from the analysis.
		Important: Only updates that agree with the final color of the tile in the canvas are considered.
	'''
	user_project_mat,users_dict,projects_dict = user_project_matrix_pixel_color(input_file_proj, projects_to_remove)

	proj_euc_dist = dict()  # Average euclidean distance between sampled users
	proj_cos_dist = dict()  # Average cosine distance between sampled users
    
	len_v=len(projects_dict.keys())
    
    
	#random pairs 
    
	proj_rand_dist = [0,0]
	cos_rand=0.0
	euc_rand=0.0
	for i in range(0,sample_size):
		rand_s=random.randint(0,len(users_dict.keys())-1)
		rand_t=rand_s
		while (rand_t==rand_s):
			rand_t= random.randint(0,len(users_dict.keys())-1)
            
		vec1 = user_project_mat[rand_s, : ].toarray()
		vec1=vec1.reshape(1,len_v)
		vec1=preprocessing.normalize(vec1, norm='l2')
		vec2 = user_project_mat[rand_t, :].toarray()
		vec2=vec2.reshape(1,len_v)
		vec2=preprocessing.normalize(vec2, norm='l2')
        
		euc_rand = euc_rand+distance.euclidean( vec1 , vec2 )
		cos_rand = cos_rand+distance.cosine( vec1 , vec2 ) 
	proj_rand_dist[0]=euc_rand/(sample_size)
	proj_rand_dist[1]=cos_rand/(sample_size)

	users_per_proj=users_per_project_list(input_file_proj, projects_to_remove)

	for pic_id in users_per_proj:
		all_users = list(users_per_proj[pic_id]) # make a list

		euc_dis=0.0
		cos_dis=0.0       
		for i in range(0,sample_size):
			rand_s=random.randint(0,len(all_users)-1)
			rand_t=rand_s
			while (rand_t==rand_s):
				rand_t= random.randint(0,len(all_users)-1)
			user1 = all_users[rand_s]
			user2 = all_users[rand_t]

			vec1 = user_project_mat[users_dict[user1], : ].toarray()
			vec1=vec1.reshape(1,len_v)
			vec1=preprocessing.normalize(vec1, norm='l2')
            
			vec2 = user_project_mat[users_dict[user2], :].toarray()
			vec2=vec2.reshape(1,len_v)
			vec2=preprocessing.normalize(vec2, norm='l2')
            
			euc_dis = euc_dis+distance.euclidean( vec1 , vec2 )
			cos_dis = cos_dis+distance.cosine( vec1 , vec2 )
            
		proj_euc_dist[pic_id]=euc_dis/sample_size
		proj_cos_dist[pic_id]=cos_dis/sample_size
  
	return proj_euc_dist, proj_cos_dist,proj_rand_dist


  

def distance_per_project_all(input_file_proj, projects_to_remove,sample_size):
	'''
		Given input file with project assignments (ts,user,x_coordinate,y_coordinate,color,pic_id,pixel,pixel_color)
		computes the distance of (a set of sampled) users inside projects. Some projects might be removed from the analysis.
		Important: Only updates that agree with the final color of the tile in the canvas are considered.
	'''

	user_project_mat,users_dict,projects_dict = user_project_matrix_pixel_color(input_file_proj, projects_to_remove)

	proj_euc_dist = dict()  # Average euclidean distance between sampled users
	proj_cos_dist = dict()  # Average cosine distance between sampled users
    
	len_v=len(projects_dict.keys())
    
    
	#random pairs 
    
	proj_rand_dist = [0,0]
	cos_rand=0.0
	euc_rand=0.0
	temp_list_euc=[]
	temp_list_cos=[]
	for i in range(0,sample_size):
		rand_s=random.randint(0,len(users_dict.keys())-1)
		rand_t=rand_s
		while (rand_t==rand_s):
			rand_t= random.randint(0,len(users_dict.keys())-1)
            
		vec1 = user_project_mat[rand_s, : ].toarray()
		vec1=vec1.reshape(1,len_v)
		vec1=preprocessing.normalize(vec1, norm='l2')
		vec2 = user_project_mat[rand_t, :].toarray()
		vec2=vec2.reshape(1,len_v)
		vec2=preprocessing.normalize(vec2, norm='l2')
        
		euc_dis = distance.euclidean( vec1 , vec2 )
		cos_dis = distance.cosine( vec1 , vec2 )
		temp_list_euc.append(euc_dis)
		temp_list_cos.append(cos_dis) 
	proj_rand_dist[0]=temp_list_euc
	proj_rand_dist[1]=temp_list_cos

	users_per_proj=users_per_project_list(input_file_proj, projects_to_remove)

	big_euc_dict=dict()
	big_cos_dict=dict()
	for pic_id in users_per_proj:
		all_users = list(users_per_proj[pic_id]) # make a list
		temp_list_euc=[]
		temp_list_cos=[]
		euc_dis=0.0
		cos_dis=0.0       
		for i in range(0,sample_size):
			rand_s=random.randint(0,len(all_users)-1)
			rand_t=rand_s
			while (rand_t==rand_s):
				rand_t= random.randint(0,len(all_users)-1)
			user1 = all_users[rand_s]
			user2 = all_users[rand_t]

			vec1 = user_project_mat[users_dict[user1], : ].toarray()
			vec1=vec1.reshape(1,len_v)
			vec1=preprocessing.normalize(vec1, norm='l2')
            
			vec2 = user_project_mat[users_dict[user2], :].toarray()
			vec2=vec2.reshape(1,len_v)
			vec2=preprocessing.normalize(vec2, norm='l2')
            
			euc_dis = distance.euclidean( vec1 , vec2 )
			cos_dis = distance.cosine( vec1 , vec2 )
            
			big_euc_dict[(user1,user2,pic_id)]=euc_dis
			big_cos_dict[(user1,user2,pic_id)]=cos_dis
                   
  
	return big_euc_dict, big_cos_dict, proj_rand_dist



