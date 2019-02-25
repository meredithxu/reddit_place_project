import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

#Canvas color code and conversions betweem different color formats
code_to_hex = {0: '#FFFFFF', 1: '#E4E4E4', 2: '#888888', 3: '#222222', 4: '#FFA7D1', 5: '#E50000',\
               6: '#E59500', 7: '#A06A42', 8: '#E5D900', 9: '#94E044', 10: '#02BE01', 11: '#00E5F0',\
               12: '#0083C7', 13: '#0000EA', 14: '#E04AFF', 15: '#820080', 16: '#DCDCDC'}

hex_to_code = {'#FFFFFF': 0, '#E4E4E4': 1, '#888888': 2, '#222222': 3, '#FFA7D1': 4, '#E50000': 5,\
        	'#E59500': 6, '#A06A42': 7, '#E5D900': 8, '#94E044': 9, '#02BE01': 10, '#00E5F0': 11,\
               '#0083C7': 12, '#0000EA': 13, '#E04AFF': 14, '#820080': 15, '#DCDCDC': 16}

hex_to_rgb = {'#FFFFFF': [255,255,255], '#E4E4E4': [228,228,228], '#888888': [136,136,136],\
		'#222222': [34,34,34], '#FFA7D1': [255,167,209], '#E50000': [229,0,0],\
        	'#E59500': [229,149,0], '#A06A42': [160,106,66], '#E5D900': [229,217,0],\
		'#94E044': [148,224,68], '#02BE01': [2,190,1], '#00E5F0': [0,229,240],\
               '#0083C7': [0,131,199], '#0000EA': [0,0,234], '#E04AFF': [224,74,255],\
	       '#820080': [130,0,128], '#DCDCDC': [220,220,220]}

code_to_rgb = {0: [255,255,255], 1: [228,228,228], 2: [136,136,136],\
		3: [34,34,34], 4: [255,167,209], 5: [229,0,0],\
        	6: [229,149,0], 7: [160,106,66], 8: [229,217,0],\
		9: [148,224,68], 10: [2,190,1], 11: [0,229,240],\
               12: [0,131,199], 13: [0,0,234], 14: [224,74,255],\
	       15: [130,0,128], 16: [220,220,220]}

rgb_to_code = {(255,255,255): 0, (228,228,228): 1, (136,136,136): 2,\
		(34,34,34): 3, (255,167,209): 4, (229,0,0): 5,\
        	(229,149,0): 6, (160,106,66): 7, (229,217,0): 8,\
		(148,224,68): 9, (2,190,1): 10, (0,229,240): 11,\
                (0,131,199): 12, (0,0,234): 13, (224,74,255): 14,\
	       (130,0,128): 15, (220,220,220): 16}

def rgb_code_match(rgb):
	"""
		Matches rgb color to the closest rgb color covered by the canvas.
		Comparing colors might be tricky. Using library colormath.

		Source: http://hanzratech.in/2015/01/16/color-difference-between-2-colors-using-python.html
	"""
	dist = np.zeros(len(hex_to_code)-1)
	for c in hex_to_rgb:
		if c != '#DCDCDC':
			#Computing distances using colormath
			code_rgb = sRGBColor(hex_to_rgb[c][0], hex_to_rgb[c][1], hex_to_rgb[c][2])
			code_rgb_lab = convert_color(code_rgb, LabColor)
			rgb_lab = convert_color(sRGBColor(rgb[0], rgb[1], rgb[2]), LabColor)
			dist[hex_to_code[c]] = delta_e_cie2000(code_rgb_lab, rgb_lab)
	
	min_d = np.argmin(dist)
	
	return code_to_rgb[min_d]

def hex_to_rgb_fun(hex_val):
	'''
		Conversion from hexadecimal to RGB code
	'''
	h = hex_val.lstrip('#')
    
	return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

def rgb_to_hex(rgb):
	"""
		Conversion from rgb to hexadecimal
	"""
	return "#{:02X}{:02X}{:02X}".format(rgb[0],rgb[1],rgb[2])

def float_to_rgb(minimum, maximum, value):
	'''
		Conversion of a float to RGB code. maximum will be assigned to red, minimum to
		blue.
	'''
	if value > maximum:
		value = maximum

	if value < minimum:
		value = minimum
 
	mi, ma = float(minimum), float(maximum)
	ratio = 2 * (value-mi) / (ma - mi)	
	b = int(max(0, 255*(1 - ratio)))
	r = int(max(0, 255*(ratio - 1)))
	g = 255 - b - r
 
	return (r, g, b)

def canvas_color_code_rgb(canvas):
	'''
		The canvas has its own color code (code_to_hex). This converts the canvas, which 
		is numpy integer matrix with dimensions (Y,X) to an RGB matrix (Y,X,3).
	'''
	data_rgb = np.uint8(np.zeros((canvas.shape[0], canvas.shape[1], 3)))

	#Converting 
	for i in range(canvas.shape[0]):
		for j in range(canvas.shape[1]):
			rgb = hex_to_rgb[code_to_hex[canvas[i][j]]]
			data_rgb[i,j][0] = rgb[0] 
			data_rgb[i,j][1] = rgb[1]
			data_rgb[i,j][2] = rgb[2]
 
	return data_rgb

def canvas_rgb_color_code(canvas):
	"""
		Converts canvas from rgb to the canvas color code.
	"""
	data_color_code = np.uint8(np.zeros((1000,1000)))

	for y in range(1000):
		for x in range(1000):
			if tuple(canvas[y][x]) in rgb_to_code:
				data_color_code[y][x] = rgb_to_code[tuple(canvas[y][x])]
			else:
				data_color_code[y][x] = rgb_to_code[tuple(rgb_code_match(canvas[y][x]))]
	
	return data_color_code	

def canvas_count_rgb(canvas, min_val, max_val):
	'''
		Converts float numbers on canvas to RGB. max_val is assigned to red
		and min_val to blue.
	'''
	data_rgb = np.uint8(np.zeros((canvas.shape[0], canvas.shape[1], 3)))
	
	#Converting
	for i in range(canvas.shape[0]):
		for j in range(canvas.shape[1]):
			rgb = float_to_rgb(min_val, max_val, canvas[i][j])
			data_rgb[i,j][0] = rgb[0] 
			data_rgb[i,j][1] = rgb[1]
			data_rgb[i,j][2] = rgb[2]
 
	return data_rgb

def draw_canvas(canvas, output_file_name):
	'''
		Drawing canvas into a file (figure) using matplotlib, a canvas is a numpy matrix 
		with dimensions (X,Y,3), where X is the number of position in the X-axis and Y
		is the number of positions in the Y-axis and each entry is an RGB code.
	'''
	plt.clf()

	#Drawing canvas
	plt.figure(figsize = (100,100))
	fig, ax = plt.subplots()

	ax.set_xticks([])
	ax.set_yticks([])

	plt.imshow(canvas)
    
	#High resolution is important
	plt.savefig(output_file_name, dpi=400, bbox_inches='tight')

def extract_canvas_color(input_file_name, x_min=0, x_max=1000, y_min=0, y_max=1000,begin_time=0,end_time=sys.maxsize):
	'''
		Extracts a rectanble section of the canvas at a given time.
		The input file has the same format as the original file with updates <ts,user,x,y,color>.
		The section is defined by (x_min-x_max,y_min-y_max) and time end_time is in miliseconds.
		Updates are assumed to be ordered based on timestamps.
	'''
	data = 16 * np.uint8(np.ones((y_max-y_min, x_max-x_min)))
    
	with open(input_file_name,'r') as file:
		# Skip first line (header row)
		next(file, None)
    
		reader = csv.reader(file)
		
		for r in reader:
			ts = int(r[0])
			user = r[1]
			x_coordinate = int(r[2])
			y_coordinate = int(r[3])
			color = int(r[4])
        
			if ts>=begin_time and ts <= end_time:
				if x_coordinate >= x_min and x_coordinate <= x_max\
					and y_coordinate >= y_min and y_coordinate <= y_max:
                
					data[y_coordinate-y_min-1][x_coordinate-x_min-1] = color
        
	return data

def extract_canvas_num_updates(input_file_name, x_min=0, x_max=1000, y_min=0, y_max=1000,begin_time=0,end_time=sys.maxsize):
	"""
		Extracts number of updates per tile in the canvas within the rectangle (x_min-x_max,y_mim,y_max) 
		during the time range [begin_time,end_time] from updates (input_file_name).
	"""
	data = np.zeros((y_max-y_min, x_max-x_min))
    
	with open(input_file_name,'r') as file:
		# Skip first line (header row)
		next(file, None)
    
		reader = csv.reader(file)
		
		for r in reader:
			ts = int(r[0])
			user = r[1]
			x_coordinate = int(r[2])
			y_coordinate = int(r[3])
			color = int(r[4])
        
			if ts >= begin_time and ts <= end_time:
				if x_coordinate >= x_min and x_coordinate <= x_max\
					and y_coordinate >= y_min and y_coordinate <= y_max:
                
					data[y_coordinate-y_min-1][x_coordinate-x_min-1] = data[y_coordinate-y_min-1][x_coordinate-x_min-1] + 1
	
	return data

def extract_project_color(input_file_name, filter_id, begin_time=0,end_time=sys.maxsize):
	'''
		Extracts pixel colors for a project filter_id based on updates within time 
		range [begin_time, end_time]. Update file is assumed to have picture ids (pic_id) and
		to be sorted by timestamp.
		See analytics_combined.py.
	'''

	#Extracting bounding rectangle of the project
	y_max = 0
	y_min = 1001
	x_max = 0
	x_min = 1001
    
	with open(input_file_name,'r') as file:
		# Skip first line (header row)
		next(file, None)
    
		reader = csv.reader(file)
		
		for r in reader:
			ts = int(r[0])
			user = r[1]
			x_coordinate = int(r[2])
			y_coordinate = int(r[3])
			color = int(r[4])
			pic_id = int(r[5])

			if pic_id == filter_id:
				#print(r)
				if x_coordinate <= x_min:
					x_min = x_coordinate
				if x_coordinate >= x_max:
					x_max = x_coordinate
				if y_coordinate <= y_min:
					y_min = y_coordinate
				if y_coordinate >= y_max:
					y_max = y_coordinate

	if x_max < x_min or y_max < y_min:
		print("Project is empty.")
		return

	#Creating sub-canvas for the project
	data = 16 * np.uint8(np.ones((y_max-y_min, x_max-x_min)))
	
	#Writing pixels
	with open(input_file_name,'r') as file:
		# Skip first line (header row)
		next(file, None)
    
		reader = csv.reader(file)
		
		for r in reader:
			ts = int(r[0])
			user = r[1]
			x_coordinate = int(r[2])
			y_coordinate = int(r[3])
			color = int(r[4])
			pic_id = int(r[5])
			pixel = int(r[7])
			
			if ts >= begin_time and ts <= end_time:
				if x_coordinate >= x_min and x_coordinate <= x_max\
					and y_coordinate >= y_min and y_coordinate <= y_max:
						if pic_id == filter_id and pixel == 1:
							data[y_coordinate-y_min-1][x_coordinate-x_min-1] = color
	
	return data


if __name__ == "__main__":
	#Beginning of the experiment
	begin_time = 1490918688000

	#Extracting rectancle (0-300,700-1000) after one day of the experiment
	data = extract_canvas_color('../data/sorted_tile_placements.csv', 0, 300, 700, 1000, begin_time, begin_time+1000*60*60*24)

