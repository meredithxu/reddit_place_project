import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

#Canvas color code
code_to_hex = {0: '#FFFFFF', 1: '#E4E4E4', 2: '#888888', 3: '#222222', 4: '#FFA7D1', 5: '#E50000',\
               6: '#E59500', 7: '#A06A42', 8: '#E5D900', 9: '#94E044', 10: '#02BE01', 11: '#00E5F0',\
               12: '#0083C7', 13: '#0000EA', 14: '#E04AFF', 15: '#820080', 16: '#DCDCDC'}

def hex_to_rgb(hex_val):
	'''
		Conversion from hexadecimal to RGB code
	'''
	h = hex_val.lstrip('#')
    
	return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

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
			rgb = hex_to_rgb(code_to_hex[canvas[i][j]])
			data_rgb[i,j][0] = rgb[0] 
			data_rgb[i,j][1] = rgb[1]
			data_rgb[i,j][2] = rgb[2]
 
	return data_rgb

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
        
	file.close()

	return data

def extract_canvas_num_updates(input_file_name, x_min=0, x_max=1000, y_min=0, y_max=1000,begin_time=0,end_time=sys.maxsize):
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
	
	file.close()

	return data

if __name__ == "__main__":
	#Beginning of the experiment
	begin_time = 1490918688000

	#Extracting rectancle (0-300,700-1000) after one day of the experiment
	data = extract_canvas_color('../data/sorted_tile_placements.csv', 0, 300, 700, 1000, begin_time, begin_time+1000*60*60*24)

