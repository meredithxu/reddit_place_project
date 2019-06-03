import cv2
import numpy as np
import operator
import csv

def denoise_freq(canvas, window=3):
	"""
		Denoise canvas by replacing pixel to the most frequent in a sliding 
		#window whenever that color happens only once in the window (size 3).
	"""
	shift = max(int((window-1) / 2),0)
	den_canvas = np.uint8(np.zeros((canvas.shape[0],canvas.shape[1], canvas.shape[2])))

	for y in range(canvas.shape[0]):
		for x in range(canvas.shape[1]):
			
			#Creating sliding window
			y_min = max(0, y-shift)
			y_max = min(canvas.shape[0],y+shift)
			x_min = max(0,x-shift)
			x_max = min(canvas.shape[1],x+shift)

			W = canvas[y_min:y_max,x_min:x_max]

			#Computing frequencies
			freq = {}

			for i in range(W.shape[0]):
				for j in range(W.shape[1]):
					if tuple(W[i,j]) not in freq:
						freq[tuple(W[i,j])] = 1
					else:
						freq[tuple(W[i,j])] = freq[tuple(W[i,j])] + 1

			#Replacing pixel
			if freq[tuple(canvas[y,x])] == 1:
				sorted_freq = sorted(freq.items(), key=operator.itemgetter(1))
				den_canvas[y,x] = np.uint8(sorted_freq[-1][0])
			else:
				den_canvas[y,x] = canvas[y,x]

	return den_canvas
						

def write_denoised_data(input_filename, canvas, output_filename):
	"""
		Appends denoised pixels to the end of the sequence of updates.
		Timestamps are assigned to one hour after the end of the experiment.
	"""
	max_time = 0
	with open(input_filename,'r') as file_in:
		with open(output_filename, 'w') as file_out:
			writer = csv.writer(file_out, delimiter = ",")
			writer.writerow(["ts", "user" ,"x_coordinate" ,"y_coordinate" ,"color"])

			# Skip first line (header row)
			next(file_in, None)

			reader = csv.reader(file_in)

			for r in reader:
				time = int(r[0])
				user = r[1]
				x = int(r[2])
				y = int(r[3])
				color = int(r[4])

				if time > max_time:
					max_time = time

				writer.writerow([time, user, x, y, color])
			
			#Appending denoised pixels
			for y in range(1000):
				for x in range(1000):
					writer.writerow([max_time+(1000*60*60), "denoiser", x+1, y+1, canvas[y][x]])
				
