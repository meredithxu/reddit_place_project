import copy
import sys
import networkx as nx
import csv
import math
import os
import subprocess
import numpy as np
import scipy
import time
import concurrent.futures
import pickle
import matplotlib.pyplot as plt

from user_embedding import *
from canvas_vis import *

class MyGraph:
    '''
        This is an undirected graph class for processing edge lists 
        as files. It is suitable for very large number of edges that 
        might not fit in memory.
    '''
    def __init__(self, name):
        '''
            *name* is prefixed to the edge file names
        '''
        self._edges = {}
        self.buffer_size = 10000000                            #Keeps at most 10M edges in memory
        self.n_edges = 0
        
        #File names used by this class
        self.edges_file_name = "./"+name+"_edges.csv"
        self.unique_edges_file_name = "./"+name+"_unique_edges.csv"
        self.sorted_edges_file_name = "./"+name+"_sorted_edges.csv"

    def clear(self):
        '''
        ''' 
        if os.path.exists(self.edges_file_name):
            os.remove(self.edges_file_name)
            
    def flush_edges(self):
        '''
            Flushes edges to the file <u,v,label>.
            
            Don't forget to run this after adding all the edges!
        '''
        with open(self.edges_file_name, 'a') as file_out:
            writer = csv.writer(file_out, delimiter = ",")
            
            for e in self._edges:
                writer.writerow([e[0], e[1], self._edges[e][0], self._edges[e][1]])
                
        self._edges = {}
        
    def flush_weights(self):
        '''
            Flushes edges and their weights to the file <u,v,label,weight>.
            
            Don't forget to run this after adding all the edges!
        '''
        with open(self.edges_file_name, 'a') as file_out:
            writer = csv.writer(file_out, delimiter = ",")
            
            for e in self._edges:
                writer.writerow([e[0], e[1], self._edges[e][0], self._edges[e][1], self._edges[e][2]])
                
        self._edges = {}
        
    def remove_repeated_edges(self):
        '''
            Uses sort command to remove repeated
            edges from the edge file.
        '''
        time.sleep(5)
        os.system("sort -u "+self.edges_file_name+" > "+self.unique_edges_file_name)
        
        if os.path.exists(self.edges_file_name):
            os.remove(self.edges_file_name)
            
        res = subprocess.Popen(['wc', '-l', self.unique_edges_file_name], 
        stdout=subprocess.PIPE)
        
        self.n_edges = int(res.stdout.read().decode("utf-8").split()[0])
            
    def add_edge(self, node1, node2, label, type_edge):
        '''
           Adds edge to the graph. 
        '''
        self.n_edges = self.n_edges + 1
        if node1 < node2:
            if (str(node1), str(node2)) not in self._edges:
                self._edges[(str(node1),str(node2))] = (label, type_edge)
        else:
            if (str(node2), str(node1)) not in self._edges:
                self._edges[(str(node2),str(node1))] = (label, type_edge)
        
        #Flushes edges to file.
        if len(self._edges) >= self.buffer_size:
            self.flush_edges()     
            
    def set_weight(self, node1, node2, label, type_edge, weight):
        '''
            Sets edge weight.
        '''
        if node1 < node2:
            if (str(node1), str(node2)) not in self._edges:
                self._edges[(str(node1),str(node2))] = [label, type_edge, weight]
        else:
            if (str(node2), str(node1)) not in self._edges:
                self._edges[(str(node2),str(node1))] = [label, type_edge, weight]
        
        #Flushes edge to file
        if len(self._edges) >= self.buffer_size:
            self.flush_weights()     
            
    def sort_edges(self):
        '''
            Sorts edges in increasing order of weight.
        '''
        time.sleep(5)
        os.system("sort -g -t, -k5,5 "+self.edges_file_name+" > "+self.sorted_edges_file_name)


def get_label(upi, upj, pixel=False):
    '''
    '''
    if pixel is True:
        if upi[5] != upj[5] and int(upi[6]) == 1 and int(upj[6]) == 1:
            return 0
        elif int(upi[6]) == 1 and int(upj[6]) == 1:
            return 1
        elif upi[5] != upj[5] and (int(upi[5]) == 0 or int(upj[5]) == 0) and (int(upi[6]) == 1 or int(upj[6]) == 1):
            return 0
        else: 
            return 2
    else:
        if upi[5] != upj[5] and int(upi[7]) == 1 and int(upj[7]) == 1:
            return 0
        elif int(upi[7]) == 1 and int(upj[7]) == 1:
            return 1
        elif upi[5] != upj[5] and (int(upi[5]) == 0 or int(upj[5]) == 0) and (int(upi[7]) == 1 or int(upj[7]) == 1):
            return 0
        else: 
            return 2 

def compute_update_durations(ups):
    '''
        Computes the time duration of each update [begin,end]
    '''
    max_uint32 = 4294967295    #Using this for undefined
    xy_ups = max_uint32 * np.ones((1001,1001),dtype=np.uint32)
    durations = []
    max_time = 0

    for i in range(len(ups)):
        ts = int(ups[i][0])
        user = ups[i][1]
        x = int(ups[i][2])
        y = int(ups[i][3])

        if xy_ups[x][y] < max_uint32:
            durations[xy_ups[x][y]][1] = ts

        xy_ups[x][y] = i
        durations.append([ts,0])

        if ts > max_time:
            max_time = ts

    for x in range(xy_ups.shape[0]):
        for y in range(xy_ups.shape[1]):
            if xy_ups[x][y] < max_uint32:
                durations[xy_ups[x][y]][1] = max_time+1

    return durations

def check_overlap(dur_i, dur_j):
    if dur_i[0] > dur_j[1] or dur_j[0] > dur_i[1]:
        return False
    else:
        return True
    
def create_graph(input_file_name, projects_to_remove, space_threshold=1, 
                 min_x=0, max_x=1002, min_y=0, max_y=1002, file_prefix="ups"):
    '''
        Creates networkx graph of updates within a given frame (or time window) and associated 
        list of update info (timestamp, user, color etc.).
        The input file contains the sorted list of updates with project assignments.
        The values of min_time and max_time define the frame time window.
        Updates are connected spatially based on the space threshold and temporally based
        on the time threshold (in seconds).
        min_x,max_x,min_y,max_y define the area of the canvas for which updates will be considered.

        It returns a networx graph for which node IDs are indexes for the list of updates.
    '''

    updates = []
    
    xy_index = []
    for i in range(1001):
        xy_index.append([])
        for j in range(1001):
            xy_index[i].append([])
    
    i = 0
    with open(input_file_name,'r') as file:
        # Skip first line (header row)
        next(file, None)
        reader = csv.reader(file)

        for r in reader:
            ts = int(r[0])
            x = int(r[2])
            y = int(r[3])
            proj = r[5]
            pixel = int(r[6])
            proj_smallest = int(r[8])
            if proj_smallest == 1 or proj == '0':
                if x >= min_x and x < max_x and y >= min_y and y < max_y:
                    user = r[1]
                    color = int(r[4])
                    
                    if proj not in projects_to_remove:
                        pixel = int(r[6])
                        pixel_color = int(r[7])

                        #Collecting update info
                        updates.append([ts, user, x, y, color, proj, pixel, pixel_color])
                    else:
                        pixel = 0
                        pixel_color = 0
                        proj = 0

                        #Collecting update info
                        updates.append([ts, user, x, y, color, proj, pixel, pixel_color])

                    #Spatial index for updates. This is used to find updates near each other.
                    xy_index[x][y].append(i)
                    i = i + 1
    
    
    
    G = MyGraph(file_prefix)
    G.clear()
    
    durations = compute_update_durations(updates)
    
    for xi in range(1001):
        for yi in range(1001):
            for xj in range(xi-space_threshold, xi+space_threshold+1):
                for yj in range(yi-space_threshold, yi+space_threshold+1):
                    if xj >= 0 and yj >= 0 and xj < 1001 and yj < 1001:

                        if xi == xj and yi == yj:
                            #Each update is connected at least to the previous update
                            #of the same pixel.
                            for i in range(0, len(xy_index[xi][yi])-1):
                                if xy_index[xi][yi][i] != xy_index[xi][yi][i+1]:
                                    label = get_label(updates[xy_index[xi][yi][i]],
                                            updates[xy_index[xi][yi][i+1]])

                                    if updates[xy_index[xi][yi][i]][4] != updates[xy_index[xi][yi][i+1]][4]:
                                        G.add_edge(xy_index[xi][yi][i], xy_index[xi][yi][i+1], label, -1)
                                    else:
                                        G.add_edge(xy_index[xi][yi][i], xy_index[xi][yi][i+1], label, 1)

                        #Connecting nodes based on spatial and temporal proximity
                        elif len(xy_index[xi][yi]) > 0 and len(xy_index[xj][yj]) > 0:
                            i = 0
                            j = 0

                            while i < len(xy_index[xi][yi]) and j < len(xy_index[xj][yj]):
                                ui = updates[xy_index[xi][yi][i]]
                                uj = updates[xy_index[xj][yj][j]]

                                #if ui[0]-uj[0] < 1000 * time_threshold and ui[0]-uj[0] >= 0:
                                #print(len(durations), xy_index[xi][yi][i], xy_index[xi][yi][i])
                                if check_overlap(durations[xy_index[xi][yi][i]], durations[xy_index[xj][yj][j]]):
                                    if xy_index[xi][yi][i] != xy_index[xj][yj][j]:
                                        label = get_label(updates[xy_index[xi][yi][i]], 
                                            updates[xy_index[xj][yj][j]])
                                        G.add_edge(xy_index[xi][yi][i], xy_index[xj][yj][j], label, +1)

                                if ui[0] < uj[0]:
                                    i = i + 1
                                else:
                                    j = j + 1

                            if j == len(xy_index[xj][yj]):
                                while i < len(xy_index[xi][yi]):
                                    ui = updates[xy_index[xi][yi][i]]
                                    uj = updates[xy_index[xj][yj][j-1]]

                                    #if ui[0]-uj[0] < 1000 * time_threshold and ui[0]-uj[0] >= 0:
                                    if check_overlap(durations[xy_index[xi][yi][i]], durations[xy_index[xj][yj][j-1]]):
                                        if xy_index[xi][yi][i] != xy_index[xj][yj][j-1]:
                                            label = get_label(updates[xy_index[xi][yi][i]], 
                                                updates[xy_index[xj][yj][j-1]])
                                            G.add_edge(xy_index[xi][yi][i], xy_index[xj][yj][j-1], label, +1)
                                    i = i + 1

                            elif i == len(xy_index[xi][yi]):
                                while j < len(xy_index[xj][yj]):
                                    ui = updates[xy_index[xi][yi][i-1]]
                                    uj = updates[xy_index[xj][yj][j]]
                                    #if ui[0]-uj[0] < 1000 * time_threshold and ui[0]-uj[0] >= 0:
                                    if check_overlap(durations[xy_index[xi][yi][i-1]], durations[xy_index[xj][yj][j]]):
                                        if xy_index[xi][yi][i-1] != xy_index[xj][yj][j]:
                                                label = get_label(updates[xy_index[xi][yi][i-1]], 
                                                    updates[xy_index[xj][yj][j]])
                                                G.add_edge(xy_index[xi][yi][i-1], xy_index[xj][yj][j], label, +1)
                                    j = j + 1

    
    G.flush_edges()
    G.remove_repeated_edges()
       
    return G, updates   

def split_updates(ups, excluded_folds, fold_boundaries):
    '''
        folds_boundaries is a list of dictionaries, where each dictionary contains the following keys:
            min_x, max_x, min_y, max_y
        Excluded folds is a list of indexes for which folds should be excluded
        All updates that fall within excluded_folds are going into ups_eval
        All other updates will go into ups_training
    '''
    ups_training = []
    ups_eval = []

    # A set of all the project IDs that are being excluded
    excluded_projects = set() 

    for update in ups:
        ts = update[0]
        user = update[1]
        x = update[2]
        y = update[3]
        color = update[4]
        proj = update[5]
        pixel = update[6]
        pixel_color = update[7]

        if proj != 0:
            for f_idx in excluded_folds:
                if is_within_fold(x, y, fold_boundaries[f_idx]):
                    excluded_projects.add(proj)
                    break
    
    for update in ups:
        ts = update[0]
        user = update[1]
        x = update[2]
        y = update[3]
        color = update[4]
        proj = update[5]
        pixel = update[6]
        pixel_color = update[7]
        
        if proj in excluded_projects:
            ups_eval.append([ts, user, x, y, color, proj, pixel, pixel_color])
            ups_training.append([ts, user, x, y, color, 0, 0, 0])
        else:
            ups_eval.append([ts, user, x, y, color, 0, 0, 0])
            ups_training.append([ts, user, x, y, color, proj, pixel, pixel_color])
       
        return ups_training, ups_eval
    
            
def compute_user_relations(G, ups, rel_type=1):
    '''
        Computes two matrices based on user neighborhood activity. 
        There are three types of user relations:
        
        (1) Negative if users overwrite each other, positive otherwise.
        (2) Positive if users apply same color, negative if they 
        overwrite each other.
        (3) Positive if users apply same color, negative otherwise.
        These matrices are in sparse format. The function also returns
        an index for mapping user ids to matrix indices.
    '''
    user_relations = {}
    user_index = {}
    
    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = r[2]
            type_edge = r[3]
            
            user_u = ups[u][1]
            user_v = ups[v][1]
            
            color_u = ups[u][4]
            color_v = ups[v][4]
            
            x_u = ups[u][2]
            x_v = ups[v][2]
            y_u = ups[u][3]
            y_v = ups[v][3]
            
            if user_u not in user_index:
                user_index[user_u] = len(user_index)
                user_relations[user_index[user_u]] = {}
                    
            if user_v not in user_index:
                user_index[user_v] = len(user_index)
                user_relations[user_index[user_v]] = {}
            
            idx_u = user_index[user_u]
            idx_v = user_index[user_v]
            
            #Only immediate neighbors considered
            if np.abs(x_u-x_v) <= 1 and np.abs(y_u-y_v) <= 1:
                if idx_u < idx_v:
                    if idx_v not in user_relations[idx_u]:
                        user_relations[idx_u][idx_v] = [0, 0]
                
                    if rel_type == 1:
                        if x_u == x_v and y_u == y_v and color_u != color_v:
                            user_relations[idx_u][idx_v][1] = user_relations[idx_u][idx_v][1] + 1
                        else:
                            user_relations[idx_u][idx_v][0] = user_relations[idx_u][idx_v][0] + 1
                    elif rel_type == 2:
                        if color_u == color_v:
                            user_relations[idx_u][idx_v][0] = user_relations[idx_u][idx_v][0] + 1
                        elif x_u == x_v and y_u == y_v:
                            user_relations[idx_u][idx_v][1] = user_relations[idx_u][idx_v][1] + 1
                    else:
                        if color_u == color_v:
                            user_relations[idx_u][idx_v][0] = user_relations[idx_u][idx_v][0] + 1
                        else:
                            user_relations[idx_u][idx_v][1] = user_relations[idx_u][idx_v][1] + 1
                
                else:
                    if idx_u not in user_relations[idx_v]:
                        user_relations[idx_v][idx_u] = [0, 0]
                
                    if rel_type == 1:
                        if x_u == x_v and y_u == y_v and color_u != color_v:
                            user_relations[idx_v][idx_u][1] = user_relations[idx_v][idx_u][1] + 1 
                        else:
                            user_relations[idx_v][idx_u][0] = user_relations[idx_v][idx_u][0] + 1
                    elif rel_type == 2:    
                        if color_u == color_v:
                            user_relations[idx_v][idx_u][0] = user_relations[idx_v][idx_u][0] + 1
                        elif x_u == x_v and y_u == y_v:
                            user_relations[idx_v][idx_u][1] = user_relations[idx_v][idx_u][1] + 1 
                    else:
                        if color_u == color_v:
                            user_relations[idx_v][idx_u][0] = user_relations[idx_v][idx_u][0] + 1
                        else:
                            user_relations[idx_v][idx_u][1] = user_relations[idx_v][idx_u][1] + 1 
                    
    #Creating sparse data structure for memory savings
    rows = []
    cols = []
    vals = []

    for ui in user_relations:
        for uj in user_relations[ui]:
            if user_relations[ui][uj][0] > 0:
                rows.append(ui)
                cols.append(uj)
                vals.append(user_relations[ui][uj][0])

    same_color = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_index)+1, len(user_index)+1), dtype=np.uint16)
    
    rows = []
    cols = []
    vals = []

    for ui in user_relations:
        for uj in user_relations[ui]:
            if user_relations[ui][uj][1] > 0:
                rows.append(ui)
                cols.append(uj)
                vals.append(user_relations[ui][uj][1])

    diff_color = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(user_index)+1, len(user_index)+1), dtype=np.uint16)
    
    return same_color, diff_color, user_index

def compute_user_color(ups):
    '''
        Computes a color histogram for each user.
    '''
    user_index = {}

    for i in range(len(ups)):
        user = ups[i][1]
        if user not in user_index:
            user_index[user] = len(user_index)

    user_color = np.zeros((len(user_index)+1, 16))

    for i in range(len(ups)):
        user = ups[i][1]
        x = ups[i][2]
        y = ups[i][3]
        color = int(ups[i][4])

        idx_user = user_index[user]
        user_color[idx_user][color] = user_color[idx_user][color] + 1

    for i in range(user_color.shape[0]):
        if user_color[i].sum() > 0:
            user_color[i] = user_color[i] / user_color[i].sum()

    return scipy.sparse.csr_matrix(user_color), user_index


def dist_duration(dur_i, dur_j):
    '''
        Distance function for two durations.
    '''
    if dur_i[0] > dur_j[1] or dur_j[0] > dur_i[1]:
        return 1. # 0 overlap
    else:
        a = max(dur_i[0], dur_j[0])
        b = min(dur_i[1], dur_j[1])

        if dur_i[1]-dur_i[0]+dur_j[1]-dur_j[0] == 0:
            return 1.

        overlap = 2 * (b-a) / (dur_i[1]-dur_i[0]+dur_j[1]-dur_j[0])

        return 1.-overlap

def create_components(ups):
    '''
        Creates one component per node in the graph.
        
        Returns a dictionary mapping nodes to components
        and also the internal weights (0) and sizes (1) of
        the components.
    '''
    comp_assign = {}
    internal_weights = []
    sizes = []
    regions = []
    
    for v in range(len(ups)):
        comp_assign[v] = len(comp_assign)
        internal_weights.append(0.)
        sizes.append(1)
        regions.append([v])
        
    return comp_assign, internal_weights, sizes, regions

def compute_m_int(int_weights, sizes, comp_u, comp_v, KAPPA):
    ''' 
        Computing internal weights for a pair of components.
        
        KAPPA is a small value that helps with small components.
    '''
    tau_1 = KAPPA / sizes[comp_u]
    tau_2 = KAPPA / sizes[comp_v]
    
    return min(int_weights[comp_u]+tau_1, int_weights[comp_v]+tau_2)

def merge(G, comp_assign, int_weights, sizes, regions, comp_u, comp_v, w):
    '''
        Merges two components (comp_u and comp_v) and updates
        their respective internal weights.
        
        Returns the updated component assignments, internal weights
        and component sizes.
    '''
    if sizes[comp_u] > sizes[comp_v]:
        new_comp = comp_u
        old_comp = comp_v
    else:
        new_comp = comp_v
        old_comp = comp_u
        
    for v in regions[old_comp]:
        comp_assign[v] = new_comp
    
    regions[new_comp].extend(regions[old_comp])
    regions[old_comp] = []
            
    sizes[new_comp] = sizes[comp_u] + sizes[comp_v]
    sizes[old_comp] = 0
        
    int_weights[new_comp] = w
    
    return comp_assign, int_weights, sizes, regions

def region_segmentation(G, ups, KAPPA):
    '''
        Implements region segmentation algorithms by Felzenszwalb and Huttenlocher.
        
        Paper: http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
        
        Returns dictionary mapping node to region.
    '''
    comp_assign, int_weights, sizes, regions = create_components(ups)
      
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = r[2]
            type_edge = int(r[3])
            w = r[4]
            w = float(w.replace("[", "").replace("]", ""))

            if type_edge > 0:
                comp_u = comp_assign[u]
                comp_v = comp_assign[v]
        
                if comp_u != comp_v:
                    m_int = compute_m_int(int_weights, sizes, comp_u, comp_v, KAPPA)
            
                    if w < m_int:
                        comp_assign, int_weights, sizes, regions = merge(G, comp_assign, int_weights, sizes, regions,
                            comp_u, comp_v, w)
        
    return comp_assign, int_weights

def extract_regions(comp_assign, _int_weights):
    '''
        Extracts actual regions (lists of updates) from region assignements.
        Also returns the region sizes.
    '''
    regions = []
    sizes = []
    int_weights = []
    
    region_ids = {}
    
    for c in comp_assign:
        if comp_assign[c] not in region_ids:
            region_ids[comp_assign[c]] = len(region_ids)
            regions.append([])
            sizes.append(0)
            int_weights.append(0)

    for c in comp_assign:
        ID = region_ids[comp_assign[c]]
        regions[ID].append(c)
        sizes[ID] = sizes[ID] + 1
        
    for c in range(len(_int_weights)):
        if c in region_ids:
            ID = region_ids[c]
            int_weights[ID] = _int_weights[c]
        
    for r in regions:
        r.sort()
    
    return regions, sizes, int_weights

def extract_canvas_region(updates, region):
    '''
        Gets canvas colors for a region
    '''
    data_color_code = np.uint8(np.zeros((1001,1001)))
    
    for pixel in region:
        u = updates[int(pixel)]
        x = u[2]
        y = u[3]
        color = u[4]
        
        data_color_code[y][x] = color
        
    return data_color_code

def draw_canvas_region(updates, region, out_file):
    '''
        Draws region into a file.
    '''
    canvas_ups = extract_canvas_region(updates, region)
    
    canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)
    draw_canvas(canvas_ups_rgb, out_file)

def extract_canvas_updates_region(updates, min_x=0, max_x=1000, min_y=0, max_y=1000):
    '''
        Creates canvas representation of a set of udpates.
    '''
    data_color_code = np.uint8(np.zeros((max_y-min_y+1,max_x-min_x+1)))
    
    for u in updates:
        x = u[2]
        y = u[3]
        color = u[4]

        data_color_code[y-min_y][x-min_x] = color

    return data_color_code


def create_folds(num_folds = 10, min_x=0, min_y=0, max_x=1002, max_y=1002):
    # Partition the data into folds

    num_yincrements = num_folds // 2
    folds = []
    for i in range(num_folds):
        folds.append([])

    halfway_x = int((min_x + max_x) // 2)
    y_increment = int((max_y - min_y) // num_yincrements)

    for j in range(num_yincrements):

        for x in range(min_x, halfway_x):
            for y in range((j * y_increment) + min_y, ((j + 1) * y_increment) + min_y):
                folds[j].append((x, y))

        for x in range(halfway_x, max_x):
            for y in range((j * y_increment) + min_y, ((j + 1) * y_increment) + min_y):
                folds[num_yincrements + j].append((x, y))

    return folds

	
def get_fold_border(fold):
    '''
        Return a dictionary with the min_x, max_x, min_y, and max_y values of the fold
    '''
    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0
    for coordinate in fold:
        x = coordinate[0]
        y = coordinate[1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}
    
def is_within_fold(x, y, boundary):
    '''
        Return true if (x,y) is within fold boundary
        Return false otherwise
    '''
    if x <= boundary["max_x"] and x >= boundary["min_x"] and y <= boundary["max_y"] and y >= boundary["min_y"]:
        return True
    return False

def build_feat_label_data(G, ups, features, pixel=False, train_x_y=None, fold_boundaries=None, excluded_folds=None):
    '''
        Extracts feature values and labels for edges in the graph.

        features is a dictionary with both functions that compute
        feature values and the data structures they require, used as:

        A[i,f] = features[f]['func'](u, v, ups, features[f]['data'])

        if pixel is true, only the final updates are considered as 
        training data, otherwise, updates that match the final color
        are also considered. 

        train_x_y allows you to select which x,y positions are used
        for training the model.

        fold_boundaries is a list of dictionaries of the following format:
        {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}
        Each dictionary indicates the boundary of a fold.

        excluded_folds is a list of indexes indicating which corresponding folds within fold_boundaries
        are to be excluded from the feature and label data.
        If fold_boundaries is None, then all folds are included.

        Returns matrix A with feature values and vector b with labels
    '''
    #Counting labelled points
    n_labelled = 0
    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = r[2]
            type_edge = int(r[3])

            #updates.append([ts, user, x, y, color, proj, pixel, pixel_color])
            x_u = ups[u][2]
            y_u = ups[u][3]
            x_v = ups[v][2]
            y_v = ups[v][3]
        
            if type_edge > 0 and (train_x_y is None or ((x_u,y_u) in train_x_y and (x_v,y_v) in train_x_y)):
                
                if fold_boundaries is not None:
                    # Check which fold this edge belongs to. If it belongs to an excluded fold, then skip it
                    skip_edge = False
                    for f_idx in excluded_folds:
                        if is_within_fold(x_u, y_u, fold_boundaries[f_idx]) or is_within_fold(x_v, y_v, fold_boundaries[f_idx]):
                            skip_edge = True

                    if skip_edge:
                        continue

                if pixel is True:
                    if int(ups[u][6]) == 1 and int(ups[v][6]) == 1:
                        n_labelled = n_labelled + 1
                else:
                    if int(ups[u][7]) == 1 and int(ups[v][7]) == 1:
                        n_labelled = n_labelled + 1
            
    A = np.zeros((n_labelled, len(features)))
    b = np.zeros(n_labelled)
    
    i = 0
    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = int(r[2])
            type_edge = int(r[3])

            x_u = ups[u][2]
            y_u = ups[u][3]
            x_v = ups[v][2]
            y_v = ups[v][3]

            if type_edge > 0 and (train_x_y is None or ((x_u,y_u) in train_x_y and (x_v,y_v) in train_x_y)):

                if fold_boundaries is not None:
                    # Check which fold this edge belongs to. If it belongs to an excluded fold, then skip it
                    skip_edge = False
                    for f_idx in excluded_folds:
                        if is_within_fold(x_u, y_u, fold_boundaries[f_idx]) or is_within_fold(x_v, y_v, fold_boundaries[f_idx]):
                            skip_edge = True

                    if skip_edge:
                        continue

                if pixel is True:
                    if int(ups[u][6]) == 1 and int(ups[v][6]) == 1:
                        for f in range(len(features)):
                            A[i,f] = features[f]['func'](u, v, ups, features[f]['data'])
                            
                        if lb == 1:
                            b[i] = float(0.)
                        else:
                            b[i] = float(1.)
                
                        i = i + 1
                else:
                    if int(ups[u][7]) == 1 and int(ups[v][7]) == 1:
                        for f in range(len(features)):
                            A[i,f] = features[f]['func'](u, v, ups, features[f]['data'])
                            
                        if lb == 1:
                            b[i] = float(0.)
                        else:
                            b[i] = float(1.)
                
                        i = i + 1
    
    return A, b

def compute_weight(edge_buffer, ups, m, features, scalerX= None, scalerY = None):
    '''
        Computes weight of edge (upi,upj) based on several features.

        scalerX and scalerY are filenames for saved standard scalers to scale the input and output of m
        If either is none, then no scaling will occur
    '''
    feat_values = np.zeros((len(edge_buffer), len(features)))
    
    for e in range(len(edge_buffer)):
        u = edge_buffer[e][0]
        v = edge_buffer[e][1]

        for f in range(len(features)):
            feat_values[e][f] = features[f]['func'](u, v, ups, features[f]['data'])
        
    print("Feature shape:", feat_values.shape)
    # Check if you need to scale the values
    results = None
    if scalerX == None or not os.path.exists(scalerX):
        results = m.predict(feat_values)
    else:
        sc = pickle.load(open(scalerX, 'rb'))
        results = m.predict(sc.transform(feat_values))

    print("result shape:", results.shape)
    # Ensure that the dimensions of result is a 2d array
    if results.ndim == 1:
        results = results.reshape(-1, 1)

    if scalerY == None or not os.path.exists(scalerY):
        return results
    else:
        sc = pickle.load(open(scalerY, 'rb'))
        results = sc.inverse_transform(results)

        return results

def compute_weight_wrapper(param):
    '''
        Simple wrapper for the compute_weight function
    '''    
    #Loading pickled features
    #Each thread has its own copy, which is quite inneficient
    pfile = open(param[3], 'rb')
    features = pickle.load(pfile)
    pfile.close()

    return compute_weight(param[0], param[1], param[2], features, param[4], param[5])

def compute_weight_multithread(edge_buffer, ups, model, features_file_name, n_threads, scalerX = None, scalerY = None):
    '''
        Computes weights for set of edges in edge_buffer using multithreading
    '''

    #Dividing the work
    edges_per_thread = int(len(edge_buffer) / n_threads)

    edge_parts = []
    for t in range(n_threads):
        edge_parts.append([])

    e = 0
    for e in range(len(edge_buffer)):
        t = e % n_threads
        edge_parts[t].append(edge_buffer[e])

    futures = []

    #Multithreading
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
        for t in range(n_threads):
            fut = executor.submit(compute_weight_wrapper, (edge_parts[t], ups, model, features_file_name, scalerX, scalerY,))
            futures.append(fut)

    #Collecting results
    W = np.zeros(len(edge_buffer))
    for t in range(n_threads):
        fut = futures[t]
        res = fut.result()
        for e in range(res.shape[0]):
            W[e*n_threads+t] = res[e]

    return W


def compute_edge_weights_multithread(G, ups, model, features, features_file_name, n_threads, scalerX=None, scalerY=None):
    '''
        Computes weights for edges in the graph using multithreading.

	Notice that edges with negative weight (one update remove another)
	are not included in the output file!

    '''

    if os.path.exists(G.edges_file_name):
        os.remove(G.edges_file_name)
    
    #Pickling feature data to be shared with threads
    if not os.path.exists(features_file_name):
         pfile = open(features_file_name, 'wb')
         pickle.dump(features, pfile)
         pfile.close()
        
    edge_buffer = []

    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = r[0]
            v = r[1]
            lb = r[2]
            type_edge = int(r[3])


            if type_edge > 0:
                edge_buffer.append((int(u), int(v), lb, type_edge))

                if len(edge_buffer) >= G.buffer_size:

                    W = compute_weight_multithread(edge_buffer, ups, model, features_file_name, n_threads, scalerX, scalerY)

                    for e in range(len(edge_buffer)):
                        u = edge_buffer[e][0]
                        v = edge_buffer[e][1]
                        lb = edge_buffer[e][2]
                        type_edge = edge_buffer[e][3]
                        w = W[e]

                        G.set_weight(u, v, lb, type_edge, w)

                    edge_buffer = []

    if len(edge_buffer) > 0:
        W = compute_weight_multithread(
            edge_buffer, ups, model, features_file_name, n_threads, scalerX, scalerY)

        for e in range(len(edge_buffer)):
            u = edge_buffer[e][0]
            v = edge_buffer[e][1]
            lb = edge_buffer[e][2]
            type_edge = edge_buffer[e][3]
            w = W[e]

            G.set_weight(u, v, lb, type_edge, w)

    G.flush_weights()


def compute_edge_weights(G, ups, model, features, scalerX, scalerY):
    '''
        Computes weights for edges in the graph.
	
	Notice that edges with negative weight (one update remove another)
	are not included in the output file!
    '''
    if os.path.exists(G.edges_file_name):
        os.remove(G.edges_file_name)
    
    edge_buffer = []        

    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = r[0]
            v = r[1]
            lb = r[2]
            type_edge = int(r[3])
        
            if type_edge > 0:
                edge_buffer.append((int(u), int(v), lb, type_edge))
                
                if len(edge_buffer) >= G.buffer_size:
                    
                    W = compute_weight(edge_buffer, ups, model, features, scalerX, scalerY)
                    
                    for e in range(len(edge_buffer)):
                        u = edge_buffer[e][0]
                        v = edge_buffer[e][1]
                        lb = edge_buffer[e][2]
                        type_edge = edge_buffer[e][3]
                        w = W[e]
                        
                        G.set_weight(u, v, lb, type_edge, w)
		
                    edge_buffer = []
                
    if len(edge_buffer) > 0:
        W = compute_weight(edge_buffer, ups, model, features, scalerX, scalerY)
                    
        for e in range(len(edge_buffer)):
            u = edge_buffer[e][0]
            v = edge_buffer[e][1]
            lb = edge_buffer[e][2]
            type_edge = edge_buffer[e][3]
            w = W[e]
                        
            G.set_weight(u, v, lb, type_edge, w)
        
    G.flush_weights()

def compute_update_conflicts(ups):
    '''
        For each update up, computes the conflicting updates
        i.e. the update that up overwrote and the one that
        overwrote up.

        max_uint32 is used as None (for first and last updates
        at a given position).
    '''
    max_uint32 = 4294967295    #Using this for undefined
    conflicts = max_uint32 * np.ones((len(ups), 2),dtype=np.uint32)
    xy_ups = max_uint32 * np.ones((1001,1001),dtype=np.uint32)

    for i in range(len(ups)):
        x = int(ups[i][2])
        y = int(ups[i][3])

        if xy_ups[x][y] < max_uint32:
            conflicts[xy_ups[x][y]][1] = i
            
        conflicts[i][0] = xy_ups[x][y]
        xy_ups[x][y] = i
                
    return conflicts

def embed_users(G, ups, ndim=40, threshold=5, total_samples=100, n_negatives=5, n_iterations=10, compute_balance=False):
    '''
        Embeds users using signed embedding algorithm.
        Edges between users (u,v) can be either negative 
        or positive depending on whether they mostly write
        same color updates nearby each other of overwrite
        each others' updates with different colors.
    '''
    usr_rel_same_color, usr_rel_diff_color, user_index = compute_user_relations(G, ups, rel_type=2)

    sign_embedding(usr_rel_same_color, usr_rel_diff_color, user_index, "signet", ndim, threshold, 
	total_samples, n_negatives, n_iterations, compute_balance)

    emb = read_embedding("signet", user_index, ndim)

    avg_distances_pos_neg(emb)
    
    return user_index, emb

def updates_region(ups, region):
    '''
        Extracts updates in a given region.
    '''
    ups_region = []
    
    for u in sorted(region):
        ups_region.append(ups[int(u)])
        
    return ups_region

def proj_graph(G, ups, projs, pixel):
    '''
        Creats update graph for projects in proj, where nodes are updates and edges within
        projects are colored green and edges across projects are colored red. Distances 
        between updates are proportional to weight of their edges.
    '''
    G_proj = nx.Graph()
    code_to_hex = {0: '#FFFFFF', 1: '#E4E4E4', 2: '#888888', 3: '#222222', 4: '#FFA7D1', 5: '#E50000',\
        6: '#E59500', 7: '#A06A42', 8: '#E5D900', 9: '#94E044', 10: '#02BE01', 11: '#00E5F0',\
        12: '#0083C7', 13: '#0000EA', 14: '#E04AFF', 15: '#820080', 16: '#DCDCDC'}

    node_colors = []
    
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            w = float(r[4])
            type_edge = int(r[3])
            
            if ups[u][5] in projs and ups[v][5] in projs and type_edge > 0:
                if (pixel and ups[u][6] == 1 and ups[v][6] == 1) or (not pixel and ups[u][7] == 1 and ups[v][7] == 1):
                    G_proj.add_edge(u,v, weight=1./(w+1.e-10))
                        
                    #G_proj.add_edge(u,v)
    
    G_proj = max(nx.connected_component_subgraphs(G_proj), key=len)
    
    for v in G_proj.nodes():
        up = ups[int(v)]
        
        if up[5] in projs:
            if (pixel and up[6] == 1) or (not pixel and up[7] == 1):
                color = int(up[4])
                node_colors.append(code_to_hex[color])
     
    edge_colors = []
    for e in G_proj.edges():
        u = int(e[0])
        v = int(e[1])
        
        if ups[u][5] == ups[v][5]:
            edge_colors.append('green')
        else:
            edge_colors.append('red')
    
    return G_proj, node_colors, edge_colors

def draw_proj_graph(G_proj, output_file_name="../plots/graph.png"):
    '''
        Draws the update graph into the file given as input.
    '''
    f = plt.figure()
    pos = nx.spring_layout(G_proj, weight='weight')
    nx.draw(G_proj, pos, node_size=6., width=0.05, node_color=node_colors, edge_color=edge_colors, linewidths=.5)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("black") 
    f.savefig(output_file_name)

def extract_region(comp_assign, r):
    '''
        Extracts single region from update assignments
    '''
    region = []
    
    for u in range(len(comp_assign)):
        if comp_assign[u] == r:
            region.append(u)
            
    return region

def draw_region(ups, region, output_file_name):
    '''
        Draws region into file.
    '''
    ups_region = updates_region(ups, region)
        
    plt.close('all')

    canvas_ups = extract_canvas_updates_region(ups_region)
    canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)

    draw_canvas(canvas_ups_rgb, output_file_name)

def draw_top_regions(ups, regions, sizes, k, output_file_name='../plots/top_region'):
    '''
        Draws top k largest regions
    '''
    ind  = np.argsort(sizes)
    
    for i in range(1, k):
        r = ind[-i]
        
        draw_region(ups, regions[r], output_file_name+"_"+str(i)+".svg")

def region_label(region, ups, projects_to_remove, pixel=False):
    '''
        Computes region label using majority voting.
        Also returns the participation of the majority project
        in the region.
    '''
    projs = {}
        
    for u in region:
        #updates.append([ts, user, x, y, color, proj, pixel, pixel_color])
        if pixel is True:
            if ups[u][6] == 1:
                proj = ups[u][5]
                if proj not in projects_to_remove:
                    if proj not in projs:
                        projs[proj] = 0

                    projs[proj] = projs[proj] + 1
        else:
            if ups[u][7] == 1:
                proj = ups[u][5]
                if proj not in projects_to_remove:
                    if proj not in projs:
                        projs[proj] = 0

                    projs[proj] = projs[proj] + 1

    if len(projs) > 0:
        max_proj, max_count = max(projs.items(), key=operator.itemgetter(1))
    else:
        max_proj = 0
        max_count = 0

    return max_proj, max_count / len(region)

def build_region_graph(G, regions, ups, label_threshold, projects_to_remove, name='reg'):
    '''
        Builds graph of adjacent regions
    '''
    comp_assign = []

    for u in ups:
        comp_assign.append(0)

    region_labels = []
    n_labelled = 0
    for r in range(len(regions)):
        for u in regions[r]:
            comp_assign[u] = r

        lb, ratio = region_label(regions[r], ups, projects_to_remove)

        #If region membership is above threshold
        #assign region to project
        if ratio >= label_threshold:
            region_labels.append(int(lb))
            n_labelled = n_labelled + 1
        else:
            region_labels.append(0)

    print("#labelled regions = ", n_labelled)
    print("#unlabelled regions = ", len(regions)-n_labelled)

    G_reg = MyGraph(name)
    G_reg.clear()

    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = r[2]
            type_edge = int(r[3])

            reg_u = comp_assign[u]
            reg_v = comp_assign[v]

            if reg_u != reg_v and type_edge > 0:
                G_reg.add_edge(reg_u, reg_v, 0, 1)

    G_reg.flush_edges()
    G_reg.remove_repeated_edges()

    true_edges = 0
    false_edges = 0
    unlabelled_edges = 0
    
    with open(G_reg.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            reg_u = int(r[0])
            reg_v = int(r[1])
            lb = r[2]
            type_edge = int(r[3])

            lb_u = region_labels[reg_u]
            lb_v = region_labels[reg_v]

            #add_edge(self, node1, node2, label, type_edge)
            if lb_u > 0 or lb_v > 0:
                if lb_u == lb_v:
                    G_reg.add_edge(reg_u, reg_v, 1, 1)
                    false_edges = false_edges + 1
                else:
                    G_reg.add_edge(reg_u, reg_v, 0, 1)
                    true_edges = true_edges + 1
            else:
                G_reg.add_edge(reg_u, reg_v, 2, 1)
                unlabelled_edges = unlabelled_edges + 1

    G_reg.flush_edges()
    G_reg.remove_repeated_edges()

    print("#true edges = ", true_edges)
    print("#false edges = ", false_edges)
    print("#unlabelled edges = ", unlabelled_edges)

    return G_reg


def compute_user_vector_regions(regions, user_vec, user_index, ups):
	'''
		Compute mean of user_vec for the region
		Works for the signed embedding and color embedding
	'''
	user_vec_regions = np.zeros((len(regions), user_vec.shape[1]))
    
	for r in range(len(regions)):
		for u in regions[r]:
			user = ups[u][1]
			idx_user = user_index[user]

			if scipy.sparse.issparse(user_vec):
				user_vec_regions[r] = user_vec_regions[r] + user_vec[idx_user].todense()
			else:
				user_vec_regions[r] = user_vec_regions[r] + user_vec[idx_user]
            
		user_vec_regions[r] = user_vec_regions[r] / len(regions[r])
            
	return user_vec_regions

def compute_region_colors(regions, ups):
    '''
        Computes sparse matrix of color frequencies
        for each region.
    '''
    rows = []
    cols = []
    data = []
    for r in range(len(regions)):
        region = regions[r]
        colors = np.zeros(16)
        for u in region:
            up = ups[u]
            color = int(up[4])
            colors[color] = colors[color] + 1
        
        colors = colors / np.sum(colors)
        
        for c in range(len(colors)):
            if colors[c] > 0:
                rows.append(r)
                cols.append(c)
                data.append(colors[c])
                
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(regions), 16))


def compute_users_per_region(regions, ups):
    '''
        Computes a list of dictionaries where for 
        each region there is a dictionary with keys
        as users and values as the number of updates
        but that user.
    '''
    users_per_region = []
    
    for region in regions:
        users_per_region.append({})
        for up in region:
            user = ups[up][1]
            
            if user not in users_per_region[-1]:
                users_per_region[-1][user] = 1
            else:
                users_per_region[-1][user] = users_per_region[-1][user] + 1
            
    return users_per_region

def compute_region_bounding_boxes(regions, ups):
    '''
        Computes the bounding box for each region.
    '''
    bounding_boxes = np.zeros((len(regions), 4))
    
    for r in range(len(regions)):
        region = regions[r]
        min_x = 2000
        max_x = 0
        min_y = 2000
        max_y = 0

        for u in region:
            up = ups[u]
            
            x = up[2]
            y = up[3]
        
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
            
        bounding_boxes[r][0] = min_x
        bounding_boxes[r][1] = max_x
        bounding_boxes[r][2] = min_y
        bounding_boxes[r][3] = max_y
        
    return bounding_boxes

def compute_user_vector_regions(regions, user_vec, user_index, ups):
        '''
                Compute mean of user_vec for the region
                Works for the signed embedding and color embedding
        '''
        user_vec_regions = np.zeros((len(regions), user_vec.shape[1]))

        for r in range(len(regions)):
                for u in regions[r]:
                        user = ups[u][1]
                        idx_user = user_index[user]

                        if scipy.sparse.issparse(user_vec):
                                user_vec_regions[r] = user_vec_regions[r] + user_vec[idx_user].todense()
                        else:
                                user_vec_regions[r] = user_vec_regions[r] + user_vec[idx_user]

                user_vec_regions[r] = user_vec_regions[r] / len(regions[r])

        return user_vec_regions

def compute_region_durations(regions, durations, ups):
	'''
		Computes duration (timestamps for begin and end)
		for each region
	'''
	duration_regions = np.zeros((len(regions), 2))
    
	for r in range(len(regions)):
		begin = sys.maxsize
		end = 0
        
		for u in regions[r]:
			up = ups[u]
			begin = min(durations[u][0], begin)
			end = max(durations[u][1], end)
            
		duration_regions[r][0] = begin
		duration_regions[r][1] = end
	
	return duration_regions

def build_feat_label_regions(G, ups, features):
    '''	
        Extracts feature values and labels for edges between regions.

        features is a dictionary with both functions that compute
        feature values and the data structures they require, used as:

        A[i,f] = features[f]['func'](u, v, ups, features[f]['data'])

        Returns matrix A with feature values and vector b with labels
    '''
    #Counting labelled points
    n_labelled = 0
    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = int(r[2])
            type_edge = int(r[3])

            if lb in [0,1]:
                n_labelled = n_labelled + 1

    A = np.zeros((n_labelled+1, len(features)))
    b = np.zeros(n_labelled+1)

    i = 0
    with open(G.unique_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = int(r[2])
            type_edge = int(r[3])

            if lb in [0,1]:
                for f in range(len(features)):
                    A[i,f] = features[f]['func'](u, v, ups, features[f]['data'])

                    if lb == 1:
                        b[i] = float(0.)
                    else:
                        b[i] = float(1.)

                i = i + 1
    return A, b

def compute_region_weight(u, v, m, features):
	'''
		Computes weight of edge (upi,upj) between two regions 
		based on several features.
	'''
	feat_values = np.zeros(len(features))

	for f in range(len(features)):
		feat_values[f] = features[f]['func'](u, v, features[f]['data'])

	return m.predict([feat_values])[0]

def compute_region_edge_weights(G, model, features):
	'''
		Computes weights for edges between regions.
	'''
	if os.path.exists(G.edges_file_name):
		os.remove(G.edges_file_name)

	with open(G.unique_edges_file_name, 'r') as file_in:
		reader = csv.reader(file_in)

		for r in reader:
			u = r[0]
			v = r[1]
			lb = r[2]
			type_edge = int(r[3])

			w = compute_region_weight(int(u), int(v), model, features)
			G.set_weight(u, v, lb, type_edge, w)

	G.flush_weights()

def extract_super_region_info(reg_regions, regions):
    '''
        Extract regions from superpixel segmentation.
    '''
    super_regions = []
    super_region_sizes = []
    super_region_assign = {}

    for sr in reg_regions:
        super_regions.append([])
        for r in sr:
            for u in regions[r]:
                super_regions[-1].append(u)
                super_region_assign[u] = len(super_regions)-1

        super_region_sizes.append(len(super_regions[-1]))

    return super_regions, super_region_sizes, super_region_assign

def create_region_video_files(ups, region, output_name, seconds_per_frame=60):
    '''
        Creates sequence of image files that can be used to produce
        a video of the evolution of the region and saves it to the folder
        given in output_name. 
        
        ffmpeg -framerate 1 -i %d.png -vcodec mjpeg video.avi
    '''
    if not os.path.exists(output_name):
        os.makedirs(output_name)
    
    min_x=1001
    max_x=0
    min_y=1001
    max_y=0
    for u in region:
        up = ups[u]
        x = up[2] 
        y = up[3]
        
        min_x = min(x,min_x)
        min_y = min(y,min_y)
        max_x = max(x,max_x)
        max_y = max(y,max_y)
    
    ups_region = updates_region(ups, region)
    sorted_ups_region = sorted(ups_region, key=lambda x: x[0])
    curr_ups_region = []
    first_tstp = sorted_ups_region[0][0]
    last_tstp = sorted_ups_region[-1][0]
    i = 0
    f = 0
    for t in range(first_tstp, last_tstp, seconds_per_frame * 1000):
        while sorted_ups_region[i][0] <= t:
            curr_ups_region.append(sorted_ups_region[i])
            i = i + 1
        
        plt.close('all')

        canvas_ups = extract_canvas_updates_region(curr_ups_region, min_x, max_x, min_y, max_y)
        canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)

        draw_canvas(canvas_ups_rgb, output_name+"/"+str(f)+".svg")
        f=f+1
        
    plt.close('all')

    canvas_ups = extract_canvas_updates_region(ups_region, min_x, max_x, min_y, max_y)
    canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)

    draw_canvas(canvas_ups_rgb, output_name+"/"+str(f)+".svg")

def region_neighborhood(region, G, ups, features, regions):
    '''
        Prints neighborhood information (edges and feature values)
        for a given region.
    '''
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)

        for r in reader:
            u = int(r[0])
            v = int(r[1])
            lb = int(r[2])
            type_edge = int(r[3])

            if (u == region or v == region) and len(regions[u]) > 50 and len(regions[v]) > 50:
                print(u, ",", v, ",",lb)
                print("size_u=", len(regions[u])," size_v=", len(regions[v]))
                for f in range(len(features)):
                    print(features[f]['name'], " ", features[f]['func'](u, v, ups, features[f]['data']))
                
                print()

def region_statistics(G, region, ups):
    '''
        Prints a few region statistics.
    '''
    #avg, min, max weights
    sum_weights_in = 0
    min_weight_in = 1e10
    max_weight_in = 0
    
    sum_weights_out = 0
    min_weight_out = 1e10
    max_weight_out = 0
    n_in = 0
    n_out = 0
    
    n_edges = 0
    region_check = {}
    
    for u in region:
        region_check[u] = 1
    
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            w = float(r[4])
            
            if u in region_check and v in region_check:
                n_edges = n_edges + 1
                sum_weights_in = sum_weights_in + w
                n_in = n_in + 1
            
                if w < min_weight_in:
                    min_weight_in = w
                
                if w > max_weight_in:
                    max_weight_in = w
                    
            elif u in region_check or v in region_check:
                sum_weights_out = sum_weights_out + w
                n_out = n_out + 1
            
                if w < min_weight_out:
                    min_weight_out = w
                
                if w > max_weight_out:
                    max_weight_out = w
    
    color_dist = np.zeros(16)
    n_pixel = 0
    n_pixel_color = 0
    min_time = sys.maxsize
    max_time = 0
    projs = {}
    for u in region:
        if ups[u][6] == 1:
            n_pixel = n_pixel + 1
            if ups[u][5] not in projs:
                projs[ups[u][5]] = 0
            
            projs[ups[u][5]] = projs[ups[u][5]] + 1
        if ups[u][7] == 1:
            n_pixel_color = n_pixel_color + 1
            
        if ups[u][0] > max_time:
            max_time = ups[u][0]
            
        if ups[u][0] < min_time:
            min_time = ups[u][0]
        
        color_dist[int(ups[u][4])] = color_dist[int(ups[u][4])] + 1
    
    print("num updates = ", len(region))
    print("num_edges = ", n_edges)
    print("num pixels = ", n_pixel, " (", 100 * n_pixel / len(region), "%)")
    print("num_pixel_colors = ", n_pixel_color, " (", 100 * n_pixel / len(region), "%)")
    
    print()
    
    sorted_projs = sorted(projs.items(), key=operator.itemgetter(1), reverse=True)
    
    for i in range(len(sorted_projs)):
        print("proj ", names[int(sorted_projs[i][0])], " #final updates = ", sorted_projs[i][1], " (",
              100 * sorted_projs[i][1] / n_pixel, "% of final, ", 100 * sorted_projs[i][1] / len(region), " of total)")
    
    print()
    
    print("colors: ", list(color_dist))
    
    print()
    
    print("duration: ", min_time, " - ", max_time, " (", (max_time-min_time)/ 1000, " seconds)")
        
    print()
    
    print("avg weight inside = ",  sum_weights_in / n_in)
    print ("min weight inside = ", min_weight_in)
    print("max weight inside = ", max_weight_in)
    print("#edges inside = ", n_in)
    
    print()
    
    print("avg weight outside = ",  sum_weights_out / n_out)
    print ("min weight outside = ", min_weight_out)
    print("max weight outside = ", max_weight_out)
    print("#edges outside = ", n_out) 

def project_statistics(G, comp_assign, ups, proj):
    '''
        Prints a few project statistics.
    '''
    #avg, min, max weights
    sum_weights_in = 0
    min_weight_in = 1e10
    max_weight_in = 0
    
    sum_weights_out = 0
    min_weight_out = 1e10
    max_weight_out = 0
    n_in = 0
    n_out = 0
    
    n_edges = 0
    regions = {}
    G_proj = nx.Graph()
    
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            w = float(r[4])
            
            if ups[u][6] == 1 and ups[v][6] == 1:
                if ups[u][5] == proj and ups[v][5] == proj:
                    G_proj.add_edge(u,v)
                    n_edges = n_edges + 1
                    sum_weights_in = sum_weights_in + w
                    n_in = n_in + 1

                    if w < min_weight_in:
                        min_weight_in = w

                    if w > max_weight_in:
                        max_weight_in = w

            elif (ups[u][6] == 1 and ups[u][5] == proj) or (ups[u][6] == 1 and ups[v][5] == proj):
                sum_weights_out = sum_weights_out + w
                n_out = n_out + 1

                if w < min_weight_out:
                    min_weight_out = w

                if w > max_weight_out:
                    max_weight_out = w
                    
    size_proj = 0
    for u in range(len(ups)):
        if ups[u][6] == 1 and ups[u][5] == proj:
            size_proj = size_proj + 1
            if comp_assign[u] not in regions:
                regions[comp_assign[u]] = 0
                
            regions[comp_assign[u]] = regions[comp_assign[u]] + 1
            
    region_sizes = {}
    
    for r in regions:
        region_sizes[r] = 0
        
    for u in range(len(ups)):
        if comp_assign[u] in regions and ups[u][6] == 1:
            region_sizes[comp_assign[u]] = region_sizes[comp_assign[u]] + 1
    
    sorted_regions = sorted(regions.items(), key=operator.itemgetter(1), reverse=True)
    
    for i in range(len(sorted_regions)):
        print("region ", sorted_regions[i][0], " #updates = ", sorted_regions[i][1], " (",
              100 * sorted_regions[i][1] / size_proj, "% of project, ", 
              100 * sorted_regions[i][1] / region_sizes[sorted_regions[i][0]], "% of region)")
    
    print()
    
    print("avg weight inside = ",  sum_weights_in / n_in)
    print ("min weight inside = ", min_weight_in)
    print("max weight inside = ", max_weight_in)
    print("#edges inside = ", n_in)
    
    print()
    
    print("avg weight outside = ",  sum_weights_out / n_out)
    print ("min weight outside = ", min_weight_out)
    print("max weight outside = ", max_weight_out)
    print("#edges outside = ", n_out)
    
    print("Graph connected : ", nx.is_connected(G_proj))
    print("Largest connected component: ", 100 * max((G_proj.subgraph(c) for c in nx.connected_components(G_proj)), key=len).number_of_nodes() / size_proj, "%")

def weight_statistics(G, ups, projs):
    '''
        Computes weight statistics for the weights within 
        and across projects.
    '''
    
    sum_in_final = 0
    sum_out_final = 0
    n_in_final = 0
    n_out_final = 0
    
    
    sum_in_color = 0
    sum_out_color = 0
    n_in_color = 0
    n_out_color = 0
    
    with open(G.sorted_edges_file_name, 'r') as file_in:
        reader = csv.reader(file_in)
    
        for r in reader:
            u = int(r[0])
            v = int(r[1])
            w = float(r[4])
            type_edge = int(r[3])
            proj_u = ups[u][5]
            proj_v = ups[v][5]
            
            if type_edge > 0:
                if projs is None or (proj_u in projs and proj_v in projs):
                    if ups[u][6] == 1 and ups[v][6] == 1:
                        if ups[u][5] == ups[v][5]:
                            sum_in_final = sum_in_final + w
                            n_in_final = n_in_final + 1
                        else:
                            sum_out_final = sum_out_final + w
                            n_out_final = n_out_final + 1

                    if ups[u][7] == 1 and ups[v][7] == 1:
                        if ups[u][5] == ups[v][5]:
                            sum_in_color = sum_in_color + w
                            n_in_color = n_in_color + 1
                        else:
                            sum_out_color = sum_out_color + w
                            n_out_color = n_out_color + 1
                            
                    
                    
    print("avg weight inside projects (pixel) = ", sum_in_final / n_in_final)
    print("avg weight outside projects (pixel) = ", sum_out_final / n_out_final)
    print("avg weight (pixel) = ", (sum_in_final + sum_out_final) / (n_in_final + n_out_final) )
    
    print()
    
    print("avg weight inside projects (color) = ", sum_in_color / n_in_color)
    print("avg weight outside projects (color) = ", sum_out_color / n_out_color)
    print("avg weight (color) = ", (sum_in_color + sum_out_color) / (n_in_color + n_out_color) )
    
