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

from user_embedding import *

class MyGraph:
    '''
        This is an undirected graph class for processing edge lists 
        as files. It is suitable for very large number of edges that 
        might not fit in memory.
    '''
    def __init__(self):
        '''
        '''
        self._edges = {}
        self.buffer_size = 10000000                            #Keeps at most 10M edges in memory
        self.n_edges = 0
        
        #File names used by this class
        self.edges_file_name = "./edges.csv"
        self.unique_edges_file_name = "./unique_edges.csv"
        self.sorted_edges_file_name = "./sorted_edges.csv"
        
        if os.path.exists(self.edges_file_name):
            os.remove(self.edges_file_name)
            
        if os.path.exists(self.unique_edges_file_name):
            os.remove(self.unique_edges_file_name)
        
        if os.path.exists(self.sorted_edges_file_name):
            os.remove(self.sorted_edges_file_name)
    
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
                min_x=0, max_x=1002, min_y=0, max_y=1002):
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
            #if proj_smallest and pixel == 1:
            if proj_smallest == 1:
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
    
    
    
    G = MyGraph()
    
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

def compute_user_relations(G, ups, rel_type=1):
    '''
        Computes two matrices based on user neighborhood activity. 
        There are three types of user relations, it is not clear which 
        one is best:
        
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
        Computes a color vector for each user.
        For each color, we add 1 if the user uses the color and
        subtract 1 if that color replaces an update of the user
        or the user changes that color via an update. The vectors
        are normalized (norm 1.)
    '''
    max_uint32 = 4294967295    #Using this for undefined
    xy_user = max_uint32 * np.ones((1001,1001),dtype=np.uint32)
    xy_color = -np.ones((1001,1001),dtype=np.int8)
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

        color_earlier = xy_color[x][y]
        user_earlier = xy_user[x][y]

        if user_earlier < max_uint32 and user_earlier != idx_user and color != color_earlier:
            user_color[user_earlier][color] = user_color[user_earlier][color] - 1

        if color_earlier >= 0 and color != color_earlier:
            user_color[idx_user][color_earlier] = user_color[idx_user][color_earlier] - 1

        xy_color[x][y] = color
        xy_user[x][y] = idx_user

    for i in range(user_color.shape[0]):
        if np.linalg.norm(user_color[i]) > 0:
            user_color[i] = user_color[i] / np.linalg.norm(user_color[i])

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
            w = float(r[4])

            if type_edge > 0:
                comp_u = comp_assign[u]
                comp_v = comp_assign[v]
        
                if comp_u != comp_v:
                    m_int = compute_m_int(int_weights, sizes, comp_u, comp_v, KAPPA)
            
                    if w < m_int:
                        comp_assign, int_weights, sizes, regions = merge(G, comp_assign, int_weights, sizes, regions,
                            comp_u, comp_v, w)
        
    return comp_assign

def extract_regions(comp_assign):
    '''
        Extracts actual regions (lists of updates) from region assignements.
        Also returns the region sizes.
    '''
    regions = []
    sizes = []
    region_ids = {}
    
    for c in comp_assign:
        if comp_assign[c] not in region_ids:
            region_ids[comp_assign[c]] = len(region_ids)
            regions.append([])
            sizes.append(0)
            
    for c in comp_assign:
        ID = region_ids[comp_assign[c]]
        regions[ID].append(c)
        sizes[ID] = sizes[ID] + 1
        
    for r in regions:
        r.sort()
    
    return regions, sizes

def assign_pixels(comp_assign, ups):
    assign = {}
    for c in comp_assign:
        if ups[c][6] == 1:
            assign[c] = comp_assign[c]
    return assign

def regions_with_proj(ups, regions, proj):
    rej_proj = []
    sizes = []
    for r in regions:
        c = 0
        for u in r:
            if ups[u][5] == proj and ups[u][7] == 1:
                c = c + 1
                
        if c > 0:
            rej_proj.append(r)
            sizes.append(c)
                
    return rej_proj, sizes

def extract_region(updates, region):
    data_color_code = np.uint8(np.zeros((1001,1001)))
    
    for pixel in region:
        u = updates[int(pixel)]
        x = u[2]
        y = u[3]
        color = u[4]
        
        data_color_code[y][x] = color
        
    return data_color_code

def extract_ground_truth(truth):
    
    data_color_code = np.uint8(np.zeros((1001,1001)))
    
    for key in truth:
        x = key[0]
        y = key[1]
        color = key[2]
        
        data_color_code[y][x] = color
        
    return data_color_code

def draw_canvas_region(updates, region, out_file, is_ground_truth = False):
    if is_ground_truth:
        canvas_ups = extract_ground_truth(region)
    else:
        canvas_ups = extract_region(updates, region)
    
    canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)
    draw_canvas(canvas_ups_rgb, out_file)

def extract_canvas_updates(updates):
    data_color_code = np.uint8(np.zeros((1001,1001)))
    
    for u in updates:
        x = u[2]
        y = u[3]
        color = u[4]
        
        data_color_code[y][x] = color
        
    return data_color_code


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


def build_feat_label_data(unique_edges_file_name, ups, features, pixel=False, train_x_y=None, fold_boundaries=None, excluded_folds=None):
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
    with open(unique_edges_file_name, 'r') as file_in:
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
    with open(unique_edges_file_name, 'r') as file_in:
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

def compute_weight(edge_buffer, ups, m, features):
    '''
        Computes weight of edge (upi,upj) based on several features.
    '''
    feat_values = np.zeros((len(edge_buffer), len(features)))
    
    for e in range(len(edge_buffer)):
        u = edge_buffer[e][0]
        v = edge_buffer[e][1]

        for f in range(len(features)):
            feat_values[e][f] = features[f]['func'](u, v, ups, features[f]['data'])
        
    return m.predict(feat_values)

def compute_weight_wrapper(param):
    '''
        Simple wrapper for the compute_weight function
    '''    
    #Loading pickled features
    #Each thread has its own copy, which is quite inneficient
    pfile = open('features.pkl', 'rb')
    features = pickle.load(pfile)
    pfile.close()

    res = compute_weight(param[0], param[1], param[2], features)

    return res

def compute_weight_multithread(edge_buffer, ups, model, n_threads):
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
            fut = executor.submit(compute_weight_wrapper, (edge_parts[t], ups, model))
            futures.append(fut)

    #Collecting results
    W = np.zeros(len(edge_buffer))
    for t in range(n_threads):
        fut = futures[t]
        res = fut.result()
        for e in range(res.shape[0]):
            W[e*n_threads+t] = res[e]

    return W

def compute_edge_weights_multithread(G, ups, model, features, n_threads):
    '''
        Computes weights for edges in the graph using multithreading.
    '''

    if os.path.exists(G.edges_file_name):
        os.remove(G.edges_file_name)
    
    #Pickling feature data to be shared with threads
    if not os.path.exists('features.pkl'):
         pfile = open('features.pkl', 'wb')
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

                    W = compute_weight_multithread(edge_buffer, ups, model, n_threads)

                    for e in range(len(edge_buffer)):
                        u = edge_buffer[e][0]
                        v = edge_buffer[e][1]
                        lb = edge_buffer[e][2]
                        type_edge = edge_buffer[e][3]
                        w = W[e]

                        G.set_weight(u, v, lb, type_edge, w)

                    edge_buffer = []

    if len(edge_buffer) > 0:
        W = compute_weight_multithread(edge_buffer, ups, model, n_threads)

        for e in range(len(edge_buffer)):
            u = edge_buffer[e][0]
            v = edge_buffer[e][1]
            lb = edge_buffer[e][2]
            type_edge = edge_buffer[e][3]
            w = W[e]

            G.set_weight(u, v, lb, type_edge, w)

    G.flush_weights()

def compute_edge_weights(G, ups, model, features):
    '''
        Computes weights for edges in the graph.
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
                    
                    W = compute_weight(edge_buffer, ups, model, features)
                    
                    for e in range(len(edge_buffer)):
                        u = edge_buffer[e][0]
                        v = edge_buffer[e][1]
                        lb = edge_buffer[e][2]
                        type_edge = edge_buffer[e][3]
                        w = W[e]
                        
                        G.set_weight(u, v, lb, type_edge, w)
		
                    edge_buffer = []
                
    if len(edge_buffer) > 0:
        W = compute_weight(edge_buffer, ups, model, features)
                    
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

def embed_users(G, ups, ndim=40, threshold=5, total_samples=100, n_negatives=5, n_iterations=10):
    '''
        Embeds users using signed embedding algorithm.
        Edges between users (u,v) can be either negative 
        or positive depending on whether they mostly write
        same color updates nearby each other of overwrite
        each others' updates with different colors.
    '''
    usr_rel_same_color, usr_rel_diff_color, user_index = compute_user_relations(G, ups, rel_type=2)

    sign_embedding(usr_rel_same_color, usr_rel_diff_color, user_index, "signet", ndim, threshold, 
	total_samples, n_negatives, n_iterations)

    emb = read_embedding("signet", user_index, ndim)

    avg_distances_pos_neg(emb)
    
    return user_index, emb

def updates_region(ups, region):
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
    ups_region = updates_region(ups, region)
        
    plt.close('all')

    canvas_ups = extract_canvas_updates(ups_region)
    canvas_ups_rgb = canvas_color_code_rgb(canvas_ups)

    draw_canvas(canvas_ups_rgb, output_file_name)

def draw_top_regions(regions, sizes, k, output_file_name='../plots/top_region'):
    '''
        Draws top k largest regions
    '''
    ind  = np.argsort(sizes)
    
    for i in range(1, k):
        r = ind[-i]
        
        draw_region(ups, regions[r], output_file_name+"_"+str(i)+".svg")
