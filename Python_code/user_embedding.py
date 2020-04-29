import csv
import numpy as np
import scipy
import networkx as nx
import math
import sys
import os
import subprocess
import scipy.stats
import scipy.optimize
import operator
from sklearn.manifold import TSNE

signet_path = "../../signet/signet.py"

def generate_user_signed_net(usr_rel_same_color, usr_rel_diff_color, user_index, output_file_name):
    '''
        Writes signed network edges based on user updates to an output file. usr_rel_same_color 
        has the positive relations and usr_rel_diff_color are negative relations. Edges are added 
        according to the majority of the relations between pairs of users.
    '''
    with open(output_file_name+".txt", 'w') as file_out:
        writer = csv.writer(file_out, delimiter = "\t")

        for u, v in zip(*usr_rel_same_color.nonzero()):
            if u < v:
                if usr_rel_same_color[u,v] > usr_rel_diff_color[u,v]:
                    writer.writerow([u+1, v+1, 1])
                else:
                    writer.writerow([u+1, v+1, -1])

        for u, v in zip(*usr_rel_diff_color.nonzero()):
            if u < v and usr_rel_same_color[u,v] == 0:
                if usr_rel_same_color[u,v] > usr_rel_diff_color[u,v]:
                    writer.writerow([u+1, v+1, 1])
                else:
                    writer.writerow([u+1, v+1, -1])

    with open(output_file_name+"_id.txt", 'w') as file_out:
        writer = csv.writer(file_out, delimiter = "\t")
        for ID in user_index:
            writer.writerow([ID])

def generate_nx_signed_net(usr_rel_same_color, usr_rel_diff_color, user_index, filter_users=None):
    '''
        Generates signed networkx network based on user relations.
        See details in function generate_user_signed_net.
    '''
    G = nx.Graph()
    reverse_index = {}

    if filter_users != None:
        for user in filter_users:
            reverse_index[user_index[user]] = 1

    for u, v in zip(*usr_rel_same_color.nonzero()):
        if u < v:
            if filter_users is None or (u in reverse_index and v in reverse_index):
                if usr_rel_same_color[u,v] > usr_rel_diff_color[u,v]:
                    G.add_edge(u,v,sign=1)
                else:
                    G.add_edge(u,v,sign=-1)

    for u, v in zip(*usr_rel_diff_color.nonzero()):
        if u < v and usr_rel_same_color[u,v] == 0:
            if filter_users is None or (u in reverse_index and v in reverse_index):
                if usr_rel_same_color[u,v] > usr_rel_diff_color[u,v]:
                    G.add_edge(u,v,sign=1)
                else:
                    G.add_edge(u,v,sign=-1)

    return G

def balance(G):
    '''
        Computes balance statistics for the network.
        Based on:
        https://stackoverflow.com/questions/52582733/how-to-find-balanced-triangles-and-unbalanced-triangles-of-a-signed-network-us
    '''
    ppp = 0    #balanced
    ppm = 0    #not balanced
    pmm = 0    #balanced
    mmm = 0    #not balanced
    neg_edges = 0
    pos_edges = 0

    for i in G.nodes():
        for j in G.neighbors(i):
            if i < j:
                sign_ij = G[i][j]['sign']

                if sign_ij > 0:
                    pos_edges = pos_edges + 1
                else:
                    neg_edges = neg_edges + 1

                for k in nx.common_neighbors(G, i, j):
                    if j < k:
                        sign_ik = G[i][k]['sign']
                        sign_jk = G[j][k]['sign']

                        s = sign_ij + sign_jk + sign_ik

                        if s == 3:
                            ppp = ppp + 1
                        elif s == 1:
                            ppm = ppm + 1
                        elif s == -1:
                            pmm = pmm + 1
                        else:
                            mmm = mmm + 1

    total = ppp + ppm + pmm + mmm
    ppp = ppp / total
    ppm = ppm / total
    pmm = pmm / total
    mmm = mmm / total

    total = pos_edges + neg_edges
    pos_edges = pos_edges / total
    neg_edges = neg_edges / total

    print("balanced:")
    print("+++ ", ppp, " rand = ", scipy.stats.binom.pmf(3,3,pos_edges))
    print("+-- ", pmm, " rand = ", scipy.stats.binom.pmf(1,3,pos_edges))

    print("unbalanced:")
    print("++- ", ppm, " rand = ", scipy.stats.binom.pmf(2,3,pos_edges))
    print("--- ", mmm, " rand = ", scipy.stats.binom.pmf(0,3,pos_edges))

    return ppp, ppm, pmm, mmm

def sign_embedding(usr_rel_same_color, usr_rel_diff_color, user_index, output_file_name, 
    ndim=2, threshold=5, total_samples=100, n_negatives=5, 
    n_iterations=10, compute_balance=False):
    '''
        Computes signed embedding based on user relations using the algorithm from: 
        SIGNet: Scalable Embeddings for Signed Networks, code from: 
        https://github.com/raihan2108/signet
    '''
    generate_user_signed_net(usr_rel_same_color, usr_rel_diff_color, user_index, output_file_name)
    print("python "+signet_path+" -l signet_id.txt -i signet.txt -o "+output_file_name+ " -d "+str(ndim)+" -t "+str(threshold)+" -s "+str(total_samples))

    os.system("python "+signet_path+" -l signet_id.txt -i signet.txt -o "+output_file_name+ " -d "+str(ndim)+" -t "+str(threshold)+" -s "+str(total_samples))

    if compute_balance is True:
        sign_G = generate_nx_signed_net(usr_rel_same_color, usr_rel_diff_color, user_index)
        res = balance(sign_G)

def read_embedding(input_file_name, user_index, ndim=2):
    '''
        Reads embedding results from input file as a matrix.
    '''
    emb = np.zeros((len(user_index), ndim))

    with open(input_file_name,'r') as file:
        reader = csv.reader(file)
        i = 0

        for r in reader:
            emb[i] = np.array(r)

            i = i + 1

    return emb

def avg_distances_pos_neg(emb):
    '''
        Computes the avg Eucliden distance between
        users with positive and negative edges.
    '''
    n_pos = 0
    n_neg = 0
    sum_pos = 0.
    sum_neg = 0.
    with open("signet.txt",'r') as file:
        # Skip first line (header row)
        reader = csv.reader(file, delimiter = "\t")

        for r in reader:
            u = int(r[0])
            v = int(r[1])
            sign = int(r[2])

            if sign > 0:
                n_pos = n_pos + 1
                sum_pos = sum_pos + np.linalg.norm(emb[u-1]-emb[v-1])
            else:
                n_neg = n_neg + 1
                sum_neg = sum_neg + np.linalg.norm(emb[u-1]-emb[v-1])

    if n_pos > 0:
    	avg_pos = sum_pos / n_pos
    else:
    	avg_pos = 0
    
    print("avg pos = ", avg_pos, ", n = ", n_pos)
	
    if n_neg > 0:
        avg_neg = sum_neg / n_neg
    else:
        avg_neg = 0
    
    print("avg neg = ", avg_neg, ", n = ", n_neg)

    return avg_pos, avg_neg

def top_users_updates(ups, k):
    '''
        Extracts top k users with the most updates.
    '''
    updates_per_user = {}
    top_users = {}
    
    for u in range(len(ups)):
        user = ups[u][1]
        
        if user not in updates_per_user:
            updates_per_user[user] = 1
        else:
            updates_per_user[user] = updates_per_user[user] + 1
    
    sorted_up_user = sorted(updates_per_user.items(), key=operator.itemgetter(1), reverse=True)

    for i in range(k):
        user = sorted_up_user[i][0]
        ups = sorted_up_user[i][1]
        
        top_users[user] = ups
        
    return top_users

def users_in_projects(ups, projs):
    '''
        Extracts users in a set of projects.
    '''
    users = {}
    n_ups = 0
    for u in range(len(ups)):
        user = ups[u][1]
        proj = ups[u][5]
        pixel_color = ups[u][7]
        
        if proj in projs and pixel_color == 1:
            if user not in users:
                users[user] = np.zeros(len(projs))
                
            users[user][projs.index(proj)] = users[user][projs.index(proj)] + 1
            n_ups = n_ups + 1
            
    return users, n_ups
