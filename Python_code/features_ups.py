import numpy as np
import time
import pickle
from segmentation import *


#Functions that compute edge features for the
#update graph. The functions receive indexes
#of a pair of updates and some necessary data
#and return a real value.

def different_color(i, j, ups, data=None):
    '''
        Simply checks if updates have different color.
    '''
    if ups[i][4] == ups[j][4]:
        return 0
    else:
        return 1


def distance_space(i, j, ups, data=None):
    '''
        Eclidean distance between updates.
    '''
    xi = ups[i][2]
    yi = ups[i][3]
    xj = ups[j][2]
    yj = ups[j][3]

    return np.sqrt(pow(xi-xj, 2)+pow(yi-yj, 2))


def distance_time(i, j, ups, data=None):
    '''
        Time distance between updates
        in hours.
    '''
    time_i = ups[i][0]
    time_j = ups[j][0]

    return np.abs(time_i-time_j) / 3600000  # hours


def distance_duration(i, j, ups, durations):
    '''
        Distance between duration of updates.
        See function dist_duration for details.
    '''
    return dist_duration(durations[i], durations[j])


def distance_color(i, j, ups, conflicts):
    '''
        Computes the distance between two colors
        based on how often one has replaced the
        other in that particular position (x,y).
    '''
    color_i = ups[i][4]
    color_j = ups[j][4]

    if color_i == color_j:
        return 0
    else:
        max_up = len(ups)
        dist = 0

        conf_i = []
        if conflicts[i][0] <= max_up:
            conf_i.append(ups[conflicts[i][0]][4])

        if conflicts[i][1] <= max_up:
            conf_i.append(ups[conflicts[i][1]][4])

        conf_j = []
        if conflicts[j][0] <= max_up:
            conf_j.append(ups[conflicts[j][0]][4])

        if conflicts[j][1] <= max_up:
            conf_j.append(ups[conflicts[j][1]][4])

        if color_i in conf_j:
            dist = dist + 1

        if color_j in conf_i:
            dist = dist + 1

        return dist


def distance_user_embedding(i, j, ups, data):
    '''
        Euclidean distance between user embeddings.
    '''
    user_i = ups[i][1]
    user_j = ups[j][1]
    user_i_id = data['index'][user_i]
    user_j_id = data['index'][user_j]

    return np.linalg.norm(data['emb'][user_i_id]-data['emb'][user_j_id])


def distance_user_colors(i, j, ups, data):
    '''
        Distance between user color histograms.
        One minus sum of minimum values for each
        color.
    '''
    user_i = ups[i][1]
    user_j = ups[j][1]
    user_i_id = data['index'][user_i]
    user_j_id = data['index'][user_j]

    return 1.-data['emb'][user_i_id].minimum(data['emb'][user_j_id]).sum()


def create_features(G_ups, ups, ndim, threshold, total_samples, n_negatives, n_iterations, features_filename='features.pkl'):
    #Prepares data for feature computation and saves it.
    #Takes a long time.

    t = time.time()

    conflicts = compute_update_conflicts(ups)
    durations = compute_update_durations(ups)
    user_color, user_index_color = compute_user_color(ups)

    user_index, emb = embed_users(G_ups, ups, ndim, threshold, total_samples, n_negatives, n_iterations, True)

    features = [{'name': "different_color", 'func': different_color, 'data': None}, 
        {'name': "distance_space",  'func': distance_space, 'data': None}, 
        {'name': "distance_time", 'func': distance_time, 'data': None}, 
        {'name': "distance_color", 'func': distance_color, 'data': conflicts},
        {'name': "distance_user_embedding", 'func': distance_user_embedding, 'data': {'index': user_index, 'emb': emb}},
        {'name': "distance_user_colors", 'func': distance_user_colors, 'data': {'index': user_index_color, 'emb': user_color}}]

    pfile = open(features_filename, 'wb')
    pickle.dump(features, pfile)
    pfile.close()

    print("time to create G_ups features= ", time.time()-t, " seconds")

    return features
