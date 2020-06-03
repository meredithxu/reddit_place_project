import numpy as np
import pickle
import math
import time

from segmentation import *

#Functions that compute edge features for the
#region graph. The functions receive indexes
#of a pair of regions and some necessary data
#and return a real value.


def difference_internal_weights(i, j, ups, data):
    '''
        Internal weights measure the diversity of
        of the edges in a region. This computes
        their difference.
    '''
    return np.abs(data[i]-data[j])


def sum_internal_weights(i, j, ups, data):
    '''
        Internal weights measure the diversity of
        of the edges in a region. This computes
        their sum.
    '''
    return data[i]+data[j]


def distance_region_user_embedding(i, j, ups, data):
    '''
        Distance between avg user embeddings for regions.
    '''
    return np.linalg.norm(data[i]-data[j])


def distance_region_user_colors(i, j, ups, data):
    '''
        Distance between avg user color embeddings for regions.
    '''
    return 1.-np.minimum(data[i], data[j]).sum()


def distance_region_durations(i, j, ups, data):
    '''
        Distance between the durations of two regions.
    '''
    return dist_duration(data[i], data[j])


def distance_users_per_region(i, j, ups, data):
    '''
        Distance between user participations between regions.
        It is weighted by user activity.
    '''
    sz = min(len(data['regions'][i]), len(data['regions'][j]))
    dist = sz
    for u in data['users'][i]:
        if u in data['users'][j]:
            dist = dist - min(data['users'][i][u], data['users'][j][u])

    return dist / sz


def superposition(i, j, ups, data):
    '''
        Superposition between two regions, which is a 
        normalized count of the number of updates in 
        the same position but with different colors
        in the final version of the region (as there
        can be layers).
    '''
    pos_i = {}
    for ui in data[i]:
        up = ups[ui]
        x = up[2]
        y = up[3]
        t = up[0]
        c = up[4]

        if (x, y) not in pos_i:
            pos_i[(x, y)] = (c, t)
        elif pos_i[(x, y)][1] < t:
            pos_i[(x, y)] = (c, t)

    pos_j = {}
    for uj in data[j]:
        up = ups[uj]
        x = up[2]
        y = up[3]
        t = up[0]
        c = up[4]

        if (x, y) not in pos_j:
            pos_j[(x, y)] = (c, t)
        elif pos_j[(x, y)][1] < t:
            pos_j[(x, y)] = (c, t)

    count = 0
    for (x, y) in pos_i:
        if (x, y) in pos_j:
            if pos_i[(x, y)][0] != pos_j[(x, y)][0]:
                count = count + 1

    return count / min(len(pos_i), len(pos_j))


def avg_update_time_distance(i, j, ups, data):
    '''
        Computes (absolute) distance between the
        average time of two regions.
    '''
    avg_ti = 0
    for ui in data[i]:
        upi = ups[ui]
        avg_ti = avg_ti + upi[0]

    avg_ti = avg_ti / len(data[i])

    avg_tj = 0
    for uj in data[j]:
        upj = ups[uj]
        avg_tj = avg_tj + upj[0]

    avg_tj = avg_tj / len(data[j])

    return np.abs(avg_ti-avg_tj) / 3600000  # hours


def avg_update_pos_distance(i, j, ups, data):
    '''
        Computes the distance between the average
        points of two regions.
    '''
    avg_xi = 0
    avg_yi = 0

    for ui in data[i]:
        upi = ups[ui]
        xi = upi[2]
        yi = upi[3]

        avg_xi = avg_xi + xi
        avg_yi = avg_yi + yi

    avg_xi = avg_xi / len(data[i])
    avg_yi = avg_yi / len(data[i])

    avg_xj = 0
    avg_yj = 0

    for uj in data[j]:
        upj = ups[uj]
        xj = upj[2]
        yj = upj[3]

        avg_xj = avg_xj + xj
        avg_yj = avg_yj + yj

    avg_xj = avg_xj / len(data[j])
    avg_yj = avg_yj / len(data[j])

    return math.sqrt(math.pow(avg_xi-avg_xj, 2)+math.pow(avg_yi-avg_yj, 2))


def sizes(i, j, ups, data):
    '''
    '''
    return len(data[i])+len(data[j])


def bounding_box_distance(i, j, ups, data):
    '''
        Amount of empty space in the bounding box
        of the combined region.
    '''
    min_xi = data[i][0]
    max_xi = data[i][1]
    min_yi = data[i][2]
    max_yi = data[i][3]

    min_xj = data[j][0]
    max_xj = data[j][1]
    min_yj = data[j][2]
    max_yj = data[j][3]

    joint_area = (max(max_xi, max_xj)-min(min_xi, min_xj)+1) * \
        (max(max_yi, max_yj)-min(min_yi, min_yj)+1)
    inters_area = (min(max_xi, max_xj)-max(min_xi, min_xj)+1) * \
        (min(max_yi, max_yj)-max(min_yi, min_yj)+1)
    inters_area = max(0, inters_area)

    area_i = (max_xi-min_xi+1) * (max_yi-min_yi+1)
    area_j = (max_xj-min_xj+1) * (max_yj-min_yj+1)

    return (joint_area - area_i - area_j + inters_area) / joint_area


def distance_region_colors(i, j, ups, data):
    '''
        Distance between distributions of user colors
        in the regions.
    '''
    return 1.-data[i].minimum(data[j]).sum()


def create_superfeatures(regions, int_weights, ups, features, durations, filename):
    #Prepares data for feature computation and saves it.
    t = time.time()

    region_durations = compute_region_durations(regions, durations, ups)
    users_per_region = compute_users_per_region(regions, ups)

    emb = features[4]['data']['emb']
    user_index = features[4]['data']['index']

    region_sign_emb = compute_user_vector_regions(regions, emb, user_index, ups)

    user_color = features[5]['data']['emb']
    user_index_color = features[5]['data']['index']

    region_colors_emb = compute_user_vector_regions(regions, user_color, user_index_color, ups)

    region_colors = compute_region_colors(regions, ups)

    bounding_boxes = compute_region_bounding_boxes(regions, ups)

    region_features = [{'name': "distance_user_embedding", 'func': distance_region_user_embedding, 'data': region_sign_emb},
        {'name': "users_per_region", 'func': distance_users_per_region, 'data':{'users': users_per_region, 'regions':regions}}, 
        {'name': "superposition", 'func': superposition, 'data':regions},
        {'name': "pos_distance", 'func': avg_update_pos_distance, 'data':regions},
        {'name': "sizes", 'func': sizes, 'data':regions},
        {'name': "bounding_box", 'func': bounding_box_distance, 'data':bounding_boxes},
        {'name': "dist_region_color", 'func': distance_region_colors, 'data':region_colors},
        {'name': 'avg_update_distance', 'func': avg_update_pos_distance, 'data':regions},
        {'name': 'dist_user_colors', 'func': distance_region_user_colors, 'data':region_colors_emb},
        {'name': 'dist_region_durations', 'func': distance_region_durations, 'data':region_durations},
        {'name': 'diff_internal_weight', 'func': difference_internal_weights, 'data':int_weights},
        {'name': 'sum_internal_weight', 'func': sum_internal_weights, 'data':int_weights}
    ]

    pfile = open(filename, 'wb')
    pickle.dump(region_features, pfile)
    pfile.close()

    print("time to create region features= ", time.time()-t, " seconds")

    return region_features
