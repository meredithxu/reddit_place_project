import sys
import os
import csv
import numpy as np
from reddit import *
from segmentation import *
from nonlinear_regressor import *
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import concurrent.futures


def create_folds(min_x=0, min_y=0, max_x=1002, max_y=1002):
    # Partition the data into folds

    num_folds = 10
    folds = []
    for i in range(num_folds):
        folds.append([])

    halfway_x = int((min_x + max_x) // 2)
    y_increment = int((min_y + max_y) // 5)

    for j in range(5):

        for x in range(0, halfway_x):
            for y in range(j * y_increment, (j + 1) * y_increment):
                folds[j].append((x, y))

        for x in range(halfway_x, max_x):
            for y in range(j * y_increment, (j + 1) * y_increment):
                folds[5 + j].append((x, y))

    return folds


def build_and_evaluate_model(ups, features, pid, unique_edges_file_name, fold_boundaries, excluded_folds, min_x=0, min_y=0, max_x=1002, max_y=1002, kappa=0.25, file_prefix = ""):
    '''
        Build a model and return the evaluation metric
    '''
    locations = store_locations("../data/atlas_complete.json")
    
    # All edges that belong to the validation fold need to be excluded
    A, b = build_feat_label_data(unique_edges_file_name, ups, features,
                                 fold_boundaries=fold_boundaries, excluded_folds=excluded_folds)

    # edges_to_exclude = fold_edges[i]
    # A_training = np.delete(A, edges_to_exclude, axis=0)
    # b_training = np.delete(b, edges_to_exclude, axis=0)

    model = createNonlinearRegressionNeuralNet(A, b)
    # model = GradientBoostingRegressor(random_state=1, n_estimators=25).fit(A, b)

    del A
    del b
    
    # Save the model in a pickle file
    model_name = str(file_prefix) +  str(pid) + "_model.pkl"
    if os.path.exists(model_name):
        os.remove(model_name)
        
        
    pfile = open(model_name, 'wb')
    pickle.dump(model, pfile)
    pfile.close()

    return model_name

def build_and_evaluate_model_wrapper(params):
     #Loading pickled features
    #Each thread has its own copy, which is quite inneficient
    pfile = open('features.pkl', 'rb')
    features = pickle.load(pfile)
    pfile.close()

    pfile = open('ups.pkl', 'rb')
    ups = pickle.load(pfile)
    pfile.close()

    return build_and_evaluate_model(ups, features, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9])



def validate_best_model(eval_function, ups, G, features, input_filename, projects_to_remove, metric, min_x=0, min_y=0, max_x=1002, max_y=1002, kappa = 0.25, n_threads = 5, load_models = False, load_segmentation = False, file_prefix = ""):
    '''
        Do 10 fold cross validation and return the best model
        if metric == recall, then use recall as evaluation metric
        if metric == precision, then use precision as evaluation metric
        otherwise, option is invalid, print error message and then return from the function

        if load_models is true, then load the modles from pickle files
        else create the models using the multithreading
        Each model is saved to a pickle file after creation. 
        The file_prefix parameter will be prepended to each filename

        kappa is the value used for region segmentation.
    '''
    #print("start validating the model")
    locations = store_locations("../data/atlas_complete.json")
    folds = create_folds(min_x, min_y, max_x, max_y)

    # List of dictionaries containing min_x, max_x, min_y, max_y for each fold
    fold_boundaries = []
    for fold in folds:
        fold_boundaries.append(get_fold_border(fold))

    model_filenames = []
    futures = []

    if load_models:
        for i in range(10):
            filename = str(file_prefix) + str(i) + "_model.pkl"
            model_filenames.append(filename)
    else:
        #Multithreading
        # For n_thread = 5, Run n_threads at once, and repeat twice for a total of 2 * n_thread = 10 folds
        for i in range(2):
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
                for t in range(n_threads * i, n_threads * (i + 1)):
                    fut = executor.submit(build_and_evaluate_model_wrapper, (t, G.unique_edges_file_name, fold_boundaries, [t], min_x, min_y, max_x, max_y, kappa, file_prefix))
                    futures.append(fut)

            #Collecting results
            for t in range(n_threads * i, n_threads * (i + 1)):
                fut = futures[t]
                res = fut.result()
                model_filenames.append(res)

    # Create a file to write the evaluation results

    metric_vals = []
    for model_filename in model_filenames:
        pfile = open(model_filename, "rb")
        model = pickle.load(pfile)
        pfile.close()
        print(model)

        model_id = int(model_filename.split("_")[0])

        compute_edge_weights_multithread(G, ups, model, features, 5)
        G.sort_edges()


        comp_assign = None
        comp_assign_filename = "component_assignment_" + model_filename
        if load_segmentation and os.path.exists(comp_assign_filename):
            pfile = open(comp_assign_filename, "rb")
            comp_assign = pickle.load(pfile)
            pfile.close()
        else:
            comp_assign = region_segmentation(G, ups, kappa)
            
            if os.path.exists(comp_assign_filename):
                os.remove(comp_assign_filename)

            pfile = open(comp_assign_filename, 'wb')
            pickle.dump(comp_assign, pfile)
            pfile.close()
    
            
        regions, sizes = extract_regions(comp_assign)
        ground_truth = create_ground_truth(input_filename, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                                           projects_to_remove=projects_to_remove, partial_canvas_boundaries=fold_boundaries[model_id])
        num_correct_counter, num_assignments_made, precision, recall, region_assignments = eval_function( locations, regions, ups, ground_truth, threshold=0.3, draw=False)

        if metric == 'recall':
            metric_val = recall
        elif metric == 'precision':
            metric_val = precision
        else:
            print("invalid metric option")
        
        metric_vals.append(metric_val)
        print(metric_val)
 

    # max_id = metric_vals.index(max(metric_vals))
    # pfile = open(str(max_id) + "_model.pkl", "rb")
    # best_model = pickle.load(pfile)
    # pfile.close()

    return metric_vals



def create_ground_truth(input_filename, min_time=0, max_time=sys.maxsize, min_x=0, max_x=1002, min_y=0, max_y=1002, projects_to_remove = [], partial_canvas_boundaries = None):
    '''
        Given the input file, create and return a dictionary of the ground truth for the
        pixel assignments.
        
        Each pixel's ID will be based upon the index it is found within the file
    '''
    line_number = 0
    ground_truth = dict()
    times = dict()
    if partial_canvas_boundaries is not None:
        # Update min_x, max_x, min_y, max_y to represent all the points within partial canvas
        min_x = partial_canvas_boundaries["min_x"]
        min_y = partial_canvas_boundaries["min_y"]
        max_x = partial_canvas_boundaries["max_x"]
        max_y = partial_canvas_boundaries["max_y"]

    with open(input_filename, 'r') as file_in:

        # Skip first line (header row)
        next(file_in, None)

        reader = csv.reader(file_in)
        for r in reader:
            ts = int(r[0])
            user = r[1]
            x = int(r[2])
            y = int(r[3])
            color = int(r[4])
            pic_id = r[5]
            final_pixel = int(r[6])
            final_pixel_color = int(r[7])
            smallest_proj = int(r[8])

            # The ground truth pixel assignments will be based on the pixel assigned to the smallest project
            if smallest_proj and ts >= min_time and ts < max_time and x >= min_x and x < max_x and y >= min_y and y < max_y and pic_id not in projects_to_remove:

                if pic_id not in ground_truth:
                    ground_truth[pic_id] = set()

                if final_pixel == 1:
                    ground_truth[pic_id].add((x, y, color))

    return ground_truth


def get_region_borders(region, updates):
    '''
        Given a region (list of lists), return the min x, min y, max x, and max y
    '''
    min_x = sys.maxsize
    min_y = sys.maxsize
    max_x = 0
    max_y = 0
    for update_id in region:

        update = updates[int(update_id)]
        x = int(update[2])
        y = int(update[3])

        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    return min_x, min_y, max_x, max_y


def get_rectangle_overlap_area(min_x1, max_x1, min_y1, max_y1, min_x2, max_x2, min_y2, max_y2):
    '''
        Given the coordinates of the corners of two rectangles, return the area of the overlapping region
    '''

    # First, calculate a bounding box around the two rectangles
    bounding_box_area = (max(max_x1, max_x2) - min(min_x1, min_x2)) * \
        (max(max_y1, max_y2) - min(min_y1, min_y2))

    overlap_max_x = min(max_x1, max_x2)
    overlap_max_y = min(max_y1, max_y2)
    overlap_min_x = max(min_x1, min_x2)
    overlap_min_y = max(min_y1, min_y2)

    if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
        return ((overlap_max_x - overlap_min_x + 1) * (overlap_max_y - overlap_min_y + 1))
    else:
        return 0


def calc_intersection_over_union(region_colors, ground_truth):

    intersection = 0
    union = 0
    for value in region_colors:
        if value in ground_truth:
            intersection += 1

    union = len(region_colors) + \
        len(list(ground_truth.difference(set(region_colors))))

    iou = intersection / union

    return iou


def get_max_iou(locations, region, updates, ground_truth):
    '''
        Given a region, check and return the maximum the iou with every project in the ground truth.
    '''
    max_iou = 0

    max_pic_id = None
    for pic_id in ground_truth:
        project = locations[pic_id]
        min_x, min_y, max_x, max_y = get_region_borders(region, updates)

        # only check this region with the pic_id if the bounding boxes overlap by at least the threshold
        overlap_area = get_rectangle_overlap_area(min_x, max_x, min_y, max_y, project.get_left(
        ), project.get_right(), project.get_bottom(), project.get_top())
#         total_area = ((abs(max_x - min_x) + 1) * (abs(max_y - min_y) + 1)) + ((abs(project.get_right() - project.get_left()) + 1) * (abs(project.get_top() - project.get_bottom()) + 1)) - overlap_area

        if overlap_area > 0:

            region_colors = list()
            for pixel in region:
                update = updates[pixel]
                ts = int(update[0])
                user = update[1]
                x = int(update[2])
                y = int(update[3])
                color = int(update[4])

                region_colors.append((x, y, color))

            iou = calc_intersection_over_union(
                region_colors, ground_truth[pic_id])

            if iou > max_iou:
                max_iou = iou
                max_pic_id = pic_id

    return max_iou, max_pic_id


# Given a set_of_updates and the ground truth data, evaluate the precision and recall
# ground_truth is a dictionary of the following format:
'''
    "image_id" : [ list of update IDS belonging to this image ]
'''

# Each update tuple is assumed to be this format:
# (updateID, time, user, x, y, color, pic_id)


def evaluate(locations, regions, updates, ground_truth, threshold=0.50, draw=False):

    if len(regions) == 0 or len(ground_truth) == 0:
        return 0, 0, 0, 0, dict()

    region_assignments = dict()
    num_assignments_made = len(regions)
    ground_truth_size = len(ground_truth)

    image_assignment = dict()
    num_correct_counter = 0

    for region in regions:
        iou, pic_id = get_max_iou(locations, region, updates, ground_truth)

        if iou > threshold:
            if region_assignments.get(pic_id) is None:
                # If this artwork has not been predicted yet, then add 1 to num_correct_counter and set the iou for
                # this image
                num_correct_counter += 1
                region_assignments[pic_id] = (iou, region)
            else:
                # Else if there is a higher iou, update the value within region_assignments
                if iou > region_assignments[pic_id][0]:
                    region_assignments[pic_id] = (iou, region)

            # if draw:
            #     print(pic_id)
            #     name = locations[pic_id].get_name().replace('/', '')
            #     draw_canvas_region(
            #         updates, region, "../drawings/" + name + "_region.png")
            #     draw_canvas_region(
            #         updates, ground_truth[pic_id], "../drawings/" + name + "_truth.png", True)
#     if draw:
#         for pic_id in region_assignments:
#             print(pic_id)
#             region = region_assignments[pic_id][1]
#             name = locations[pic_id].get_name().replace('/', '')
#             draw_canvas_region(updates, region, "../drawings/" + name + "_region.png")
#             draw_canvas_region(updates, ground_truth[pic_id], "../drawings/" + name + "_truth.png", True)

    precision = num_correct_counter / num_assignments_made

    recall = num_correct_counter / ground_truth_size

    return num_correct_counter, num_assignments_made, precision, recall, region_assignments


def compute_overlap_area(locations, region, updates, ground_truth):
    '''
        Given a region, return a dictionary that lists the percent overlap that region has with every ground truth
    '''
    truncated_ground_truth = dict()
    for pic_id in ground_truth:

        if pic_id not in truncated_ground_truth:
            truncated_ground_truth[pic_id] = set()
        
        for datapoint in ground_truth[pic_id]:
            # datapoint is a tuple: (ts, user, x, y, color)
            truncated_ground_truth[pic_id].add( (datapoint[2], datapoint[3])  )
    

    regions_xy = []
 
    for idx in region:
        update = updates[idx]
        x = int(update[2])
        y = int(update[3])

        regions_xy.append((x, y))
        
    
    overlap_statistics = dict()
    for pic_id in truncated_ground_truth:

        project = locations[pic_id]

        min_x, min_y, max_x, max_y = get_region_borders(region, updates)
        overlap_area = get_rectangle_overlap_area(min_x, max_x, min_y, max_y, project.get_left(
        ), project.get_right(), project.get_bottom(), project.get_top())

        if overlap_area > 0:

            total_num_pixels = len(region)
            overlap_pixels = len( regions_xy.intersect(truncated_ground_truth[ pic_id ]) )

            overlap_statistics[pic_id] = float(overlap_pixels) / total_num_pixels
        else:
            overlap_statistics[pic_id] = 0


    return overlap_statistics

