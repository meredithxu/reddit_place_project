import sys
import os
import csv
import numpy as np
from reddit import *
from segmentation import *
from features_ups import *
from features_super import *
from nonlinear_regressor import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import concurrent.futures
import time

def create_regions(iterations, 
                    modeltype,
                    num_threads,
                    input_file,
                    projects_to_remove,
                    dist_threshold, 
                    ndim, 
                    feature_threshold, 
                    total_samples, 
                    n_negatives, 
                    n_iterations,
                    min_x = 0, 
                    max_x = 1002, 
                    min_y = 0, 
                    max_y = 1002,
                   
                    ):
    '''
       Take all of the updates within input_file and cluster them into regions
       PARAMS:
            - modeltype indicated which type of model will be used to cluster the
            updates.
            modeltype = 'gboost' will use a GradientBoostingRegressor
            modeltype = 'nn' will use a neural network



        NOTE: This function saves several pkl files in order to avoid
        recomputation in case of failure.
        However, this also means that these pkl files must be manually deleted
        if you want to regenerate the data structures.
    '''

    # Verify that the user selected a valid modeltype
    valid_modeltypes = set(['nn', 'gboost'])
    if not modeltype in valid_modeltypes:
        return None

    graph_filename = 'graph.pkl'
    ups_filename = 'ups.pkl'
    features_filename = 'features.pkl'

    # Create a graph of the updates where each update is one node
    # First check if this graph has already been created. If so, load the pickle
    # Else create it and save to a pickle
    G_ups = None
    ups = None
    if not (os.path.exists(graph_filename) and os.path.exists(ups_filename)):
        t = time.time()
        G_ups, ups = create_graph(input_file, projects_to_remove, dist_threshold, min_x, max_x, min_y, max_y)

        print("num edges = ", G_ups.n_edges)

        pfile = open(graph_filename, 'wb')
        pickle.dump(G_ups, pfile)
        pfile.close()

        pfile = open(ups_filename, 'wb')
        pickle.dump(ups, pfile)
        pfile.close()

        print("time to create G_ups= ", time.time()-t, " seconds")
    else:
        pfile = open(graph_filename, 'rb')
        G_ups = pickle.load(pfile)
        pfile.close()

        pfile = open(ups_filename, 'rb')
        ups = pickle.load(pfile)
        pfile.close()
    
    # Create features for the updates graph
    # First check if this features dictionary has already been created. If so, load the pickle
    # Else create it and save to a pickle
    features = None
    if not os.path.exists(features_filename):
        features = create_features(G_ups, ups, ndim, feature_threshold,
                                   total_samples, n_negatives, n_iterations, features_filename)
    else:
        pfile = open(features_filename, 'rb')
        features = pickle.load(pfile)
        pfile.close()  
    

    model0_filename = 'model0.pkl'
    model = None
    if not os.path.exists(model0_filename):
        #Creating feature matrix A and vector of labels b
        #for learning an edge weight model
        t = time.time()

        A,b = build_feat_label_data(G_ups, ups, features)

        print(A.shape)
        print(b.sum() / b.shape[0])

        print("time to build feat label data = ", time.time()-t, " seconds")

        t = time.time()
        if modeltype == 'gboost':
            model = GradientBoostingRegressor(random_state=1, n_estimators=25).fit(A, b)
        else:
            model = createNonlinearRegressionNeuralNet(A, b)
        pfile = open(model0_filename, 'wb')
        pickle.dump(model, pfile)
        pfile.close()

        print("time to fit model= ", time.time()-t, " seconds")
    else:
        pfile = open(model0_filename, 'rb')
        model = pickle.load(pfile)
        pfile.close()
                
    regions_filename = 'up_regions.pkl'
    regions = None
    sizes = None
    int_weights = None
    if not os.path.exists(regions_filename):
        #Computing edge weights using multithreading
        #and sorting the edges in increasing order.
        #Not every edge in the the unique_edges file
        #will be included in the new file.

        t = time.time()

        compute_edge_weights_multithread(G_ups, ups, model, features, features_filename, num_threads)
        G_ups.sort_edges()

        print("time to compute edge weights= ", time.time()-t, " seconds")

        #Performing region segmentation on the update graph

        t = time.time()

        comp_assign, int_weights = region_segmentation(G_ups, ups, 0.8)
        regions, sizes, int_weights = extract_regions(comp_assign, int_weights)

        pfile = open(regions_filename, 'wb')
        pickle.dump([regions, sizes, int_weights], pfile)
        pfile.close()

        print("num regions = ", len(regions), " max size region = ", np.max(sizes))

        print("time to create regions= ", time.time()-t, " seconds")
    else:
        pfile = open(regions_filename, 'rb')
        ups_region_info = pickle.load(pfile)
        pfile.close()

        regions = ups_region_info[0]
        sizes = ups_region_info[1]
        int_weights = ups_region_info[2]
     
    durations = compute_update_durations(ups)
    for i in range(1, iterations):
        reg_graph_filename = "reg_graph" + str(i) + ".pkl"
        reg_features_filename = "features_regions" + str(i) + ".pkl"
        reg_model_filename = 'model_regions' + str(i) + '.pkl'
        super_regions_filename = 'super_regions' + str(i) + '.pkl'

        G_reg = None

        if not os.path.exists(reg_graph_filename):
            t = time.time()

            G_reg = build_region_graph(G_ups, regions, ups, .5, projects_to_remove)

            pfile = open(reg_graph_filename, 'wb')
            pickle.dump(G_reg, pfile)
            pfile.close()

            print("time to create reg_graph",i,"= ", time.time()-t, " seconds")
        else:
            pfile = open(reg_graph_filename, 'rb')
            G_reg = pickle.load(pfile)
            pfile.close()

        region_features = None
        if not os.path.exists(reg_features_filename):
            region_features = create_superfeatures(
                regions, int_weights, ups, features, durations, reg_features_filename)
            
        else:
            #Reading existing feature data
            pfile = open(reg_features_filename, 'rb')
            region_features = pickle.load(pfile)
            pfile.close()
        
        model = None

        if not os.path.exists(reg_model_filename):
            t = time.time()

            A, b = build_feat_label_regions(G_reg, ups, region_features)

            print(A.shape)
            print(b.sum() / b.shape[0])

            print("time to build feat label data",i,"= ", time.time()-t, " seconds")

            #Learning region edge weight model using gradient boosting regression
            #and saving model.

            t = time.time()

            if modeltype == 'gboost':
                model = GradientBoostingRegressor(random_state=1, n_estimators=25).fit(A, b)
            else:
                model = createNonlinearRegressionNeuralNet(A, b)

            pfile = open(reg_model_filename, 'wb')
            pickle.dump(model, pfile)
            pfile.close()

            print("time to fit model", i, "= ", time.time()-t, " seconds")
        else:
            #Reading an existing region edge weight model

            pfile = open(reg_model_filename, 'rb')
            model = pickle.load(pfile)
            pfile.close()

        
        #Computing region edge weights using multithreading
        #and sorting the edges in increasing order.
        super_region_sizes = None
        super_region_assign = None
        
        if not os.path.exists(super_regions_filename):
            t = time.time()

            compute_edge_weights_multithread(G_reg, ups, model, region_features, 'features_regions.pkl', num_threads)
            G_reg.sort_edges()

            print("time compute edge weights",i,"= ", time.time()-t, " seconds")

            #Performing segmentation on the region graph
            t = time.time()

            comp_assign_reg, int_weights_reg = region_segmentation(G_reg, regions, 0.55)
            reg_regions, reg_sizes, int_weights = extract_regions(comp_assign_reg, int_weights_reg)
            regions, super_region_sizes, super_region_assign = extract_super_region_info(reg_regions, regions)

            pfile = open(super_regions_filename, 'wb')
            pickle.dump([regions, super_region_sizes, super_region_assign, int_weights], pfile)
            pfile.close()

            print("time to create super regions",i,"= ", time.time()-t, " seconds")
        else:
            pfile = open(super_regions_filename, 'rb')
            super_region_info = pickle.load(pfile)
            pfile.close()

            regions = super_region_info[0]
            super_region_sizes = super_region_info[1]
            super_region_assign = super_region_info[2]
            int_weights = super_region_info[3]
        
        print("num regions = ", len(regions), " max size region = ", np.max(super_region_sizes))


    return regions
    


def create_folds(min_x=0, min_y=0, max_x=1002, max_y=1002):
    # Partition the data into folds

    num_folds = 4
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


def build_and_evaluate_model(G, ups, 
                                features, 
                                pid, 
                                fold_boundaries, 
                                excluded_folds, 
                                file_prefix = "", 
                                modeltype = 0,):
    '''
        Build a model and return the evaluation metric

        modeltype determines what type the model that computes the edge weights will be
            modeltype = 0 will use a sklearn.ensemble.GradientBoostingRegressor
            modeltype = 1 will create a keras.models.Sequential neural network
    '''
    locations = store_locations("../data/atlas_complete.json")
    scaler_A = StandardScaler()
    scaler_b = StandardScaler()
    
    # All edges that belong to the validation fold need to be excluded
    A, b = build_feat_label_data(G, ups, features,
                                 fold_boundaries=fold_boundaries, excluded_folds=excluded_folds)
    
    scaler_A.fit(A)
    b = np.matrix(b).T
    scaler_b.fit(b)
    A = scaler_A.transform(A)
    b = (scaler_b.transform(b)).T[0]

    model = None
    if modeltype == 0:
        model = GradientBoostingRegressor(random_state=1, n_estimators=25).fit(A, b)
    elif modeltype == 1:
        model = createNonlinearRegressionNeuralNet(A, b)

    del A
    del b
    model_name = str(file_prefix) + "_" +  str(pid) + "_model"
    if modeltype == 0:
        # Save the model in a pickle file
        model_name = model_name + ".pkl"
        if os.path.exists(model_name):
            os.remove(model_name)
            
        pfile = open(model_name, 'wb')
        pickle.dump(model, pfile)
        pfile.close()     
    else:
        model.save(model_name)

    filenameA = str(pid) + 'std_scaler_A.pkl'
    filenameb = str(pid) + 'std_scaler_b.pkl'

    if os.path.exists(filenameA):
        os.remove(filenameA)

    pickle.dump(scaler_A, open(filenameA, 'wb'))

    if os.path.exists(filenameb):
        os.remove(filenameb)
    
    pickle.dump(scaler_b, open(filenameb, 'wb'))

    return model_name

def build_and_evaluate_model_wrapper(params):
    #Loading pickled features
    #Each thread has its own copy if passed as a param, which is quite inneficient
    file_prefix = params[3]
    pfile = open(file_prefix + 'graph.pkl', 'rb')
    G = pickle.load(pfile)
    pfile.close()

    pfile = open(file_prefix + 'features.pkl', 'rb')
    features = pickle.load(pfile)
    pfile.close()

    pfile = open(file_prefix + 'ups.pkl', 'rb')
    ups = pickle.load(pfile)
    pfile.close()

    return build_and_evaluate_model(G, ups, features, params[0], params[1], params[2], params[3], params[4])


def validate_best_model(eval_function, ups, G, features, input_filename, projects_to_remove, metric, ground_truth,
                            min_x=0, 
                            min_y=0, 
                            max_x=1002,
                            max_y=1002, 
                            kappa = 0.25, 
                            load_models = False, 
                            load_segmentation = False, 
                            file_prefix = "",
                            evaluation_threshold = 0.5,
                            modeltype = 0):
    '''
        Do 10 fold cross validation and return the best model
        if metric == recall, then use recall as evaluation metric
        if metric == precision, then use precision as evaluation metric
        otherwise, option is invalid, print error message and then return from the function

        if load_models is true, then load the models from pickle files
        else create the models using the multithreading
        Each model is saved to a pickle file after creation. 
    
        The file_prefix parameter will be prepended to each filename

        kappa is the value used for region segmentation.

        evaluation_threshold is the proportion of the ground truth image that a prediction must 
        match to be consideredd correct

        modeltype determines what type the model that computes the edge weights will be
            modeltype = 0 will use a sklearn.ensemble.GradientBoostingRegressor
            modeltype = 1 will create a keras.models.Sequential neural network
    '''
    locations = store_locations("../data/atlas_complete.json")
    folds = create_folds(min_x, min_y, max_x, max_y)
    
    # List of dictionaries containing min_x, max_x, min_y, max_y for each fold
    fold_boundaries = []
    for fold in folds:
        fold_boundaries.append(get_fold_border(fold))

    model_filenames = []
    futures = []

    # Check if all the model filenames you want exist. 
    # If they do, then you can load them. Otherwise they must be regenerated
    missing_model = False
    if load_models:
        for i in range(10):
            filename = str(file_prefix) + "_" + str(i) + "_model"
            if modeltype == 0:
                filename = filename + ".pkl"

            if not os.path.exists(filename):
                missing_model = True
            else:
                model_filenames.append(filename)

    if not load_models or missing_model:
        n_threads = 4
        t_model = time.time()

        #Multithreading: 4 fold cross validation
      
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_threads) as executor:
            for t in range(0, n_threads):
                filename = str(file_prefix) + "_" + str(t) + "_model"
                if modeltype == 0:
                    filename = filename + ".pkl"

                if not load_models or (missing_model and not os.path.exists(filename)):
                    fut = executor.submit(build_and_evaluate_model_wrapper, (t, fold_boundaries, 
                                                                                [t], file_prefix, modeltype))
                    futures.append(fut)

        #Collecting results
        for t in range(len(futures)):
            fut = futures[t]
            res = fut.result()
            model_filenames.append(res)
        
        print("time to create models= ", time.time()-t_model, " seconds") 

    # Create a file to write the evaluation results

    metric_vals = []
    for model_filename in model_filenames:
        print(model_filename)
        pfile = open(model_filename, "rb")
        model = pickle.load(pfile)
        pfile.close()

        model_id = int(model_filename.split("_")[1])


        # When predicting the edge weights, check if 
        filenameA= str(model_id) + 'std_scaler_A.pkl'
        if not os.path.exists(filenameA):
            filenameA = None

        filenameb = str(model_id) + 'std_scaler_b.pkl'
        if not os.path.exists(filenameb):
            filenameb = None

        comp_assign = None
        comp_assign_filename = "component_assignment_" + model_filename
        if load_segmentation and os.path.exists(comp_assign_filename):
            pfile = open(comp_assign_filename, "rb")
            comp_assign = pickle.load(pfile)
            pfile.close()
        else:
            t = time.time()
            compute_edge_weights_multithread(G, ups, model, features, file_prefix + 'features.pkl', 5, filenameA, filenameb)
            G.sort_edges()
            print("time to calculate and sort edge weigths= ", time.time()-t, " seconds")

            t = time.time()
            comp_assign = region_segmentation(G, ups, kappa)
            print("time to segment regions= ", time.time()-t, " seconds")

            if os.path.exists(comp_assign_filename):
                os.remove(comp_assign_filename)

            pfile = open(comp_assign_filename, 'wb')
            pickle.dump(comp_assign, pfile)
            pfile.close()
        
        print()
    
            
        regions, sizes = extract_regions(comp_assign)
        t = time.time()
        num_correct_counter, num_assignments_made, precision, recall = eval_function(
            locations, regions, ups, ground_truth, threshold=evaluation_threshold,
            min_x=fold_boundaries[model_id]["min_x"], max_x=fold_boundaries[model_id]["max_x"],
            min_y=fold_boundaries[model_id]["min_y"], max_y=fold_boundaries[model_id]["max_y"])
        print("time to evaluate regions= ", time.time()-t, " seconds")


        if metric == 'recall':
            metric_val = recall
        elif metric == 'precision':
            metric_val = precision
        else:
            print("invalid metric option")
        
        metric_vals.append(metric_val)
        print(metric_val)
 

    return metric_vals



def create_ground_truth(input_filename, min_time=0, max_time=sys.maxsize, min_x=0, max_x=1002, min_y=0, max_y=1002, projects_to_remove = []):
    '''
        Given the input file, create and return a dictionary of the ground truth for the
        pixel assignments.
        
        Each pixel's ID will be based upon the index it is found within the file
    '''
    line_number = 0
    ground_truth = dict()
    times = dict()

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

    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}



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


def get_max_iou(locations, region, updates, ground_truth, region_borders):
    '''
        Given a region, check and return the maximum the iou with every project in the ground truth.
    '''
    max_iou = 0

    max_pic_id = None
    for pic_id in ground_truth:
        project = locations[pic_id]
        min_x = region_borders["min_x"]
        max_x = region_borders["max_x"]
        min_y = region_borders["min_y"]
        max_y = region_borders["max_y"]

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




def evaluate(locations, regions, updates, ground_truth, threshold=0.50, min_x=0,max_x=1002,min_y=0,max_y=1002):

    # Given a set_of_updates and the ground truth data, evaluate the precision and recall
    # ground_truth is a dictionary of the following format:
    
    #   "image_id" : [ list of update IDS belonging to this image ]

    # Each update tuple is assumed to be this format:
    # (updateID, time, user, x, y, color, pic_id)
    # min_x, max_x, min_y, max_y determines the area of the canvas that you are evaluating over

    if len(regions) == 0 or len(ground_truth) == 0:
        return 0, 0, 0, 0, dict()

    region_assignments = dict()
    num_assignments_made = len(regions)
    ground_truth_size = len(ground_truth)

    image_assignment = dict()
    num_correct_counter = 0

    for region in regions:
        region_borders = get_region_borders(region, updates)

        # check that the region is within the area we are evaluating
        if region_borders["min_x"] < min_x or region_borders["max_x"] > max_x or region_borders["min_y"] < min_y or region_borders["max_y"] > max_y:
            continue

        iou, pic_id = get_max_iou(locations, region, updates, ground_truth, region_borders)

        if iou > threshold:
            if region_assignments.get(pic_id) is None:
                # If this artwork has not been predicted yet, then add 1 to num_correct_counter and set the iou for
                # this image
                num_correct_counter += 1
                region_assignments[pic_id] = True
           


    precision = num_correct_counter / num_assignments_made

    recall = num_correct_counter / ground_truth_size

    return num_correct_counter, num_assignments_made, precision, recall

