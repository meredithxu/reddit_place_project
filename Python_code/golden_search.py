import math
import operator
from segmentation import *


def compute_updates_per_project(ups, pixel):
    '''
        Returns a dictionary of dictionaries
        where [proj][update] is True if 
        the update belongs to the project.
    '''
    updates_proj = {}

    for u in range(len(ups)):
        up = ups[u]
        if pixel is True:
            if up[6] == 1:
                if up[5] not in updates_proj:
                    updates_proj[up[5]] = {}

                updates_proj[up[5]][u] = True
        else:
            if up[7] == 1:
                if up[5] not in updates_proj:
                    updates_proj[up[5]] = {}

                updates_proj[up[5]][u] = True

    return updates_proj


def max_intersection_over_union(region, updates_proj, ups, pixel):
    '''
        Computes value of intersection over union for 
        maximum value project. If pixel is True takes into
        account only final updates as true positives, otherwise
        also takes into account updates with the same color as those.
    '''
    inters = {}
    size_final = 0

    for u in region:
        up = ups[u]

        if pixel is True:
            if up[6] == 1:
                if up[5] not in inters:
                    inters[up[5]] = 0
                    size_final = size_final + 1

                inters[up[5]] = inters[up[5]] + 1
        else:
            if up[7] == 1:
                if up[5] not in inters:
                    inters[up[5]] = 0

                inters[up[5]] = inters[up[5]] + 1
                size_final = size_final + 1

    for proj in inters:
        inters[proj] = inters[proj] / \
            (len(updates_proj[proj])+size_final-inters[proj])

    if len(inters) > 0:
        max_proj, max_score = max(inters.items(), key=operator.itemgetter(1))
    else:
        max_proj = 0
        max_score = 0

    return max_proj, max_score


def recall(regions, updates_proj, ups, threshold, pixel):
    '''
        Computes recall of the regions based on 
        matching threshold for intersection-over-union.
    '''
    recovered_projs = {}
    for region in regions:
        proj, score = max_intersection_over_union(
            region, updates_proj, ups, pixel)

        if score > threshold:
            recovered_projs[proj] = True

    return len(recovered_projs) / len(updates_proj)


def precision(regions, updates_proj, ups, threshold, pixel):
    '''
        Computes precision of the regions based on 
        matching threshold for intersection-over-union.
    '''
    tp = 0.
    fp = 0.
    for region in regions:
        proj, score = max_intersection_over_union(
            region, updates_proj, ups, pixel)

        if score > 0:
            if score > threshold:
                tp = tp + 1
            else:
                fp = fp + 1

    return tp / (tp + fp)


def f1_score(regions, updates_proj, ups, threshold, pixel):
    '''
        Computes F1-score of the regions based on 
        matching threshold for intersection-over-union.
    '''
    prec = precision(regions, updates_proj, ups, threshold, pixel)
    rec = recall(regions, updates_proj, ups, threshold, pixel)

    return 2 * prec * rec / (prec + rec)


def f_gss_ups(kappa, args):
    '''
        Performs update segmentation with the given value of
        kappa and returns the score using the given metric.
    '''
    G, ups, updates_proj, threshold, pixel, metric = args

    comp_assign, int_weights = region_segmentation(G, ups, kappa)
    regions, sizes, int_weights = extract_regions(comp_assign, int_weights)

    m = metric(regions, updates_proj, ups, threshold, pixel)

    print("kappa = ", kappa, " score = ", m)

    return -m


def f_gss_region(kappa, args):
    '''
        Performs super region segmentation using the given value
        of kappa and returns the score using the given metric.
    '''
    G, ups, regions, updates_proj, threshold, pixel, metric = args

    comp_assign_reg, int_weights_reg = region_segmentation(G, regions, kappa)
    reg_regions, reg_sizes, super_int_weights = extract_regions(
        comp_assign_reg, int_weights_reg)
    super_regions, super_region_sizes, super_region_assign = extract_super_region_info(
        reg_regions, regions)

    m = metric(super_regions, updates_proj, ups, threshold, pixel)

    print("kappa = ", kappa, " score = ", m)

    return -m


def gss(f, args, a, b, tol=1e-5):
    """Golden section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    modified from: https://en.wikipedia.org/wiki/Golden-section_search
    
    Usage: gss(f_gss, [G_reg_2, ups, super_regions, updates_proj, .5, False, recall], 0., 1.)
    """
    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c, args)
    yd = f(d, args)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c, args)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d, args)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)


def superv_reg_segm_ups(G, ups, ups_eval, min_kappa, max_kappa, updates_proj, updates_proj_eval, metric, threshold=0.5, pixel=False, tol=1e-2):
    '''
        Returns segmentation that maximizes the metric by performing a 1-d search
        over kappa.
    '''
    kappa_range = gss(
        f_gss_ups, [G, ups, updates_proj, threshold, False, recall], 0., 1.)
    kappa = (kappa_range[0]+kappa_range[1]) / 2
    comp_assign, int_weights = region_segmentation(G, ups, kappa)
    regions, sizes, int_weights = extract_regions(comp_assign, int_weights)
    
    print("score = ", metric(regions, updates_proj_eval, ups_eval, threshold, pixel))
    return regions, sizes, int_weights


def superv_reg_segm_reg(G, ups, ups_eval, regions, min_kappa, max_kappa, updates_proj, updates_proj_eval, metric, threshold=0.5, pixel=False, tol=1e-2):
    '''
        Returns segmentation that maximizes the metric by performing a 1-d search
        over kappa.
    '''
    kappa_range = gss(
        f_gss_region, [G, ups, regions, updates_proj, threshold, False, recall], 0., 1.)
    kappa = (kappa_range[0]+kappa_range[1]) / 2

    comp_assign_reg, int_weights_reg = region_segmentation(G, regions, kappa)

    reg_regions, reg_sizes, super_int_weights = extract_regions(comp_assign_reg, int_weights_reg)
    super_regions, super_region_sizes, super_region_assign = extract_super_region_info(reg_regions, regions)
    
    print("score = ", metric(super_regions, updates_proj_eval, ups_eval, threshold, pixel))
    
    return super_regions, super_region_sizes, super_int_weights


# super_regions, super_region_sizes, super_int_weights = superv_reg_segm_reg(G_reg, ups, regions, 0., 2., updates_proj, recall)
