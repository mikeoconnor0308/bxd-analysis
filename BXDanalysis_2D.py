#!/usr/bin/python
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import json
from itertools import tee
import os
import errno
from pudb import set_trace
try:
    import progressbar
    progress_bar = True
except ImportError:
    progress_bar = False
    print("Warning: Could not find progressbar module. Try running:  ",
          "\n\tpip install progressbar33\n This will allow for pretty",
          "progress bar output")

output_dir = "analysis"

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127,
                                                127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
tableau20 = [tuple(float(c) / 256 for c in x) for x in tableau20]


def make_sure_path_exists(path):
    """
    Ensures that the path specified exists, and makes it if necessary
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def pairwise(iterable):
    """
    Produces a pairwise list over an iterable
    e.g. [x,y,z,w] - > [(x,y),(y,z), (z,w)]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def ComputeBoxCenters(plane_points):
    """
    Given a set of points, computes the midpoint of the
    straight line between each consecutive pair of points
    """
    if plane_points is None:
        return None
    box_centers = []
    for (a, b) in pairwise(plane_points):
        box_centers.append((a + b) * 0.5)
    return box_centers


def ComputeDistancesAlongCV(centers, start=None):
    """
    Given a list of points, performs a prefix sum to compute
    total distance along the shortest path through the points.
    Returns a list of these distances
    e.g. [1,4,7,9] returns [0,3,6,8]

    optional argument start gives an initial start point for the distance,
    otherwise first value will always be zero.
    """
    if start is None:
        points_diff = [centers[0]] + centers
    else:
        points_diff = [start] + centers
    points_diff = [np.linalg.norm(points_diff[i] -
                                  points_diff[i - 1])
                   for i in range(1, len(points_diff))]
    points_diff = np.cumsum(points_diff)
    return points_diff


def BisectPlane(plane_1, plane_2, fraction):
    """
    Creates plane that goes through the intersection between plane_1 and
    plane_2. Fraction governs how far from plane1 to plane2 it will be placed.

    Args:
        plane_1 (list): first plane
        plane_2 (list): second plane
        fraction (float): proportion from plane_1 to plane_2.
    """
    assert len(plane_1) == len(plane_2), "Planes do not share dimension"
    assert len(plane_1) - 1 == 2, "Currently only have 2D working!"

    plane_1 = np.array(plane_1, dtype=np.float64)
    plane_2 = np.array(plane_2, dtype=np.float64)
    # TODO Generalise this to hyperplanes
    norm_1 = np.array(plane_1[:2])
    norm_2 = np.array(plane_2[:2])

    plane = []
    if np.array_equal(norm_1, norm_2):
        D = plane_1[-1] + (plane_2[-1] - plane_1[-1]) * fraction
        plane = np.append(norm_1, D)
    else:
        # slightly heavy-handed, but i wanted to see how np solve works
        x0 = np.array([-plane_1[-1], -plane_2[-1]])
        X = np.array([norm_1, norm_2])
        try:
            point = np.linalg.solve(X, x0)
            new_norm = np.add((1-fraction) * norm_1, (fraction) * norm_2)
            length = np.linalg.norm(new_norm)
            if(length != 0):
                new_norm /= length
            # compute D
            D = np.dot(new_norm, point)
            # invert sign of D so plane is in form ax + by +cz + d = 0
            D = -D
            plane = np.append(new_norm, D)
        except np.linalg.linalg.LinAlgError:
            print(
                "WARNING: Linalg error trying to find intersection between",
                "planes ", plane_1, plane_2,
                "will simply place plane between them")
            D = plane_1[-1] + (plane_2[-1] - plane_1[-1]) * fraction
            plane = np.append(norm_1, D)

    return np.array(plane)


def CreateBisectionPlanes(planes, plane_points, max_distance):
    """
    Given a set of planes and points on those planes, bisects the planes
    such that the distance between the given point on each intermediate plane
    is at most max_distance
    """

    planes = np.array(planes, dtype=np.float64)
    plane_points = np.array(plane_points, dtype=np.float64)
    new_planes = []
    hist_plane_points = []
    hist_centers = []
    hist_bools = []
    for (a, b), (ca, cb) in zip(pairwise(planes), pairwise(plane_points)):
        dist = np.linalg.norm(ca - cb)
        Nbisections = 0
        if(dist > max_distance):
            Nbisections = int(dist / max_distance)
        step = 1.0 / (Nbisections + 1)
        frac = step
        bisections = []
        new_planes.append(a)
        hist_plane_points.append(ca)
        hist_bools.append(False)
        if(Nbisections == 0):

            continue
        for i in range(Nbisections):
            bisections.append(BisectPlane(a, b, frac))
            hist_plane_points.append(ca + frac * (cb - ca))
            hist_bools.append(True)
            frac += step
        new_planes += bisections
    new_planes.append(planes[-1])
    hist_bools.append(False)
    hist_plane_points.append(plane_points[-1])
    for (a, b) in pairwise(hist_plane_points):
        hist_centers.append((a + b) / 2.0)
    assert len(new_planes) == len(hist_centers) + \
        1, "Error: number of centers generated does not match number of planes!"
    return new_planes, hist_centers, hist_plane_points, hist_bools


def ComputeHistError(count, count_in_box, box_id, box_free_energies,
                     box_errors):
    """
    Calculates the error in the high resolution histogram free energy
    calculation
    """

    #estimation for error on histogram counts
    #count_var = math.sqrt(count)
    count_var = count*(1.0 - count/count_in_box)

    #propagation of error formula
    total_box_free_energy = np.sum(box_free_energies)
    error_var = math.pow((1.0/count_in_box - 1.0/count), 2)*count_var
    if total_box_free_energy > 0:
        error_var += math.pow((1.0/total_box_free_energy - 1.0) *
                              box_errors[box_id], 2)

        i = 0
        for box_std in box_errors:
            if i == box_id:
                continue
            error_var += math.pow((1.0/total_box_free_energy) * box_std, 2)
            i += 1
    #return deviation
    return math.sqrt(error_var)


def BXDanalysis(TrajectoryFiles, BoundsFilename, Ndim,
                MFPTthreshold=0.0, BoxLowerID=0,
                BoxUpperID=None, Nsweeps=None, plot=True,
                MaxDistance=float("inf"),
                ThresholdFile=None, ReverseBounds=False, read_fpts=False,
                FPTDirectories=None, HistFiles=None):

    # First read all the bounds
    print("Reading bounds from json file:", BoundsFilename)
    BoundaryList, plane_points = read_bounds_json(
        BoundsFilename, ReverseBounds)
    nBounds = len(BoundaryList)
    assert len(BoundaryList) > 0, 'No bounds found!'

    # If no upper box ID has been specified, then set it as last box
    if BoxUpperID is None:
        BoxUpperID = len(BoundaryList) - 2

    # Error checking
    assert BoxUpperID >= 0 and BoxUpperID < len(
        BoundaryList) - 1, 'Upper Box ID must be within range 0' \
        'and number of bounds - 2 (' + str(len(BoundaryList) - 2) + ')'
    assert BoxLowerID >= 0 and BoxLowerID <= BoxUpperID, 'Lower Box ID must' \
        'be within range 0 and upper box ID (' + str(BoxUpperID) + ')'

    #create output directory
    make_sure_path_exists(output_dir)

    # print some info about the bounds
    for i in range(0, nBounds - 1):
        print('\tBox ', i, ' spans ',
              BoundaryList[i], ' to ', BoundaryList[i + 1])

    # now create histogram bin planes
    assert plane_points is not None, 'Need points on each plane ' \
        'please make JSON file'
    hist_planes, hist_centers, \
        hist_plane_points, hist_bools = CreateBisectionPlanes(
            BoundaryList[BoxLowerID:(BoxUpperID + 2)],
            plane_points[BoxLowerID:(BoxUpperID + 2)], MaxDistance)

    # Store the generated histogram bounds to file
    print("Writing histogram bounds to hist_bounds.json")
    hist_out = output_dir + "/" + "hist_bounds.json"
    store_bounds_json(hist_out, hist_planes, hist_plane_points,
                      hist_centers, hist_bools)

    # now calculate the MFPT
    kUpperList = []
    kLowerList = []

    fpt_lower_list, fpt_upper_list = GetFPTs(FPTDirectories, TrajectoryFiles,
                                             BoundaryList, BoxLowerID,
                                             BoxUpperID, Ndim, Nsweeps,
                                             nBounds - 1)

    assert MFPTthreshold >= 0.0, 'MFPT threshold must be greater than zero'

    thresholds_lower = [0.0] * (nBounds - 1)
    thresholds_upper = [0.0] * (nBounds - 1)
    if ThresholdFile:
        thresholds_lower, thresholds_upper = ReadThresholds(
            threshold_file, thresholds_lower, thresholds_upper)

    print("Computing MFPTs")
    kLowerList, kUpperList = ComputeMFPTs(fpt_lower_list, fpt_upper_list,
                                          BoundaryList, BoxLowerID, BoxUpperID,
                                          thresholds_lower, thresholds_upper)

    print("Computing Box Free Energies")
    g, p, e = ComputeBoxFreeEnergies(kLowerList, kUpperList, fpt_lower_list,
                                     fpt_upper_list, BoxLowerID,
                                     BoxUpperID, plane_points)
    boxFreeEnergy = g
    boxProbability = p
    boxError = e

    # populate histogram
    hist_counts = GetHistogram(HistFiles, TrajectoryFiles, hist_planes,
                               hist_centers, Ndim)

    # count the events in each box
    idx = 0
    nBoxes = BoxUpperID + 1 - BoxLowerID
    TotalCountsInBox = [0.0] * nBoxes
    for i in range(nBoxes):
        bid = i + BoxLowerID
        TotalCountsInBox[i] += hist_counts[idx]
        while not np.array_equal(hist_planes[idx + 1], BoundaryList[bid + 1]):
            print(hist_planes[idx + 1], BoundaryList[bid + 1])
            idx = idx + 1
            TotalCountsInBox[i] += hist_counts[idx]
        idx = idx + 1

    hist_errors = []
    normalized_hist = []
    # normalized the raw histogram & obtain the box probability, and do print
    # outs
    normalizedHistogram = open(output_dir + '/normalizedHistogram.txt', 'w')
    rawBoxNormalizedHistogram = open(
        output_dir + '/rawBoxNormalizedHistogram.txt', 'w')
    idx = 0
    cv_dist = ComputeDistancesAlongCV(hist_centers)
    try:
        for i in range(nBoxes):
            bid = i + BoxLowerID
            string = '%s\t%s\n' % (
                cv_dist[idx], hist_counts[idx] / TotalCountsInBox[i])
            rawBoxNormalizedHistogram.write(string)
            hist_errors.append(ComputeHistError(hist_counts[idx],
                               TotalCountsInBox[i], i, boxFreeEnergy,
                               boxError))

            hist_counts[idx] = boxProbability[i] * \
                hist_counts[idx] / TotalCountsInBox[i]
            normalized_hist.append(hist_counts[idx])
            string = '%s\t%s\n' % (cv_dist[idx], hist_counts[idx])
            normalizedHistogram.write(string)
            while not np.array_equal(hist_planes[idx + 1],
                                     BoundaryList[bid + 1]):
                idx = idx + 1
                string = '%s\t%s\n' % (
                    cv_dist[idx], hist_counts[idx] / TotalCountsInBox[i])
                rawBoxNormalizedHistogram.write(string)

                hist_errors.append(ComputeHistError(hist_counts[idx],
                                   TotalCountsInBox[i],
                                   i, boxFreeEnergy, boxError))
                hist_counts[idx] = boxProbability[i] * \
                    hist_counts[idx] / TotalCountsInBox[i]
                normalized_hist.append(hist_counts[idx])
                string = '%s\t%s\n' % (cv_dist[idx], hist_counts[idx])
                normalizedHistogram.write(string)
            idx += 1
    finally:
        normalizedHistogram.close()
        rawBoxNormalizedHistogram.close()
    #Plot histograms
    plot2D(cv_dist, normalized_hist, output_dir +
           "/normalizedHist.png", "Distance along CV / Angstrom",
           "$p(\\rho)$", ls='-')
    plot2D(cv_dist, np.cumsum(normalized_hist), output_dir +
           "/normalizedHistSum.png",
           "Distance along CV / Angstrom",
           "$P(\\rho)$", ls='-')
    print("Histogram errors: ", hist_errors)

    print("\nThe raw histogram with each box normalized to 1 is in ",
          rawBoxNormalizedHistogram.name)
    print("\nThe fully corrected & normalized histogram is printed in ",
          normalizedHistogram.name)

    #print out the final free energy surface
    finalFreeEnergy = open(output_dir + '/finalFreeEnergy.txt', 'w')
    free_energy = []
    try:
        for i in range(0, len(hist_counts)):
            if (hist_counts[i] != 0):
                free_energy.append(-1.0 * math.log(
                    hist_counts[i] / np.linalg.norm(
                        hist_plane_points[i + 1] - hist_plane_points[i])))

                string = '%s\t%s\n' % (cv_dist[i], free_energy[-1])
                finalFreeEnergy.write(string)
            else:
                print(
                    "\nThe final free energy surface cannot be constructed",
                    "because of zeros in the Histogram...\n")
                sys.exit()
    finally:
        finalFreeEnergy.close()
    print('\nThe final PMF is printed out to %s\n' % (finalFreeEnergy.name))
    free_energy_plot_file = output_dir + '/finalFreeEnergy.png'
    print('PMF graph plotted to ', free_energy_plot_file)

    #plot the free energy surface
    plot2D(cv_dist, free_energy, free_energy_plot_file, 'Distance along CV',
           'RT', marker='o', color=tableau20[0], ms=8, ls='-')


def ReadThresholds(threshold_file, thresholds_lower, thresholds_upper):
    """

    Reads thresholds from file in form "boxid lower upper"
    and updates thresholds array with those found.

    """

    f = open(threshold_file, 'r')
    try:
        for line in f:
            fields = line.split()
            assert len(
                fields) == 3, "Expected 3 fields in line in threshold file"
            box = int(fields[0])
            lower = float(fields[1])
            upper = float(fields[2])
            assert len(
                thresholds_upper) > box, "Box specified in thresholds file is outside range"
            assert len(
                thresholds_lower) > box, "Box specified in thresholds file is outside range"
            thresholds_lower[box] = lower
            thresholds_upper[box] = upper
    finally:
        f.close()
    for i in range(len(thresholds_lower)):
        print("First passage times less than ", thresholds_lower[
              i], "for box ", i, " to box ", i - 1, "will be neglected")
        print("First passage times less than ", thresholds_upper[
              i], "for box ", i, " to box ", i + 1, "will be neglected")
    return thresholds_lower, thresholds_upper


def ComputeBoxFreeEnergies(kLowerList, kUpperList, lower_fpts, upper_fpts,
                           BoxLowerID, BoxUpperID, plane_points):

    # calculate the box averaged Free Energy distribution
    boxFreeEnergy = [0.0]
    box_id = BoxLowerID
    box_energy_var = [0.0]
    for i in range(BoxLowerID, BoxUpperID):
        #compute box free energy
        if kLowerList[i + 1] > 0:
            Keq = kUpperList[i] / kLowerList[i + 1]
            if (Keq > 0.0):
                dG = -1.0 * math.log(Keq)
            else:
                print("\nKeq is zero between box ", i,
                      " and ", i + 1, ".")
                return
        else:
            print("kLower for box", i + 1, "is not greater than 0")
            return
        #compute variance in box free energy calculation
        print("Computing variance for box:", i)
        var = ComputeBoxError(kLowerList[i+1], kUpperList[i], lower_fpts[i+1],
                              upper_fpts[i])
        box_energy_var.append(var)
        #cumulate box free energy
        boxFreeEnergy.append(boxFreeEnergy[-1] + dG)

    #compute standard error of each box free energy difference
    box_energy_std = [math.sqrt(v) for v in box_energy_var]
    #compute cumulative box_free_energy std deviation.
    #via sqrt of summation of previous square of error for each box
    box_energy_std_cuml = [math.sqrt(v) for v in np.cumsum(box_energy_var)]

   #calculate the unnormalized box averaged Probability distribution
    Z = 0.0
    boxProbability = []
    for i in range(0, len(boxFreeEnergy)):
        boxProbability.append(math.exp(-1.0 * boxFreeEnergy[i]))
        Z = Z + boxProbability[i]

    #normalize the box averaged Probability distribution
    for i in range(0, len(boxFreeEnergy)):
        boxProbability[i] = boxProbability[i] / Z

    #print outs
    free_energy_file = output_dir + "/boxFreeEnergy.txt"
    boxFreeEnergyFile = open(free_energy_file, 'w')
    box_probability_file = output_dir + "/boxProbability.txt"
    boxProbabilityFile = open(box_probability_file, 'w')
    #compute the distance through the path along the planes
    box_lines = ComputeDistancesAlongCV(plane_points)
    mid_point_dist = []
    #compute the midpoints between box lines
    for a, b in pairwise(box_lines):
        mid_point_dist.append((b-a)*0.5 + a)
    print(box_lines)
    print(mid_point_dist)
    print("\nBox averaged energies/RT along path through plane points:")
    for i in range(0, len(boxFreeEnergy)):
        print('\tDistance %s: %f \t %f' %
              (mid_point_dist[i], boxFreeEnergy[i], boxProbability[i]))

    boxFreeEnergyFile.write("Distance\tBoxFreeEnergy\tError\tCumulative\n")
    for x, y, z, w in zip(mid_point_dist, boxFreeEnergy, box_energy_std,
                          box_energy_std_cuml):
        boxFreeEnergyFile.write('{0}\t{1}\t{2}\t{3}\n'.format(x, y, z, w))
    for x, y in zip(mid_point_dist, boxProbability):
        boxProbabilityFile.write('{0}\t{1}\n'.format(x, y))

    boxFreeEnergyFile.close()
    boxProbabilityFile.close()

    #plot box free energies with the two different error bars
    if plane_points is not None:
        x = mid_point_dist
        print(x)
        print(box_lines)
    else:
        x = range(len(boxFreeEnergy))
        box_lines = []
    plot2D(x, boxFreeEnergy, output_dir + "/boxFreeEnergy.png",
           "Distance along CV / Angstrom", "RT",
           color=tableau20[0],
           error=box_energy_std,
           ecolor=tableau20[4], capthick=2, ms=2, lw=2)
    plot2D(x, boxFreeEnergy, output_dir +
           "/boxFreeEnergyCumError.png", "Distance along CV / Angstrom", "RT",
           color=tableau20[0], error=
           box_energy_std_cuml, ecolor=tableau20[0], capthick=2, ms=2, lw=2)
    plot2D(x, np.cumsum(boxProbability), output_dir +
           "/boxProbability.png", "Distance along CV / Angstrom ", "$P_n$",
           ylimits=[-0.1, 1.1],
           color=tableau20[0],
           ms=10, lw=2)
    return boxFreeEnergy, boxProbability, box_energy_std_cuml


def store_bounds_json(json_file, planes, points, centers, hists):
    """
    Stores bounds as json file bxd.

    Args:
        json_file (str) : path and file name of json bounds file
        planes (list) : list of n-dimensional planes
        points (list) : list of points on planes
        centers (list): list of midpoints between planes
        hist (list) : list of whether plane is part of histogram or not

    """

    assert isinstance(json_file, str), "Json file is not a string"

    data = {}
    bounds = []
    centers = [None] + centers
    for plane, point, center, hist in zip(planes, points, centers, hists):
        bound = {}
        bound["plane"] = plane.tolist()
        bound["point"] = point.tolist()
        if center is not None:
            bound["center"] = center.tolist()
        bound["hist"] = hist
        bounds.append(bound)
    data["bounds"] = bounds
    data_json_string = json.dumps(data, indent=4)

    f = open(json_file, 'w')
    try:
        f.write(data_json_string)
    finally:
        f.close()


def read_bounds_json(json_file, reverse_bounds=False):
    """
    Reads the bounds json file outputted by adaptive bxd.
    Will then compute centers of boxes from the plane points

    Args:
        json_file (str) : path and file name of json bounds file
        ndim (int) : number of dimensions of CVs
        reverse_bounds=False (bool) : whether to reverse the direction of
            bounds
    """

    assert isinstance(json_file, str), "Json_file is not a string"

    json_data = open(json_file).read()
    data = json.loads(json_data)

    planes = []
    plane_points = []
    box_centers = []
    bounds = data["bounds"]
    for b in bounds:
        p = b["plane"]
        plane = np.array([float(x) for x in p])
        point_str = b["point"]
        point = np.array([float(x) for x in point_str])
        planes.append(plane)
        plane_points.append(point)
    for i in range(len(plane_points) - 1):
        box_centers.append((plane_points[i] + plane_points[i + 1]) * 0.5)

    if(reverse_bounds):
        planes.reverse()
        box_centers.reverse()
        plane_points.reverse()

    return planes, plane_points


def read_centers(centers_file, ndim):
    """
    Reads the centers of the trajectory within each box from a file
    The order of the centers is assumed to match the order of the BXD
    boundaries defined in the boundary file, and are expected to have the
    same dimensionality as the plane

    Args:
        centers_file (str) : path and filename
        ndim (int) : Number of dimensions of CVs

    Returns:
        list : list of centers
    """

    assert isinstance(
        centers_file, str), "centers_file is not a string: %r " % centers_file
    assert isinstance(
        ndim, int), "Number of dimensions is not an integer: %r" % ndim

    f = open(centers_file, 'r')
    line_no = 1
    centers = []
    try:
        for line in f:
            fields = line.split()
            assert len(fields) == ndim, 'Length of line ' + str(line_no) + \
                ' is not equal to number of dimensions (' + str(
                    ndim) + ')'
            centers.append(np.array([float(x) for x in fields]))
    finally:
        f.close()
    return centers


def IsInsideBox(rho, lowerbound, upperbound, ndim, debug=False):
    """

    Determines whether the passed CV value is within a box bounded by
    lowerbound and upperbound

    Args:
        rho (array/list, len:ndim+1: value of reaction coordinate
        lowerbound (array/list, len:ndim+1) : lower hyperplane
        upperbound (array/list, len:ndim+1) : upper hyperplane
        ndim (int) : number of dimensions of CV space

    Returns:
        bool : True if inside box, False otherwise
    """

    rho = np.asarray(rho)
    lowerbound = np.asarray(lowerbound)
    upperbound = np.asarray(upperbound)
    assert isinstance(
        ndim, int), "Number of dimensions is not an integer: %r" % ndim
    assert len(
        rho) == ndim, "Number of dimensions of CV does not equal ndim: %r" % len(rho)
    assert len(lowerbound) == ndim + \
        1, "Hyperplane dimension does not equal ndim + 1: %r" % len(lowerbound)
    assert len(upperbound) == ndim + \
        1, "Hyperplane dimension does not equal ndim + 1: %r" % len(upperbound)
    trace = False
    if trace:
        set_trace()
    # Compute the signed distance from the plane to rho from each bound
    dist_lower = np.dot(rho, lowerbound[:ndim]) + lowerbound[ndim]
    dist_upper = np.dot(rho, upperbound[:ndim]) + upperbound[ndim]

    """
    There is a potential floating point error here when the distance to
    the plane is very near zero.
    Because of the precision of the plane outputted to file,
    it is possible that the sign of distance to the plane could be incorrect.
    In this case, either an inversion will occur, or the next step will place
    us more firmly in the the next box. This function will return None in this
    case, and the calling function will make a decision.
    """
    threshold = 1.0e-06
    if abs(dist_lower) < threshold or abs(dist_upper) < threshold:
        return None

    if debug:
        print("Comparing rho:", rho, " to lowerbound: ",
              lowerbound, ", upperbound: ", upperbound)
        print("Distance to lower bound: ", dist_lower)
        print("Distance to upper bound: ", dist_upper)
    """
    The sign of the distance tells us which direction from the plane the point
    is. The sign is different between the lower and upper bound distances then
    the point is between the boundaries.
    """
    sign = lambda x: math.copysign(1, x)
    if sign(dist_lower) != sign(dist_upper):
        return True
    else:
        return False


def UpdateBoundaryFPT(FPT_list, boundary_hit, boundary_test, found_hit,
                      hit_time, last_hit_time, numhits,
                      current_box_id, lower, line):
    """
    Given the boundary just hit and the bound to test, determines whether this
    boundary was hit and updates.

    If it has not been hit before it will be marked as found and hit time
    recorded.
    If it has been hit before the passage time will be calculated and appended
    to FPT_list

    Args:
        FPT_list (list) : List of FPTs for bound
        boundary_hit (list/array) : Boundary which has been hit
        boundary_test (list/array) : Boundary to be tested against
        found_hit (bool) : Whether already hit boundary
        hit_time : Time boundary was hit
        last_hit_time : Last time boundary_test was hit
        numhits : Number of times boundary_test has been hit
    Returns
        FPT_list : Updated with new FPT if necessary
        found_hit : Updated if boundary_test hit
        last_hit_time : Updated if boundary_test hit before
        numhits : Updated if boundary_test hit
    """

    np.array(boundary_hit)
    np.array(boundary_test)

    if np.allclose(boundary_hit, boundary_test):
        if(found_hit):
            passageTime = hit_time - last_hit_time
            # Negative passage time will occur if files concatenated
            if(passageTime <= 0):
                found_hit = False
            else:
                FPT_list.append(passageTime)
                last_hit_time = hit_time
                numhits = numhits + 1
        else:
            last_hit_time = hit_time
            found_hit = True
    return FPT_list, found_hit, last_hit_time, numhits


def GetHistogram(histogram_files, trajectory_files, bin_planes, bin_centers,
                 ndim):

    """
    Given directories to existing histogram files from previous run
    and path to trajectory file, will populate a histogram.
    WARNING: Assumes that the histogram bins are the same between runs!
    """

    nbins = len(bin_planes) - 1
    counts = [0.0] * (nbins)
    if trajectory_files is not None:
        for trajectory in trajectory_files:
            print("Filling histogram from trajectory file",
                  trajectory)
            numlines = GetNumLinesInFile(trajectory)
            new_counts = FillHistogram(trajectory, bin_planes,
                                       bin_centers, ndim, numlines)
            print("debug: len new counts: ",len(new_counts))
            print("debug: len counts: ", len(counts))
            if len(new_counts) != len(counts):
                print("Error! The number of bins from this file does not",
                      "match the rest of the input. Skipping this file.")
            else:
                for i in range(len(counts)):
                    counts[i] += new_counts[i]
    if histogram_files is not None:
        for hist_file in histogram_files:
            print("Filling histogram from precomputed ",
                  "histogram file ", hist_file)
            new_counts = ReadHistogram(hist_file)
            print("debug: len new counts: ",len(new_counts))
            print("debug: len counts: ", len(counts))
            if len(new_counts) != len(counts):
                print("Error! The number of bins from this file does not",
                      "match the rest of the input. Skipping this file.")
            else:
                for i in range(len(counts)):
                    counts[i] += new_counts[i]

    # write out the raw histogram
    rawHistogram = open(output_dir + '/rawHistogram.txt', 'w')
    cv_dist = ComputeDistancesAlongCV(bin_centers)
    try:
        for x, c in zip(cv_dist, counts):
            string = '%s\t%s\n' % (x, c)
            rawHistogram.write(string)
    finally:
        rawHistogram.close()
    print("Raw histogram printed out to rawHistogram.txt")

    return counts


def ReadHistogram(filename):
    """
    Reads a precomputed histogram from file,
    produced by FillHistogram on a previous run
    """
    hist_file = open(filename)
    counts = []
    try:
        for line in hist_file:
            fields = line.split()
            counts.append(float(fields[1]))
    finally:
        hist_file.close()
    return counts


def FillHistogram(opfilename, bin_planes, bin_centers, ndim, numlines):

    opfile = open(opfilename, 'r')
    n_planes = len(bin_planes)
    nbins = n_planes - 1
    counts = [0.0] * (nbins)
    line = 0
    bin_pairs = []
    current_bin = None
    bin_num = 0
    old_method_bin = None
    for a, b in pairwise(bin_planes):
        bin_pairs.append(tuple([a, b]))
    print("Making a histogram out of the bisected boundaries...")
    #set up progress bar
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=numlines,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()
    try:
        for linestring in opfile:
            line = line + 1
            if progress_bar:
                bar.update(line)
            else:
                if line % 1000 == 0:
                    print('.', end="", flush=True)
            linelist = linestring.split()
            # Skip lines where inversion occurred, they should not be counted
            if len(linelist) != ndim + 1:
                continue
            try:
                cv = [float(x) for x in linelist[1:ndim + 1]]
            except ValueError:
                break

            find_bin = False
            in_box = False
            #If current bin has been set, then check if in it
            if current_bin is not None:
                in_box = IsInsideBox(cv, current_bin[0], current_bin[1], ndim)
            #if in current box, add it to counts
            if in_box: 
                counts[bin_num] += 1
            #if not in box, then need to find the bin
            if not in_box: 
                find_bin = True
            if find_bin:
                found_bin = False
                i = 0 
                for pair in bin_pairs:
                    if not found_bin: 
                        if IsInsideBox(cv, pair[0], pair[1], ndim):
                            bin_num = i
                            counts[bin_num] += 1
                            found_bin = True
                            current_bin = pair
                            break
                        else:
                            i = i + 1
                    else:
                        break

    finally:
        opfile.close()

    return counts


def plot2D(x, y, outputfile, xlabel, ylabel, xlimits=None,
           ylimits=None, ylines=None, **kwargs):
    """
    Useful little plotter. Any arguments to ax.plot can be added to kwargs
    """
    # You typically want your plot to be ~1.33x wider than tall.
    # Common sizes: (10, 7.5) and (12, 9)
    fig = plt.figure(figsize=(9, 9))

    ax = fig.add_subplot(111)

    # Ensure that the axis ticks only show up on the bottom and left of the
    # plot.
    # Ticks on the right and top of the plot are generally unnecessary
    # chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='x', labelsize='22')
    ax.tick_params(axis='y', labelsize='22')

    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

    if xlimits is not None:
        ax.set_xlim(xlimits[0], xlimits[1])
    if ylimits is not None:
        ax.set_ylim(ylimits[0], ylimits[1])

    error_bar = False
    error_kwargs = dict()
    if 'error' in kwargs:
        error_bar = True
        err = kwargs['error']
        del kwargs['error']
        error_kwargs['fmt'] = 'none'
        error_options = ['capthick', 'ecolor', 'elinewidth', 'capsize']
        for s in error_options:
            if s in kwargs:
                error_kwargs[s] = kwargs[s]
                del kwargs[s]

    kwargs['lw'] = 1.5
    kwargs['alpha'] = 1.0
    kwargs['marker'] = 'o'
    print(kwargs)
    if error_bar:
        ax.errorbar(x, y, yerr=err, **error_kwargs)
    ax.plot(x, y, **kwargs)
    if ylines is not None:
        y = ax.get_ylim()
        for x_val in ylines:
            x = [x_val]*len(y)
            ax.plot(x, y, color="black", ls='--')
    if 'label' in kwargs:
        ax.legend()

    plt.savefig(outputfile, bbox_inches="tight")
    plt.close()


def plotDecay(decay_array, outputfile, label, color_id):

    keylist = list(decay_array.keys())
    keylist.sort()
    values = [decay_array[k] for k in keylist]

    plot2D(keylist, values, outputfile, "FPT", "R(t)", label=label,
           color=tableau20[color_id])


def ReadFPTs(fpt_dir, lower_box, upper_box, nboxes):
    """
    Read FPTs that have been precomputed by GetFirstPassageTimesAllBoxes
    """
    LowerFPTs = [[] for x in range(nboxes)]
    UpperFPTs = [[] for x in range(nboxes)]
    print("Reading previously generated FPTs from directory: ", fpt_dir)
    for boxIdx in range(lower_box, upper_box + 1):
        lower_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx - 1)
        lowerFile = open(lower_fpt_name)
        for fpt in lowerFile:
            LowerFPTs[boxIdx].append(float(fpt))
        lowerFile.close()

        upper_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
        upperFile = open(upper_fpt_name)
        for fpt in upperFile:
            UpperFPTs[boxIdx].append(float(fpt))
        upperFile.close()

    return LowerFPTs, UpperFPTs


def GetNumLinesInFile(opfilename):
    print("Counting number of lines in file...")
    num_lines = sum(1 for line in open(opfilename, 'r'))
    print("There are ", num_lines, " lines in the file ", opfilename)
    return num_lines


def GetFPTs(fpt_dirs, trajectory_files, BoundaryList, BoxLowerID, BoxUpperID, Ndim, Nsweeps,
            nBoxes):

    """
    Given directories to existing fpts directories and path to trajectory file,
    will gather the FPTs from all sources
    """

    fpt_lower_list = [[] for x in range(nBoxes)]
    fpt_upper_list = [[] for x in range(nBoxes)]
    if trajectory_files is not None:
        for trajectory in trajectory_files:
            print("Getting passage times for all boxes from trajectory file ",
                  trajectory)
            numlines = GetNumLinesInFile(trajectory)
            lower_list, upper_list = GetFirstPassageTimesAllBoxes(
                trajectory, BoundaryList, BoxLowerID,
                BoxUpperID, Ndim, Nsweeps, numlines)
            for i in range(nBoxes):
                fpt_lower_list[i] += lower_list[i]
                fpt_upper_list[i] += upper_list[i]
    if fpt_dirs is not None:
        for fpt_dir in fpt_dirs:
            print("Getting passage times for all boxes from precomputed ",
                  "FPT directory ", fpt_dir)
            lower_list, upper_list = ReadFPTs(fpt_dir, BoxLowerID,
                                              BoxUpperID, nBoxes)
            for i in range(nBoxes):
                fpt_lower_list[i] += lower_list[i]
                fpt_upper_list[i] += upper_list[i]

    # output FPTs to save repeat analysis time
    for boxIdx in range(BoxLowerID, BoxUpperID + 1):
        fpt_dir = output_dir + "/FPT_arrays"
        make_sure_path_exists(fpt_dir)

        lower_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx - 1)
        lowerFile = open(lower_fpt_name, 'w')
        for fpt in fpt_lower_list[boxIdx]:
            print(str(fpt), file=lowerFile)
        lowerFile.close()

        upper_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
        upperFile = open(upper_fpt_name, 'w')
        for fpt in fpt_upper_list[boxIdx]:
            print(str(fpt), file=upperFile)
        upperFile.close()

    return fpt_lower_list, fpt_upper_list


def GetFirstPassageTimesAllBoxes(opfilename, bounds, LowerBoxID, UpperBoxID,
                                 ndim, nsweeps, numlines):

    #first get the number of lines in the file

    opfile = open(opfilename, 'r')
    line = 0
    StepsInsideBox = 1
    nboxes = len(bounds) - 1
    LastUpperHitTime = [0.0] * nboxes
    LastLowerHitTime = [0.0] * nboxes
    FoundFirstUpperHit = [False] * nboxes
    FoundFirstLowerHit = [False] * nboxes
    InsideTheBox = False
    current_box_id = -1
    NumUpperHits = [0] * nboxes
    NumLowerHits = [0] * nboxes

    FoundFirstBox = False
    sweep_count = -1

#   initialize two lists
    UpperFPTs = [[] for x in range(nboxes)]
    LowerFPTs = [[] for x in range(nboxes)]
    hits = 0
    #set up progress bar
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=numlines,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()
    try:
        for linestring in opfile:
            line = line + 1

            #update progress bar
            if progress_bar:
                bar.update(line)
            linelist = linestring.split()
            """
            assert len(linelist) >= ndim + 1, 'Length of line ' + \
                str(line) + ' less than ndim +1 (' + str(ndim + 1) + ')'
            """
            if len(linelist) < ndim + 1:
                print("Odd line, skipping: ", line)
                continue
            try:
                time = float(linelist[0])
                distance = [float(x) for x in linelist[1:ndim + 1]]
            except ValueError:
                print("Error reading line, skipping ", line)
                break
            new_box_id = -1
            debug = False

            find_box = False
            if current_box_id == -1:
                find_box = True
            elif IsInsideBox(distance, bounds[current_box_id], bounds[current_box_id + 1], ndim, debug):
                new_box_id = current_box_id
            else:
                find_box = True
            if find_box:
                #first try boxes adjacent to current one, then the rest of the list
                box_list = []
                if current_box_id != -1:
                    box_list += [max(current_box_id - 1, LowerBoxID),
                                 min(current_box_id + 1, UpperBoxID + 1)]
                box_list += range(LowerBoxID, UpperBoxID+1)
                for i in box_list:
                    in_box = IsInsideBox(
                        distance, bounds[i], bounds[i + 1], ndim, debug)
                    if in_box is None:
                        new_box_id = current_box_id
                        break
                    elif in_box:
                        new_box_id = i
                        break

            if current_box_id != new_box_id:
                if current_box_id >= 0:
                    FoundFirstLowerHit[current_box_id] = False
                    FoundFirstUpperHit[current_box_id] = False
                if new_box_id == LowerBoxID:
                    sweep_count += 1
                    if nsweeps is not None and sweep_count > nsweeps:
                        print("Reached {0} sweep, stopping".format(nsweeps))
                        break
                current_box_id = new_box_id
                hits = 0
            if current_box_id >= 0:
                StepsInsideBox = StepsInsideBox + 1
                # Set up if first time enter box
                if(not InsideTheBox):
                    InsideTheBox = True

                # If there is an inversion boundary
                if((len(linelist)) > ndim + 1):
                    hits += 1
                    # if hits < equilibration_hits:
                    #    print("Discarding hit")
                    #    continue

                    # Make sure length of line is correct
                    if len(linelist) != 2 + 2 * ndim:
                        print("Odd line, skipping:", line)
                        continue
                    InversionBoundary = np.array(
                        [float(x) for x in linelist[ndim + 1:]])
                    # Update upper boundary FPT
                    u, f, t, n = UpdateBoundaryFPT(
                        UpperFPTs[current_box_id],
                        InversionBoundary,
                        bounds[current_box_id + 1],
                        FoundFirstUpperHit[current_box_id],
                        time,
                        LastUpperHitTime[current_box_id],
                        NumUpperHits[current_box_id],
                        current_box_id,
                        False,
                        line
                    )
                    UpperFPTs[current_box_id] = u
                    FoundFirstUpperHit[current_box_id] = f
                    LastUpperHitTime[current_box_id] = t
                    NumUpperHits[current_box_id] = n

                    # Update lower boundary FPT
                    u, f, t, n = UpdateBoundaryFPT(
                        LowerFPTs[current_box_id],
                        InversionBoundary,
                        bounds[current_box_id],
                        FoundFirstLowerHit[current_box_id],
                        time,
                        LastLowerHitTime[current_box_id],
                        NumLowerHits[current_box_id],
                        current_box_id,
                        True,
                        line
                    )
                    LowerFPTs[current_box_id] = u
                    FoundFirstLowerHit[current_box_id] = f
                    LastLowerHitTime[current_box_id] = t
                    NumLowerHits[current_box_id] = n

    finally:
        opfile.close()
    return LowerFPTs, UpperFPTs


def HistogramFPTs(Upper, Lower, bin):
    """
    Produce histograms of upper and lower FPTs
    """

    Upper = np.array(Upper)
    Lower = np.array(Lower)
    upper_str = str(bin) + " - Upper "

    upper_bins = 10 ** np.linspace(np.log10(min(Upper)), np.log10(max(Upper)))
    lower_bins = 10 ** np.linspace(np.log10(min(Lower)), np.log10(max(Lower)))

    ab_min = min(min(Upper), min(Lower))
    ab_max = max(max(Upper), max(Lower))
    f, axarr = plt.subplots(2)
    n, b, p = axarr[0].hist(
        Upper, upper_bins, alpha=0.5, label=upper_str, color='g')
    axarr[0].set_xscale('log')
    axarr[0].set_xlim(ab_min, ab_max)
    axarr[0].set_ylabel("Count")
    axarr[0].legend()

    lower_str = str(bin) + " - Lower "

    n, b, p = axarr[1].hist(
        Lower, lower_bins, alpha=0.5, label=lower_str, color='b')
    axarr[1].set_xscale('log')
    axarr[1].set_xlim(ab_min, ab_max)
    axarr[1].set_ylabel("Count")
    axarr[1].legend()
    FPT_hist_path = output_dir + "/FPT_histograms"
    make_sure_path_exists(FPT_hist_path)
    plt.savefig(
        FPT_hist_path + "/" + str(bin).zfill(2) + ".png", bbox_inches="tight")
    plt.close()


def reject_outliers(Upper, Lower, bin, threshold_lower=0.0,
                    threshold_upper=0.0):
    """
    Removes FPTs lower than threshold_lower and upper
    """

    Upper = np.array(Upper)
    Lower = np.array(Lower)
    if len(Upper) == 0 or len(Lower) == 0:
        return Upper, Lower
    Upper = Upper[Upper > threshold_upper]

    Lower = Lower[Lower > threshold_lower]

    if len(Upper) == 0:
        raise ValueError("upper FPTs empty for box ", bin,
                         " after threshold of ", threshold_upper, " applied.",
                         "Check decay trace!")
    if len(Lower) == 0:
        raise ValueError("lower FPTs empty for box ", bin,
                         " after threshold of ", threshold_lower, " applied.",
                         "Check decay trace!")
    #plot the FPT histogram
    #HistogramFPTs(Upper, Lower, bin)
    return Upper, Lower


def ComputeBoxError(k_lower, k_upper, lower_FPTs, upper_FPTs):
    """
    Computes the propagation of error in MFPTs to the box free energy
    via the formula:

    dG_(i-1,i)^2 = (k_upper)^2 * err_upper^2 +
                   (-k_lower)^2 * err_lower^2

    Returns the error of each box-to-box as well as the cumulative free energy
    box error.
    """

    #compute standard error = sample std/sqrt(N)
    err_upper = np.std(upper_FPTs, ddof=1)/math.sqrt(float(len(upper_FPTs)))
    err_lower = np.std(lower_FPTs, ddof=1)/math.sqrt(float(len(lower_FPTs)))
    #err_upper = np.std(upper_FPTs, ddof=1)
    #err_lower = np.std(lower_FPTs, ddof=1)
    print("Upper std error :", err_upper)
    print("Lower std_err_lower:", err_lower)
    var_dg = math.pow(k_upper * err_upper, 2) + math.pow(k_lower * err_lower, 2)
    print("dG variance:", var_dg)
    std_dg = math.sqrt(var_dg)
    return var_dg


def ComputeMFPTs(LowerFPTs, UpperFPTs, bounds, LowerBoxID, UpperBoxID,
                 recrossTime_lower, recrossTime_upper):

    """
    Computes the MFPTs from the list of lower and upper FPTs
    for each box
    """
    ctr = 0
#   calculate Mean First Passage Time
#   MPFTS will only be averaged if they are greater than recrossTime
    nboxes = len(bounds) - 1
    klower = [0] * nboxes
    kupper = [0] * nboxes
    for boxIdx in range(LowerBoxID, UpperBoxID + 1):

        lower_fpts = LowerFPTs[boxIdx]
        upper_fpts = UpperFPTs[boxIdx]

        upper_fpts, lower_fpts = reject_outliers(
            upper_fpts, lower_fpts, boxIdx, recrossTime_lower[boxIdx],
            recrossTime_upper[boxIdx])

        Initial = len(lower_fpts)
        decay_dir = output_dir + "/decays"
        make_sure_path_exists(decay_dir)
        if (Initial != 0):
            MFPT = np.mean(lower_fpts)
            klower[boxIdx] = 1 / MFPT

            lowerFileName = decay_dir + '/%sto%s.txt' % (boxIdx, boxIdx - 1)
            print("\tMFPT box %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx - 1, MFPT, lowerFileName))
            print("\tbox %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx - 1, klower[boxIdx], lowerFileName))

            #calculate Decay Profile
            LowerDecayArray = CalculateDecay(lower_fpts)
            lowerDecayPlotName = decay_dir + \
                '/%sto%s.png' % (boxIdx, boxIdx - 1)
            plotlabel = str(boxIdx) + " to " + str(boxIdx - 1)
            color = boxIdx

            #Plot the decay profile
            plotDecay(LowerDecayArray, lowerDecayPlotName, plotlabel, color)

            keylist = list(LowerDecayArray.keys())
            keylist.sort()
            lowerFile = open(lowerFileName, 'w')
            lowerFile.write('0.0\t%s\n' % (Initial))
            # plotDecayArray()
            for key in keylist:
                lowerFile.write(
                    '%s\t%s\n' % (str(key), (LowerDecayArray[key])))
            lowerFile.close()
        else:
            print("\tbox %s to box %s = N/A " % (boxIdx, boxIdx - 1))
            klower[boxIdx] = 0.0

        # Upper FPTs
        Initial = len(upper_fpts)

        if (Initial != 0):
            MFPT = np.mean(upper_fpts)
            kupper[boxIdx] = 1 / MFPT

            upperFileName = decay_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
            print("\tMFPT box %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx + 1, MFPT, upperFileName))
            print("\tbox %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx + 1, kupper[boxIdx], upperFileName))

            #calculate Decay Profile

            UpperDecayArray = CalculateDecay(upper_fpts)
            upperDecayPlotName = decay_dir + \
                '/%sto%s.png' % (boxIdx, boxIdx + 1)
            plotlabel = str(boxIdx) + " to " + str(boxIdx + 1)
            color = boxIdx

            plotDecay(UpperDecayArray, upperDecayPlotName, plotlabel, color)

            keylist = list(UpperDecayArray.keys())
            keylist.sort()
            upperFile = open(upperFileName, 'w')
            upperFile.write('0.0\t%s\n' % (Initial))
            for key in keylist:
                upperFile.write(
                    '%s\t%s\n' % (str(key), (UpperDecayArray[key])))
            upperFile.close()
        else:
            print("\tbox %s to box %s = N/A " % (boxIdx, boxIdx + 1))
            kupper[boxIdx] = 0.0
    return klower, kupper


def CalculateDecay(PassageTimeArray):
    """
    Produces a decay trace for a particular list of passage times
    """
    Elements = len(PassageTimeArray)
    PassageTimeArray.sort()
    #initialize a Map
    DecayArray = {}
    NumberElementsLeft = Elements

    for i in range(0, Elements):
        NumberElementsLeft = NumberElementsLeft - 1
        DecayArray[PassageTimeArray[i]] = NumberElementsLeft

    return DecayArray

if __name__ == "__main__":
    """
    This analysis script calculates MFPTs and free energies for a BXD
    run with multidimensional CVs.

    The script can be run in multiple ways and has more options than prior
    versions.
    In the simplest case, run via

    python BXDanalysis_2D.py bounds.json ndim --trajectory bxd.juj

    Here, bounds.json is the json file with the bxd bounds in it,
    ndim is the number of dimensions of CV space
    and bxd.juj is the bxd output file from a trajectory.

    After running, BXD will create a new directory called analysis
    with the following files:

    boxFreeEnergy.dat - Box to box free energies along the path through
                        the boxes
    boxFreeEnergy.png - Plot of above
    boxFreeEnergyCumError.png - plot of above with error bars.
    rawHistogram.txt - Raw histogram values used to produce free energies.
                       Can be passed to the script with --
    rawBoxNormalizedHistogram.txt - Histogram values normalized
                                    within each box.
    normalizedHistogram.txt - Normalized histogram across rho.
    normalizedHist.png - plot of p(rho)
    normalizedHistSum.png - plot of P(rho)
    finalFreeEnergy.txt - The computed free energies at the resolution of
                          the histogram
    finalFreeEnergy.png - Plot of G(rho)
    FPT_arrays - Text files of the FPTs recorded, can be used with --fpt_dir to
                 rerun analysis quickly.
    FPT_histograms - Plots of the distributions of FPTs in each box.
    decays - Plots and text files of decay traces of each box.

    To run another run using precomputed data, run as follows:

    python BXDanalysis_2D.py bounds.json ndim --fpt_dir path/to/fpts
           --hist_file path/to/rawHistogram.txt
    """

    argparser = argparse.ArgumentParser()

    # Required arguments
    argparser.add_argument("bounds", help="File containing the BXD bounds")
    argparser.add_argument(
        "ndim", help="Number of dimensions of BXD boundaries", type=int)
    argparser.add_argument("--trajectories", nargs='+', help="Path of BXD output file(s)")
    argparser.add_argument("--fpt_dirs", nargs='+', help=("Directory/Directories of FPTs",
                           " outputted by this script"))
    argparser.add_argument("--hist_files", nargs='+', help=("Directory/Directories to raw histogram file",
                           " outputted by this script"))
    argparser.add_argument(
        "--MFPTthreshold",
        help=("FPT threshold, any FPTs less than the given value",
              "will be discarded. Default 0.0"), type=float)
    argparser.add_argument(
        "--threshold_file",
        help=("More detailed FPT thresholds for each ",
              "boundary. Path to a file containing upper",
              "and lower MFPT thresholds for each box.",
              "Note this overrides the MFPTthreshold setting"), type=str)
    argparser.add_argument(
        "--LowerBoxID", help="Lower BXD box number to be used in analysis",
        type=int)
    argparser.add_argument(
        "--UpperBoxID", help="Upper BXD box number to be used in analysis",
        type=int)
    argparser.add_argument(
        "--nsweeps", help="Number of sweeps across boxes", type=int)
    argparser.add_argument(
        "--histogram_bin_width",
        help="The maximum size for a histogram bin.",
        type=float)
    argparser.add_argument(
        "--reverse_boxes", help="Reverse the direction of the boxes",
        action="store_true")
    argparser.add_argument(
        "--output_dir", help="Output directory, default: analysis")
    args = argparser.parse_args()
    bounds_file = args.bounds
    ndim = args.ndim
    # Optional argument defaults
    lower_box_id = 0
    upper_box_id = None
    mfpt = 0.0
    nsweeps = None
    center_file = None
    max_width = float("inf")
    threshold_file = None
    reverse_boxes = False
    traj_files = None
    fpt_dirs = None
    hist_files = None
    read_fpt = False
    if args.trajectories:
        traj_files = args.trajectories
    if args.fpt_dirs and args.hist_files:
        fpt_dirs = args.fpt_dirs
        hist_files = args.hist_files
        read_fpt = True
    if not args.trajectories and not (args.fpt_dirs and args.hist_files):
        print("Error: Trajectory file was not specificed (--trajectories). ",
              "If rerunning analysis, need both FPT directory and histogram",
              "file flag specfied (--fpt_dirs and --hist_files)")
        sys.exit()
    if args.LowerBoxID:
        lower_box_id = args.LowerBoxID
    if args.UpperBoxID:
        upper_box_id = args.UpperBoxID
    if args.MFPTthreshold:
        mfpt = args.MFPTthreshold
    if args.nsweeps:
        nsweeps = args.nsweeps
    if args.histogram_bin_width:
        max_width = args.histogram_bin_width
    if args.threshold_file:
        threshold_file = args.threshold_file
    if args.reverse_boxes:
        reverse_boxes = True
    if args.output_dir:
        output_dir = args.output_dir

    BXDanalysis(traj_files, bounds_file, ndim,
                MFPTthreshold=mfpt, BoxLowerID=lower_box_id,
                BoxUpperID=upper_box_id, Nsweeps=nsweeps,
                MaxDistance=max_width,
                ThresholdFile=threshold_file, ReverseBounds=reverse_boxes,
                FPTDirectories=fpt_dirs, HistFiles=hist_files)
