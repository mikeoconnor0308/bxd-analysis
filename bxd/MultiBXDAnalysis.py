#!/usr/bin/python
import argparse
import json
import math
import sys

import numpy as np
import bxd.bxd.utility as utils
import bxd.bxd.visualisation as vis
from bxd.bxd.tableau import Tableau

try:
    import progressbar

    progress_bar = True
except ImportError:
    progress_bar = False
    print("Warning: Could not find progressbar module. Try running:  ",
          "\n\tpip install progressbar33\n This will allow for pretty",
          "progress bar output")


def _read_bounds_json(json_file: str, reverse_bounds: bool = False):
    """
    Reads the bounds json file outputted by adaptive bxd.
    Will then compute centers of boxes from the plane points

    :param json_file: json file with bounds.
    :param reverse_bounds: whether to reverse the boundaries upon reading them in.
    :return: boundary planes, boundary points, boundary centers, histogram centers.
    """

    assert isinstance(json_file, str), "Json_file is not a string"

    json_data = open(json_file).read()
    data = json.loads(json_data)

    planes = []
    plane_points = []
    centers = []
    hists = []
    bounds = data["bounds"]
    for b in bounds:
        p = b["plane"]
        plane = np.array([float(x) for x in p])
        point_str = b["point"]
        point = np.array([float(x) for x in point_str])
        planes.append(plane)
        plane_points.append(point)
        if "hist" in b:
            hists.append(b["hist"])
        else:
            hists.append(False)
    for i in range(len(plane_points) - 1):
        centers.append((plane_points[i] + plane_points[i + 1]) * 0.5)

    if (reverse_bounds):
        planes.reverse()
        centers.reverse()
        plane_points.reverse()

    return planes, plane_points, centers, hists


class BXDAnalysis:
    def __init__(self, trajs, bounds, ndim: int,
                 output_dir="analysis",
                 prev_analysis_dir=None,
                 bin_range=[0, None]):
        """
        Initialises a BXD analysis session. 

        :type bin_range: list of length two
        :type prev_analysis_dir: path
        :type output_dir: path
        :type bounds: json file
        :type trajs: list of BXD output files
        """
        self.trajFiles = trajs
        self.boundsFiles = bounds
        self.ndim = ndim
        self.outputDir = output_dir
        self.prevAnalysis = prev_analysis_dir
        self.BoxLowerID = bin_range[0]
        self.BoxUpperID = bin_range[1]
        self.bounds = None
        self.plane_points = None
        self.nBounds = None
        self.hist_planes = None
        self.hist_centers = None
        self.hist_bools = None
        self.hist_plane_points = None
        self.fpt_lower_list = []
        self.fpt_upper_list = []
        self.hist_counts = []

    def read_bounds(self, reverse_bounds=False, max_distance: float = float("inf"), x_cv_label: str = "s_1",
                    y_cv_label: str = "s_2"):
        """
        Reads the bounds from file and generates histogram bins. 
        :param reverse_bounds: Whether to reverse the order of the bounds.
        :param max_distance: Maximum distance between bins.
        :param x_cv_label: Label to use for plots.
        :param y_cv_label: Label to use for plots. 
        :return: BXD bounds, histogram planes, histogram points, histogram centers. 
        """
        # First read all the bounds
        print("Reading bounds from json file:", self.boundsFiles)
        self.bounds, self.plane_points = _read_bounds_json(
            self.boundsFiles, reverse_bounds)[:2]
        n_bounds = len(self.bounds)
        self.nBounds = n_bounds
        assert len(self.bounds) > 0, 'No bounds found!'

        # If no upper box ID has been specified, then set it as last box
        if self.BoxUpperID is None:
            self.BoxUpperID = len(self.bounds) - 2

        # Error checking
        assert 0 <= self.BoxUpperID < len(
            self.bounds) - 1, 'Upper Box ID must be within range 0' \
                              'and number of bounds - 2 (' + str(len(self.bounds) - 2) + ')'
        assert 0 <= self.BoxLowerID <= self.BoxUpperID, 'Lower Box ID must' \
                                                        'be within range 0 and upper box ID (' + str(
            self.BoxUpperID) + ')'

        utils.make_sure_path_exists(output_dir)

        # print some info about the bounds
        for i in range(0, n_bounds - 1):
            print('\tBox ', i, ' spans ',
                  self.bounds[i], ' to ', self.bounds[i + 1])
        print("Performing analysis between boxes", self.BoxLowerID, self.BoxUpperID)

        if self.prevAnalysis is not None:
            bin_file = self.prevAnalysis[0] + "/hist_bounds.json"
            print("Reading histogram bins from file: ", bin_file)
            self.hist_planes, self.hist_plane_points, \
            self.hist_centers, self.hist_bools = _read_bounds_json(bin_file)
        else:
            # now create histogram bin planes
            assert self.plane_points is not None, 'Need points on each plane ' \
                                                  'please make JSON file'
            self.hist_planes, self.hist_centers, \
            self.hist_plane_points, self.hist_bools = CreateBisectionPlanes(
                self.bounds[self.BoxLowerID:(self.BoxUpperID + 2)],
                self.plane_points[self.BoxLowerID:(self.BoxUpperID + 2)], max_distance)
        # Store the generated histogram bounds to file
        print("Writing histogram bounds to hist_bounds.json")
        hist_out = output_dir + "/" + "hist_bounds.json"
        store_bounds_json(hist_out, self.hist_planes, self.hist_plane_points,
                          self.hist_centers, self.hist_bools)
        hist_png_out = output_dir + "/" + "hist_bounds.png"
        print("Plotting the histogram bins to ", hist_png_out)
        vis.plotHistogramBins(self.hist_planes, self.hist_centers, self.hist_plane_points,
                              self.hist_bools, x_cv_label, y_cv_label, hist_png_out, contour_file)

        return self.bounds, self.hist_planes, self.hist_plane_points, self.hist_centers

    def analyse(self, passage_time_limit: float):
        """
        Analyse the trajectories. 
        :param passage_time_limit: Minimum passage time to be considered.  
        :return: FPTs against lower bounds, FPTs against upper bounds, and histogram counts in each bin.
        """
        # Get FPTs and populate histogram.
        self.fpt_lower_list, \
        self.fpt_upper_list, \
        self.hist_counts = GetFPTsAndHist(self.trajFiles,
                                          self.prevAnalysis,
                                          self.bounds,
                                          self.plane_points,
                                          self.BoxLowerID,
                                          self.BoxUpperID,
                                          self.hist_planes,
                                          self.hist_centers,
                                          passage_time_limit,
                                          self.ndim, self.nBounds - 1)
        return self.fpt_lower_list, self.fpt_upper_list, self.hist_counts


output_dir = "analysis"

# 2d histogram for each box
box_counts_twod_x = []
box_counts_twod_y = []

x_cv_label = "C - D Distance (Angstrom)"
y_cv_label = "F - D Distance (Angstrom)"
contour_file = None


def ComputeBoxCenters(plane_points):
    """
    Given a set of points, computes the midpoint of the
    straight line between each consecutive pair of points
    """
    if plane_points is None:
        return None
    box_centers = []
    for (a, b) in utils.get_pairwise_list(plane_points):
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
            new_norm = np.add((1 - fraction) * norm_1, (fraction) * norm_2)
            length = np.linalg.norm(new_norm)
            if (length != 0):
                new_norm /= length
            # compute D
            D = np.dot(new_norm, point)
            # invert sign of D so plane is in form ax + by +cz + d = 0
            D = -D
            plane = np.append(new_norm, D)
        except np.linalg.linalg.LinAlgError:
            D = plane_1[-1] + (plane_2[-1] - plane_1[-1]) * fraction
            plane = np.append(norm_1, D)

    return np.array(plane)


def CreateBisectionPlanes(planes, plane_points, max_distance):
    """
    Given a set of planes and points on those planes, bisects the planes
    such that the distance between the given point on each intermediate plane
    is at most max_distance
    """
    plane_points = np.array(plane_points, dtype=np.float64)
    new_planes = []
    hist_plane_points = []
    hist_centers = []
    hist_bools = []
    for (a, b), (ca, cb) in zip(utils.get_pairwise_list(planes), utils.get_pairwise_list(plane_points)):
        dist = np.linalg.norm(ca - cb)
        Nbisections = 0
        if (dist > max_distance):
            Nbisections = int(dist / max_distance)

        step = 1.0 / (Nbisections + 1)
        frac = step
        bisections = []
        new_planes.append(a)
        hist_plane_points.append(ca)
        hist_bools.append(False)
        if (Nbisections == 0):
            continue
        for i in range(Nbisections):
            new_plane = BisectPlane(a, b, frac)
            bisections.append(new_plane)
            plane_point = ca + frac * (cb - ca)
            plane_point_proj = (np.dot(new_plane[:-1], plane_point) + new_plane[-1]) * new_plane[:-1]
            plane_point = plane_point - plane_point_proj
            hist_plane_points.append(plane_point)
            hist_bools.append(True)
            frac += step
        new_planes += bisections
    new_planes.append(planes[-1])
    hist_bools.append(False)
    hist_plane_points.append(plane_points[-1])
    for (a, b) in utils.get_pairwise_list(hist_plane_points):
        hist_centers.append((a + b) / 2.0)
    return new_planes, hist_centers, hist_plane_points, hist_bools


def ComputeHistError(count, count_in_box, box_id, box_free_energies,
                     box_errors):
    """
    Calculates the error in the high resolution histogram free energy
    calculation
    """

    # estimation for error on histogram counts
    # count_var = math.sqrt(count)
    count_var = count * (1.0 - count / count_in_box)

    # propagation of error formula
    total_box_free_energy = np.sum(box_free_energies)
    error_var = math.pow((1.0 / count_in_box - 1.0 / count), 2) * count_var
    if total_box_free_energy > 0:
        error_var += math.pow((1.0 / total_box_free_energy - 1.0) *
                              box_errors[box_id], 2)

        i = 0
        for box_std in box_errors:
            if i == box_id:
                continue
            error_var += math.pow((1.0 / total_box_free_energy) * box_std, 2)
            i += 1
    # return deviation
    return math.sqrt(error_var)


def BXDanalysis(TrajectoryFiles, BoundsFilename, Ndim,
                MFPTthreshold=0.0, BoxLowerID=0,
                BoxUpperID=None, plot=True,
                MaxDistance=float("inf"),
                BinFile=None,
                ReverseBounds=False, PrevAnalysisDir=None):
    # First read all the bounds
    print("Reading bounds from json file:", BoundsFilename)
    BoundaryList, plane_points = _read_bounds_json(
        BoundsFilename, ReverseBounds)[:2]
    nBounds = len(BoundaryList)
    assert len(BoundaryList) > 0, 'No bounds found!'

    # If no upper box ID has been specified, then set it as last box
    if BoxUpperID is None:
        BoxUpperID = len(BoundaryList) - 2

    # Error checking
    assert 0 <= BoxUpperID < len(
        BoundaryList) - 1, 'Upper Box ID must be within range 0' \
                           'and number of bounds - 2 (' + str(len(BoundaryList) - 2) + ')'
    assert 0 <= BoxLowerID <= BoxUpperID, 'Lower Box ID must' \
                                          'be within range 0 and upper box ID (' + str(BoxUpperID) + ')'
    assert MFPTthreshold >= 0.0, 'MFPT threshold must be greater than zero'

    utils.make_sure_path_exists(output_dir)

    # print some info about the bounds
    for i in range(0, nBounds - 1):
        print('\tBox ', i, ' spans ',
              BoundaryList[i], ' to ', BoundaryList[i + 1])
    print("Performing analysis between boxes", BoxLowerID, BoxUpperID)

    if BinFile is not None:
        print("Reading histogram bins from file: ", BinFile)
        hist_planes, hist_plane_points, \
        hist_centers, hist_bools = _read_bounds_json(BinFile)
    else:
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
    hist_png_out = output_dir + "/" + "hist_bounds.png"
    print("Plotting the histogram bins to ", hist_png_out)
    vis.plotHistogramBins(hist_planes, hist_centers, hist_plane_points,
                          hist_bools, x_cv_label, y_cv_label, hist_png_out, contour_file)
    # Get FPTs and populate histogram.
    fpt_lower_list, \
    fpt_upper_list, \
    hist_counts = GetFPTsAndHist(TrajectoryFiles,
                                 PrevAnalysisDir,
                                 BoundaryList,
                                 plane_points,
                                 BoxLowerID,
                                 BoxUpperID,
                                 hist_planes,
                                 hist_centers,
                                 MFPTthreshold,
                                 Ndim, nBounds - 1)

    print("Computing MFPTs")
    kLowerList, kUpperList = ComputeMFPTs(fpt_lower_list, fpt_upper_list,
                                          BoundaryList, BoxLowerID, BoxUpperID)

    print("Computing Box Free Energies")
    g, p, e = ComputeBoxFreeEnergies(kLowerList, kUpperList, fpt_lower_list,
                                     fpt_upper_list, BoxLowerID,
                                     BoxUpperID, plane_points)
    boxFreeEnergy = g
    boxProbability = p
    boxError = e

    # count the events in each box
    idx = 0
    nBoxes = BoxUpperID + 1 - BoxLowerID
    TotalCountsInBox = [0.0] * nBoxes
    for i in range(nBoxes):
        bid = i + BoxLowerID
        TotalCountsInBox[i] += hist_counts[idx]
        while not np.array_equal(hist_planes[idx + 1], BoundaryList[bid + 1]):
            idx = idx + 1
            TotalCountsInBox[i] += hist_counts[idx]
        idx = idx + 1

    hist_errors = []
    normalized_hist = []
    # normalized the raw histogram & obtain the box probability
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
    # Plot histograms
    vis.plot2D(cv_dist, normalized_hist, output_dir +
               "/normalizedHist.png", "Distance along CV / Angstrom",
               "$p(\\rho)$")
    vis.plot2D(cv_dist, np.cumsum(normalized_hist), output_dir +
               "/normalizedHistSum.png",
               "Distance along CV / Angstrom",
               "$P(\\rho)$")

    print("\nThe raw histogram with each box normalized to 1 is in ",
          rawBoxNormalizedHistogram.name)
    print("\nThe fully corrected & normalized histogram is printed in ",
          normalizedHistogram.name)

    # print out the final free energy surface
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
    # plot the free energy surface
    # choose different colors for each box
    colors = []
    i = 0
    for hist_plane in hist_bools:
        if hist_plane is False:
            i = (i + 1) % len(Tableau.tableau20)
        colors.append(Tableau.tableau20[i])
    vis.plot2D(cv_dist, free_energy, free_energy_plot_file, 'Distance along CV',
               'RT', show_plot=False, s=100, linecolor=Tableau.tableau20[14], c=colors)


def ComputeBoxFreeEnergies(kLowerList, kUpperList, lower_fpts, upper_fpts,
                           BoxLowerID, BoxUpperID, plane_points):
    # calculate the box averaged Free Energy distribution
    boxFreeEnergy = [0.0]
    box_id = BoxLowerID
    box_energy_var = [0.0]
    for i in range(BoxLowerID, BoxUpperID):
        # compute box free energy
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
        # compute variance in box free energy calculation
        var = ComputeBoxError(kLowerList[i + 1], kUpperList[i], lower_fpts[i + 1],
                              upper_fpts[i])
        box_energy_var.append(var)
        # cumulate box free energy
        boxFreeEnergy.append(boxFreeEnergy[-1] + dG)

    # compute standard error of each box free energy difference
    box_energy_std = [math.sqrt(v) for v in box_energy_var]
    # compute cumulative box_free_energy std deviation.
    # via sqrt of summation of previous square of error for each box
    box_energy_std_cuml = [math.sqrt(v) for v in np.cumsum(box_energy_var)]

    # calculate the unnormalized box averaged Probability distribution
    Z = 0.0
    boxProbability = []
    for i in range(0, len(boxFreeEnergy)):
        boxProbability.append(math.exp(-1.0 * boxFreeEnergy[i]))
        Z = Z + boxProbability[i]

    # normalize the box averaged Probability distribution
    for i in range(0, len(boxFreeEnergy)):
        boxProbability[i] = boxProbability[i] / Z

    # print outs
    free_energy_file = output_dir + "/boxFreeEnergy.txt"
    boxFreeEnergyFile = open(free_energy_file, 'w')
    box_probability_file = output_dir + "/boxProbability.txt"
    boxProbabilityFile = open(box_probability_file, 'w')
    # compute the distance through the path along the planes
    box_lines = ComputeDistancesAlongCV(plane_points)
    mid_point_dist = []
    # compute the midpoints between box lines
    for a, b in utils.get_pairwise_list(box_lines):
        mid_point_dist.append((b - a) * 0.5 + a)
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

    # plot box free energies with the two different error bars
    if plane_points is not None:
        x = mid_point_dist[BoxLowerID:BoxUpperID + 1]
    else:
        x = range(len(boxFreeEnergy))
        box_lines = []
    vis.plot2D(x, boxFreeEnergy, output_dir + "/boxFreeEnergy.png",
               "Distance along CV / Angstrom", "RT",
               color=Tableau.tableau20[0],
               error=box_energy_std,
               ecolor=Tableau.tableau20[4], capthick=2, s=2)
    vis.plot2D(x, boxFreeEnergy, output_dir +
               "/boxFreeEnergyCumError.png", "Distance along CV / Angstrom", "RT",
               color=Tableau.tableau20[0], error=
               box_energy_std_cuml, ecolor=Tableau.tableau20[0], capthick=2, s=2)
    vis.plot2D(x, np.cumsum(boxProbability), output_dir +
               "/boxProbability.png", "Distance along CV / Angstrom ", "$P_n$",
               ylimits=[-0.1, 1.1],
               color=Tableau.tableau20[0],
               s=10)
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
                      current_box_id, lower, line, threshold, last_bound_hit):
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
    valid = True
    if np.allclose(boundary_hit, boundary_test):
        last_bound_hit = boundary_hit
        if (found_hit):
            passageTime = hit_time - last_hit_time
            # Negative passage time will occur if files concatenated
            if (passageTime <= 0):
                found_hit = False
            # if hit against against different boundary to previous
            # and less than threshold, get rid of it.
            elif passageTime < threshold:
                if not np.allclose(boundary_hit, last_bound_hit):
                    valid = False
                    print("Found short passage time at step", hit_time)
            else:
                FPT_list.append(passageTime)
                last_hit_time = hit_time
                numhits = numhits + 1
        else:
            last_hit_time = hit_time
            found_hit = True
    return FPT_list, found_hit, last_hit_time, numhits, valid


def GetFPTsAndHist(trajectory_files,
                   prev_analysis_dir,
                   BoundaryList,
                   plane_points,
                   BoxLowerID,
                   BoxUpperID,
                   bin_planes,
                   bin_centers,
                   passage_threshold,
                   Ndim,
                   nBoxes):
    """
    Given directories to existing FPTs from previous and/or path to trajectory
    file, will read the FPTs and populate a histogram
    WARNING: Assumes the histogram bins are the same between analysis runs
    """

    # set up empty returns
    nbins = len(bin_planes) - 1
    fpt_lower_list = [[] for x in range(nBoxes)]
    fpt_upper_list = [[] for x in range(nBoxes)]
    counts = [0.0] * (nbins)

    global box_counts_twod_x
    box_counts_twod_x = [[] for x in range(nBoxes)]
    global box_counts_twod_y
    box_counts_twod_y = [[] for x in range(nBoxes)]

    # loop over any precomputed histogram files (RawHistogram.txt)
    if prev_analysis_dir is not None:
        for directory in prev_analysis_dir:
            fpt_dir = directory + "/FPT_arrays"
            hist_file = directory + "/rawHistogram.txt"
            print("Filling histogram from precomputed ",
                  "histogram file ", hist_file)
            new_counts = ReadHistogram(hist_file)
            if len(new_counts) != len(counts):
                print("Error! The number of bins from this file does not",
                      "match the rest of the input. Skipping this file.")
            else:
                for i in range(len(counts)):
                    counts[i] += new_counts[i]
            print("Getting passage times for all boxes from precomputed ",
                  "FPT directory ", fpt_dir)
            lower_list, upper_list = ReadFPTs(fpt_dir, BoxLowerID,
                                              BoxUpperID, nBoxes)
            for i in range(nBoxes):
                fpt_lower_list[i] += lower_list[i]
                fpt_upper_list[i] += upper_list[i]

    # loop over any trajectory files and get the FPTs and fill histogram
    if trajectory_files is not None:
        for trajectory in trajectory_files:
            print("Filling histogram and computing FPTs from trajectory file",
                  trajectory)
            numlines = utils.get_number_lines(trajectory)
            lower, upper, new_counts \
                = GetFPTsAndHistFromTraj(trajectory, BoundaryList, BoxLowerID,
                                         BoxUpperID, passage_threshold,
                                         bin_planes, bin_centers, Ndim,
                                         numlines)
            for i in range(nBoxes):
                fpt_lower_list[i] += lower[i]
                fpt_upper_list[i] += upper[i]
            if len(new_counts) != len(counts):
                print("Error! The number of bins from this file does not",
                      "match the rest of the input. Skipping this file.")
            else:
                for i in range(len(counts)):
                    counts[i] += new_counts[i]

    heatmap_dir = output_dir + "/2D_Histograms"
    # output FPTs to save repeat analysis time
    for boxIdx in range(BoxLowerID, BoxUpperID + 1):
        fpt_dir = output_dir + "/FPT_arrays"
        utils.make_sure_path_exists(fpt_dir)

        lower_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx - 1)
        lowerFile = open(lower_fpt_name, 'w')
        for fpt in fpt_lower_list[boxIdx]:
            print("{0},{1}".format(fpt[0], fpt[1]), file=lowerFile)
        lowerFile.close()

        upper_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
        upperFile = open(upper_fpt_name, 'w')
        for fpt in fpt_upper_list[boxIdx]:
            print("{0},{1}".format(fpt[0], fpt[1]), file=upperFile)
        upperFile.close()

        # TODO generalise this.
        # produce heat maps for each box.
        utils.make_sure_path_exists(heatmap_dir)
        # can only do this if we've gathered data from a trajectory, it is not stored.
        if trajectory_files != None and len(trajectory_files) > 0:
            vis.plot_box_2d_hist(box_counts_twod_x, box_counts_twod_y, BoundaryList, plane_points, boxIdx, x_cv_label,
                                 y_cv_label,
                                 heatmap_dir + "/box_" + str(boxIdx) + ".png")
    if trajectory_files != None and len(trajectory_files) > 0:
        vis.plot_time_in_boxes(box_counts_twod_x, box_counts_twod_y, "time_in_box.png")
        print("plotting all 2d boxes")
        vis.plot_all_box_2d_hist(box_counts_twod_x, box_counts_twod_y, BoundaryList, plane_points,
                                 x_cv_label, y_cv_label, heatmap_dir + "/all_boxes.png")

    print("FPTs outputted to " + fpt_dir)

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

    return fpt_lower_list, fpt_upper_list, counts


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


def GetFPTsAndHistFromTraj(opfilename, bounds, LowerBoxID, UpperBoxID,
                           passage_threshold,
                           bin_planes, bin_centers, ndim, numlines):
    """
    Given opfilename, a BXD trajectory file, a list of BXD bounds and histogram
    bins, will calculate FPTs for each box and fill histogram.
    """

    # Variables for FPTs
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
    last_bound_hit = None
    hit_counter = 0

    #   initialize two lists
    UpperFPTs = [[] for x in range(nboxes)]
    LowerFPTs = [[] for x in range(nboxes)]
    hits = 0

    # Variables for histogram binning
    n_planes = len(bin_planes)
    nbins = n_planes - 1
    counts = [0.0] * (nbins)
    bin_pairs = []
    current_bin = None
    bin_num = 0
    # temporary counts which are discarded if passage time less than threshold
    tmp_counts = [0.0] * (nbins)
    last_time = 0
    max_steps = 20000  # max steps to allow before assuming fresh file
    for a, b in utils.get_pairwise_list(bin_planes):
        bin_pairs.append(tuple([a, b]))
    # set up progress bar
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=numlines,
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ',
                                               progressbar.Percentage()])
        bar.start()

    try:
        for linestring in opfile:
            line = line + 1

            # update progress bar
            if progress_bar:
                bar.update(line)
            else:
                if line % 1000 == 0:
                    print('.', end="", flush=True)
            linelist = linestring.split()
            if len(linelist) < ndim + 1:
                print("Odd line, skipping: ", line)
                continue
            # Read CV and time
            try:
                time = float(linelist[0])
                cv = np.array([float(x) for x in linelist[1:ndim + 1]])
            except ValueError:
                print("Error reading line, skipping ", line)
                continue

            # if time is significantly greater than previous then treat
            # as if separate file
            if current_box_id is not None and time - last_time > max_steps:
                FoundFirstUpperHit[current_box_id] = False
                FoundFirstLowerHit[current_box_id] = False

            last_time = time
            # perform histogram filling on lines that are not inversions.
            stride = 100
            if len(linelist) == ndim + 1 or hit_counter > stride:
                if hit_counter > stride:
                    hit_counter = 0

                find_bin = False
                in_box = False
                # If current bin has been set, then check if in it
                if current_bin is not None:
                    in_box = IsInsideBox(cv, current_bin[0], current_bin[1],
                                         ndim)
                # if in current box, add it to counts
                if in_box:
                    tmp_counts[bin_num] += 1

                # if not in box, then need to find the bin
                elif in_box is not None:
                    find_bin = True
                if find_bin:
                    found_bin = False
                    i = 0
                    for pair in bin_pairs:
                        if not found_bin:
                            if IsInsideBox(cv, pair[0], pair[1], ndim):
                                bin_num = i
                                tmp_counts[bin_num] += 1
                                found_bin = True
                                current_bin = pair
                                break
                            else:
                                i = i + 1
                        else:
                            break
                # this check is not totally correct, a small number will be dropped when crossing boundary.
                if IsInsideBox(cv, bounds[current_box_id], bounds[current_box_id + 1], ndim, False):
                    box_counts_twod_x[current_box_id].append(cv[0])
                    box_counts_twod_y[current_box_id].append(cv[1])

            # perform FPT calculations on lines that are inversions
            if len(linelist) > ndim + 1:
                hit_counter += 1
                new_box_id = -1
                debug = False
                find_box = False
                if current_box_id == -1:
                    find_box = True
                elif IsInsideBox(cv, bounds[current_box_id],
                                 bounds[current_box_id + 1], ndim, debug):
                    new_box_id = current_box_id
                else:
                    find_box = True
                if find_box:
                    # first try boxes adjacent to current one,
                    # then the rest of the list
                    box_list = []
                    if current_box_id != -1:
                        box_list += [max(current_box_id - 1, LowerBoxID),
                                     min(current_box_id + 1, UpperBoxID)]
                    box_list += range(LowerBoxID, UpperBoxID + 1)
                    for i in box_list:
                        in_box = IsInsideBox(
                            cv, bounds[i], bounds[i + 1], ndim, debug)
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
                    current_box_id = new_box_id
                    hits = 0
                if current_box_id >= 0:
                    StepsInsideBox = StepsInsideBox + 1
                    # Set up if first time enter box
                    if (not InsideTheBox):
                        InsideTheBox = True

                    hits += 1

                    # Make sure length of line is correct
                    if len(linelist) != 2 + 2 * ndim:
                        print("Odd line, skipping:", line)
                        continue
                    boundary_hit = np.array(
                        [float(x) for x in linelist[ndim + 1:]])

                    valid = True
                    if np.allclose(boundary_hit, bounds[current_box_id + 1]):
                        hit_time = time
                        FPT_list = UpperFPTs[current_box_id]
                        found_hit = FoundFirstUpperHit[current_box_id]
                        last_hit_time = LastUpperHitTime[current_box_id]
                        numhits = NumUpperHits[current_box_id]

                        if (found_hit):
                            passageTime = hit_time - last_hit_time
                            # Negative passage time will occur if files
                            # are concatenated
                            if (passageTime <= 0):
                                found_hit = False
                            # if hit against against different boundary to
                            # previous
                            # and less than threshold, get rid of it.
                            elif passageTime < passage_threshold:
                                # if not np.allclose(boundary_hit,
                                #                   last_bound_hit):
                                valid = False
                                print("Found short passage time at step",
                                      hit_time)
                            else:
                                FPT_list.append([hit_time, passageTime])
                                last_hit_time = hit_time
                                numhits = numhits + 1
                        else:
                            last_hit_time = hit_time
                            found_hit = True

                        UpperFPTs[current_box_id] = FPT_list
                        FoundFirstUpperHit[current_box_id] = found_hit
                        LastUpperHitTime[current_box_id] = last_hit_time
                        NumUpperHits[current_box_id] = numhits
                    elif np.allclose(boundary_hit, bounds[current_box_id]):
                        boundary_hit = boundary_hit
                        hit_time = time
                        FPT_list = LowerFPTs[current_box_id]
                        found_hit = FoundFirstLowerHit[current_box_id]
                        last_hit_time = LastLowerHitTime[current_box_id]
                        numhits = NumLowerHits[current_box_id]

                        if (found_hit):
                            passageTime = hit_time - last_hit_time
                            # Negative passage time will occur if files
                            # concatenated
                            if (passageTime <= 0):
                                found_hit = False
                            # if hit against against different boundary to
                            # previous
                            # and less than threshold, get rid of it.
                            elif passageTime < passage_threshold:
                                # if not np.allclose(boundary_hit,
                                #                   last_bound_hit):
                                valid = False
                                print("Found short passage time at step",
                                      hit_time)
                            else:
                                FPT_list.append([hit_time, passageTime])
                                last_hit_time = hit_time
                                numhits = numhits + 1
                        else:
                            last_hit_time = hit_time
                            found_hit = True

                        LowerFPTs[current_box_id] = FPT_list
                        FoundFirstLowerHit[current_box_id] = found_hit
                        LastLowerHitTime[current_box_id] = last_hit_time
                        NumLowerHits[current_box_id] = numhits
                    else:
                        print("Warning, hit a boundary,",
                              "but not the box I'm in: ", line)

                    last_bound_hit = boundary_hit
                    # if passage time was less than passage_threshold,
                    # then discard all counts in that time
                    # otherwise, add to counts
                    remove_short_fpts = True
                    if remove_short_fpts and not valid:
                        for i in range(len(tmp_counts)):
                            tmp_counts[i] = 0
                    else:
                        for i in range(len(tmp_counts)):
                            counts[i] += tmp_counts[i]
                            tmp_counts[i] = 0

    finally:
        opfile.close()
    return LowerFPTs, UpperFPTs, counts


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
            # read time and passage time
            LowerFPTs[boxIdx].append([float(x) for x in fpt.split(',')])
        lowerFile.close()

        upper_fpt_name = fpt_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
        upperFile = open(upper_fpt_name)
        for fpt in upperFile:
            UpperFPTs[boxIdx].append([float(x) for x in fpt.split(',')])
        upperFile.close()

    return LowerFPTs, UpperFPTs


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
    # plot the FPT histogram
    # HistogramFPTs(Upper, Lower, bin)
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

    # compute standard error = sample std/sqrt(N)
    err_upper = np.std(upper_FPTs, ddof=1) / math.sqrt(float(len(upper_FPTs)))
    err_lower = np.std(lower_FPTs, ddof=1) / math.sqrt(float(len(lower_FPTs)))
    # err_upper = np.std(upper_FPTs, ddof=1)
    # err_lower = np.std(lower_FPTs, ddof=1)
    var = math.pow(k_upper * err_upper, 2) + math.pow(k_lower * err_lower, 2)
    return var


def ComputeMFPTs(LowerFPTs, UpperFPTs, bounds, LowerBoxID, UpperBoxID):
    """
    Computes the MFPTs from the list of lower and upper FPTs
    for each box
    """
    #   calculate Mean First Passage Time
    #   MPFTS will only be averaged if they are greater than recrossTime
    nboxes = len(bounds) - 1
    klower = [0] * nboxes
    kupper = [0] * nboxes
    mfpts = [0] * nboxes
    box_plot_dir = output_dir + "/mfpt_box_plots"
    utils.make_sure_path_exists(box_plot_dir)
    print("Box and whisker plots of MFPT for each box outputted to ",
          box_plot_dir)
    for boxIdx in range(LowerBoxID, UpperBoxID + 1):

        lower_fpts = np.array(LowerFPTs[boxIdx])
        upper_fpts = np.array(UpperFPTs[boxIdx])

        boxplotname = box_plot_dir + "/" + str(boxIdx)
        vis.plotBoxPlot([lower_fpts[:, 1], upper_fpts[:, 1]],
                        [str(boxIdx) + " to " + str(boxIdx - 1),
                         str(boxIdx) + " to " + str(boxIdx + 1)],
                        "Box", "FPT", boxplotname)

        fpt_count = len(lower_fpts)
        decay_dir = output_dir + "/decays"
        utils.make_sure_path_exists(decay_dir)
        if (fpt_count != 0):
            MFPT = np.mean(lower_fpts[:, 1])
            mfpts.append(MFPT)
            klower[boxIdx] = 1 / MFPT

            lowerFileName = decay_dir + '/%sto%s.txt' % (boxIdx, boxIdx - 1)
            print("\tMFPT box %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx - 1, MFPT, lowerFileName))
            print("\tbox %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx - 1, klower[boxIdx], lowerFileName))

            # calculate Decay Profile
            LowerDecayArray = CalculateDecay(lower_fpts)
            lowerDecayPlotName = decay_dir + \
                                 '/%sto%s.png' % (boxIdx, boxIdx - 1)
            plotlabel = str(boxIdx) + " to " + str(boxIdx - 1)
            color = boxIdx

            # Plot the decay profile
            vis.plotDecay(LowerDecayArray, lowerDecayPlotName, plotlabel, color)

            keylist = list(LowerDecayArray.keys())
            keylist.sort()
            lowerFile = open(lowerFileName, 'w')
            lowerFile.write('0.0\t%s\n' % (fpt_count))
            # plotDecayArray()
            for key in keylist:
                lowerFile.write(
                    '%s\t%s\n' % (str(key), (LowerDecayArray[key])))
            lowerFile.close()
        else:
            print("\tbox %s to box %s = N/A " % (boxIdx, boxIdx - 1))
            klower[boxIdx] = 0.0

        # Upper FPTs
        fpt_count = len(upper_fpts)

        if (fpt_count != 0):
            MFPT = np.mean(upper_fpts[:, 1])
            kupper[boxIdx] = 1 / MFPT

            upperFileName = decay_dir + '/%sto%s.txt' % (boxIdx, boxIdx + 1)
            print("\tMFPT box %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx + 1, MFPT, upperFileName))
            print("\tbox %s to box %s = %s (See file %s)" %
                  (boxIdx, boxIdx + 1, kupper[boxIdx], upperFileName))

            # calculate Decay Profile

            UpperDecayArray = CalculateDecay(upper_fpts)
            upperDecayPlotName = decay_dir + \
                                 '/%sto%s.png' % (boxIdx, boxIdx + 1)
            plotlabel = str(boxIdx) + " to " + str(boxIdx + 1)
            color = boxIdx

            vis.plotDecay(UpperDecayArray, upperDecayPlotName, plotlabel, color)

            keylist = list(UpperDecayArray.keys())
            keylist.sort()
            upperFile = open(upperFileName, 'w')
            upperFile.write('0.0\t%s\n' % (fpt_count))
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
    # initialize a Map
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

    python BXDanalysis_2D.py bounds.json ndim --prev_analysis analysis
    """

    argparser = argparse.ArgumentParser()

    # Required arguments
    argparser.add_argument("bounds", help="File containing the BXD bounds")
    argparser.add_argument(
        "ndim", help="Number of dimensions of BXD boundaries", type=int)
    argparser.add_argument("--trajectory", nargs='+',
                           help="Path of BXD output file(s)")
    argparser.add_argument("--prev_analysis", nargs='+',
                           help=("Directory/Directories of previous analysis",
                                 "performed with this script"))
    argparser.add_argument(
        "--fpt_threshold",
        help=("FPT threshold, any FPTs less than the given value",
              "will be discarded. Default 0.0"), type=float)
    argparser.add_argument(
        "--lower_box_id", help="Lower BXD box number to be used in analysis",
        type=int)
    argparser.add_argument(
        "--upper_box_id", help="Upper BXD box number to be used in analysis",
        type=int)
    argparser.add_argument(
        "--histogram_bin_width",
        help="The maximum size for a histogram bin.",
        type=float)
    argparser.add_argument(
        "--histogram_json_file",
        help="A file containing planes to use as histogram bins",
        type=str)
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
    center_file = None
    max_width = float("inf")
    reverse_boxes = False
    traj_files = None
    prev_analysis = None
    histogram_json_file = None
    if args.trajectory:
        traj_files = args.trajectory
    if args.prev_analysis:
        prev_analysis = args.prev_analysis
    if not args.trajectory and not prev_analysis:
        print("Error: Trajectory file was not specified (--trajectory). ",
              "If rerunning analysis, need directory of previous output",
              "(--prev_analysis)")
        sys.exit()
    if args.lower_box_id:
        lower_box_id = args.lower_box_id
    if args.upper_box_id:
        upper_box_id = args.upper_box_id
    if args.fpt_threshold:
        mfpt = args.fpt_threshold
    if args.histogram_bin_width:
        max_width = args.histogram_bin_width
    if args.reverse_boxes:
        reverse_boxes = True
    if args.output_dir:
        output_dir = args.output_dir
    if args.histogram_json_file:
        histogram_json_file = args.histogram_json_file
    BXDanalysis(traj_files, bounds_file, ndim,
                MFPTthreshold=mfpt, BoxLowerID=lower_box_id,
                BoxUpperID=upper_box_id,
                MaxDistance=max_width,
                BinFile=histogram_json_file,
                ReverseBounds=reverse_boxes,
                PrevAnalysisDir=prev_analysis)
