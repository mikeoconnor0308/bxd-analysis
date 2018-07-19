"""
Plots BXD bounds used for histogram binning produced by BXD analysis script
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import argparse
import ast
import json


class Bound(object):
    def __init__(self, time, plane, point=None, centroid=None):
        self.time = time
        self.plane = plane
        self.point = point
        self.centroid = centroid


def read_bounds_json(json_file, ndim):
    """
    Reads the bounds json file outputted by adaptive bxd.
    Will then compute centroids of boxes from the plane points

    Args:
        json_file (str) : path and file name of json bounds file
        ndim (int) : number of dimensions of CVs
    """

    assert isinstance(json_file, str), "Json_file is not a string"
    assert isinstance(ndim, int), "Number of dimensions is not an integer"

    json_data = open(json_file).read()

    data = json.loads(json_data)

    planes = []
    plane_points = []
    box_centroids = []
    hist_bools = []
    bounds = data["bounds"]
    print(bounds)
    plane_points_np = []
    for b in bounds:
        p = b["plane"]
        plane = tuple([float(x) for x in p])
        point_str = b["point"]
        point = tuple([float(x) for x in point_str])
        planes.append(plane)
        plane_points.append(point)
        plane_points_np.append(np.array([float(x) for x in point_str]))
        if "hist" in b: 
            hist_bools.append(bool(b["hist"]))
        else: 
            hist_bools.append(False)
    for i in range(len(plane_points)-1):
        c = (plane_points_np[i] + plane_points[i+1])*0.5
        c_list = [x for x in c]
        box_centroids.append(tuple(c_list))

    print(planes)
    return planes, box_centroids, plane_points, hist_bools

bohr_to_angstrom = 0.529177211
BXDBounds = True

parser = argparse.ArgumentParser()

#Required arguments
parser.add_argument("--input", help="JSON boundary file", required=True)
parser.add_argument("--output", help="Output filename, .png or .svg recommended")
#Optional arguments
parser.add_argument("--xlabel", help="Label of xaxis", type=str)
parser.add_argument("--ylabel", help="Label of yaxis", type=str)
parser.add_argument("--title", help="Title of chart", type=str)
parser.add_argument("--color", help="Tableau color (1-20)", type=int)
args = parser.parse_args()

filename = args.input
print(filename)
output_name = args.output

xlabel = ""
ylabel = ""
xcol = 1
ycol = 2
title = ""
color_id = 0
ndim = 2

if args.xlabel:
    xlabel = args.xlabel
if args.ylabel:
    ylabel = args.ylabel
if args.title:
    title = args.title
if args.color:
    color_id = args.color

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

data_x = []
data_y = []
bounds_dict = {}
bounds_set = set()
times = []

f_x = []
f_y = []

planes, centroids, plane_points, hist_bools = read_bounds_json(filename, 2)

# You typically want your plot to be ~1.33x wider than tall.
# Common sizes: (10, 7.5) and (12, 9)
fig = plt.figure(figsize=(12, 9), dpi=100)

# Remove the plot frame lines. They are unnecessary chartjunk.
ax = fig.add_subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Along the same vein, make sure your axis labels are large
# enough to be easily read as well. Make them slightly larger
# than your axis tick labels so they stand out.
ax.set_xlabel(xlabel, fontsize=20)
ax.set_ylabel(ylabel, fontsize=20)

y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(y_formatter)
ax.tick_params(axis='x', labelsize='20')
ax.tick_params(axis='y', labelsize='20')
# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
#ax.set_ylim(0, 90)

#ax.set_xlim(0.5,3.0)
#ax.set_ylim(0.5,3.0)
print("Plotting")

i = color_id
lines = []
print(centroids)
print(plane_points)
print(centroids + plane_points)
x_min = min([x[0] for x in ((centroids + plane_points))])

y_min = min([y[1] for y in ((centroids + plane_points))])
x_max = max([x[0] for x in ((centroids + plane_points))])
y_max = max([y[1] for y in ((centroids + plane_points))])
x_min = x_min - 0.05*abs(x_max - x_min)
x_max = x_max + 0.05*abs(x_max - x_min)
y_min = y_min - 0.05*abs(y_max - y_min)
y_max = y_max + 0.05*abs(y_max - y_min)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

bound_length = 0.4
box_id = 0
for b, point, hist in zip(planes, plane_points, hist_bools): 
    print("plotting bound", b, point)
    if abs(b[1]) < 0.01:
        print("Line parallel to y")
        lower = max(y_min, point[1] - bound_length*0.5)
        upper = min(y_max, point[1] + bound_length*0.5)
        y = np.array(np.linspace(lower, upper))
        x = [-(b[2] / b[0])] * len(y)
    else:
        norm_perp = np.array([-b[1], b[0]])
        point_np = np.array(point)
        x1 = (bound_length * 0.5 * norm_perp + point_np)[0]
        x2 = (point_np - bound_length * 0.5 * norm_perp)[0]
        lower = min(x1,x2)
        upper = max(x1,x2)   
        #lower = 0.0
        x = np.array(np.linspace(lower, upper))
        y = [(-b[0] * v - b[2]) / b[1] for v in x]
    if hist:
        bline, = ax.plot(x, y, ls=":", lw=1.2, color="black", alpha=0.8)
    else:
        bline, = ax.plot(x, y, "-", lw=1.2, color="black", alpha=1.0)
        box_id +=1
        print("switched to box: ", box_id)

print(centroids)
c_x = [x[0] for x in centroids]
c_y = [y[1] for y in centroids]
p_x = [x[0] for x in plane_points]
p_y = [y[1] for y in plane_points]
centroids_line = ax.plot(c_x, c_y, "--", marker="o", lw=1.0, ms=5, color=tableau20[14])
points_line = ax.plot(p_x, p_y, "--", marker="o", lw=1.0, ms=5, color=tableau20[5])

ax.set_title(title, y=1.06, fontsize=22)
if(args.output):
    plt.savefig(args.output, bbox_inches="tight");

plt.show()





