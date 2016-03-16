# 2D Scatter Plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import argparse
import ast
import json
from matplotlib.mlab import griddata
from pudb import set_trace
from matplotlib.colors import LogNorm

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
    for i in range(len(plane_points)-1):
        c = (plane_points_np[i] + plane_points[i+1])*0.5
        c_list = [x for x in c]
        box_centroids.append(tuple(c_list))

    print(planes)
    return planes, box_centroids, plane_points


def prep_contour(opfilename, x_min, x_max, y_min, y_max, npoints):
    """
    Given filename with x,y and z values, produces grid over which 
    a contour plot may be produced
    """

    x = []
    y = []
    z = []

    #read x,y and z from file
    opfile = open(opfilename, 'r')
    try:
        for line in opfile: 
            fields = line.strip().split()
            x.append(float(fields[0]))
            y.append(float(fields[1]))
            z.append(float(fields[2]))
    finally:
        opfile.close()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # define grid.
    xi = np.linspace(x_min, x_max, npoints)
    yi = np.linspace(y_min, y_max, npoints)
    # grid the data.
    zi = griddata(x, y, z, xi, yi, interp='linear')

    return xi, yi, zi

bohr_to_angstrom = 0.529177211
BXDBounds = True

parser = argparse.ArgumentParser()

#Required arguments
parser.add_argument("--input", help="Path of input file containing 2D data", required=True)
parser.add_argument("--output", help="Output filename, .png or .svg recommended")
#Optional arguments
parser.add_argument("--xlabel", help="Label of xaxis", type=str)
parser.add_argument("--ylabel", help="Label of yaxis", type=str)
parser.add_argument("--title", help="Title of chart", type=str)
parser.add_argument("--stride", help="Stride, plot every n elements", type=str)
parser.add_argument("--range", help="Plot range", nargs=2, type=int)
parser.add_argument("--color", help="Tableau color (1-20)", type=int)
parser.add_argument("--ylines", nargs='+', help="Any extra lines to be drawn on y axis, give y value", type=float)
parser.add_argument("--xcol", help="X Column number (starting from 0)", type=int)
parser.add_argument("--ycol", help="Y Column number (starting from 0)", type=int)
parser.add_argument("--animate",help="Show animated plot", dest='animate_plot',action='store_true')
parser.set_defaults(animate_plot=False)
parser.add_argument("--save_animation", help="Save animated plot", type=str)
parser.add_argument("--bounds_json", help="Json boundary file", type=str)
parser.add_argument("--surface", help="File with PES to plot (x, y, z points)", type=str)
args = parser.parse_args()

filename = args.input
print(filename)
output_name = args.output

xlabel = ""
ylabel = ""
xcol = 1
ycol = 2
title = ""
stride = 1
plot_subset = False
plot_range = [0, 0]
color_id = 0
ndim = 2
animate_plot=False
mov_output_name = ""
save_animation = False

if args.xlabel:
    xlabel = args.xlabel
if args.ylabel:
    ylabel = args.ylabel
if args.title:
    title = args.title
if args.stride:
    stride = int(args.stride)
if args.range:
    plot_subset = True
    plot_range = args.range
    print("Plotting range: " + str(plot_range))
if args.color:
    color_id = args.color
if args.ylines:
    ylines = args.ylines
if args.xcol:
    xcol = args.xcol
if args.ycol:
    ycol = args.ycol
if args.animate_plot:
    animate_plot=True
    if args.save_animation:
        mov_output_name = args.save_animation
        save_animation = True
surface_file = None
if args.surface:
    surface_file = args.surface
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

with open(filename) as f:
    print("Opened file " + filename)

    for line in f:
        data_point = line.strip().split()
        try:
            time = int(line.strip().split()[0])
            times.append(time)
            #do not include inversion points: 
            if len(data_point) == ndim + 1:
                f_x.append(float(data_point[xcol]))
                f_y.append(float(data_point[ycol]))
            if len(data_point) > ndim + 1 and not args.bounds_json:
                bound = tuple([float(x) for x in data_point[3:6]])
                if bound not in bounds_set:
                    bounds_set.add(bound)
                    bounds_dict[bound] = Bound([(int(data_point[0]))], bound)
        except ValueError as e:
            print("Bad line: ", line, e)
    f.close()

start_step = min(times)
end_step = max(times)
start_step_index = times.index(min(times))
end_step_index = times.index(max(times))
if plot_subset:
    start_step = plot_range[0]
    end_step = plot_range[1]
    start_step_index = times.index(plot_range[0])
    end_step_index = times.index(plot_range[1])
print("start step: ", start_step, "last step: ", end_step)
data_x = f_x
data_y = f_y

# You typically want your plot to be ~1.33x wider than tall.
# Common sizes: (10, 7.5) and (12, 9)
dpi = 100
fig = plt.figure(figsize=(12, 9), dpi=dpi)

ax = fig.add_subplot(111)


# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Along the same vein, make sure your axis labels are large
# enough to be easily read as well. Make them slightly larger
# than your axis tick labels so they stand out.
ax.set_xlabel(xlabel, fontsize=24)
ax.set_ylabel(ylabel, fontsize=24)
ax.tick_params(axis='x', labelsize='22')
ax.tick_params(axis='y', labelsize='22')

y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(y_formatter)

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
#ax.set_ylim(0, 90)

#ax.set_xlim(0.5,3.0)
#ax.set_ylim(0.5,3.0)
print("Plotting")


x_min = min(data_x)
x_min = x_min - 0.05*x_min
x_max = max(data_x)
x_max = x_max + 0.01*x_max
y_min = min(data_y)
y_min = y_min - 0.01*y_min
y_max = max(data_y)
y_max = y_max + 0.01*y_max

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

i = color_id
lines = []
#plot_x = data_x[start_step_index:end_step_index:stride]
#plot_y = data_y[start_step_index:end_step_index:stride]
plot_x = data_x
plot_y = data_y
#print(plot_x)
#print(plot_y)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

if surface_file is not None: 
    con_xi, con_yi, con_zi = prep_contour(surface_file, x_min, x_max, 
                                          y_min, y_max, 20)
    cmap = "Greys"
    CS = ax.contourf(con_xi, con_yi, con_zi, 20, cmap=cmap, extend="both",
                     vmax=abs(con_zi).max(), vmin=-abs(con_zi).max())
    CS = plt.contour(con_xi, con_yi, con_zi, 10, linewidths=0.5, colors='k')

plt.hist2d(plot_x, plot_y, bins=100, norm=LogNorm())
plt.colorbar()


bound_length = 0.5

for b in bounds_set:
    bound_steps = bounds_dict[b].time
    x = []
    y = []
    cx = []
    cy = []
    px = []
    py = []
    vx = []
    vy = []
    centroid = bounds_dict[b].centroid
    point = bounds_dict[b].point
    skip = False

    if bound_steps[0] <= end_step:
        print("plotting bound ", b, bound_steps[0], end_step)
        if len(bound_steps) > 1:
            if end_step > bound_steps[1]:
                skip = True
        if b[1] == 0.0 and b[0] == 0.0:
            skip = True
        if not skip:
            if point is not None: 
                if b[1] == 0.0: 
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
                    lower = 0.0
                    x = np.array(np.linspace(lower, upper, num=100))   
                    y = [(-b[0] * v - b[2]) / b[1] for v in x]
            else:         
                if b[1] == 0.0:
                    y = np.array(np.linspace(y_min, y_max))
                    x = [-(b[2] / b[0])] * len(y)
                else:
                    x = np.array(np.linspace(x_min, x_max))
                    y = [(-b[0] * v - b[2]) / b[1] for v in x]
            if centroid is not None and point is not None:
                assert len(centroid) == 2, "Centroid not 2D!"
                assert len(point) == 2, "Point is not 2D!"
                px = [point[0]]
                py = [point[1]]
                cx = [centroid[0]]
                cy = [centroid[1]] 
                vx = [-0.2*b[0]]
                vy = [-0.2*b[1]]
    bline1, = ax.plot(x, y, "-", lw=6.0, color="white", alpha=1.0)
    bline, = ax.plot(x, y, "-", lw=4.0, color="black", alpha=1.0)

# Finally, save the figure as a PNG and display it.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
#print "Saving to " + output_name
if(args.output):
    plt.savefig(args.output, bbox_inches="tight");

if dpi <= 200:
    print("Displaying plot")
    plt.show()





