# 2D Scatter Plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import argparse
import ast
import json
from pudb import set_trace

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
parser.add_argument("--centroids_file", help="File containing points on each boundary plane", type=str)
parser.add_argument("--bounds_json", help="Json boundary file", type=str)
args = parser.parse_args()

filename = args.input
print(filename)
output_name = args.output

xlabel = ""
ylabel = ""
xcol = 1
ycol = 2
title = ""
every_N = 1
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
    every_N = int(args.stride)
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
centroids_file = None
if args.centroids_file:
    centroids_file = args.centroids_file

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

if args.bounds_json:
    b_, c_, p_ = read_bounds_json(args.bounds_json, 2)
    for b, c, p in zip(b_, c_, p_):
        if b not in bounds_set:
            bounds_set.add(b)
            bounds_dict[b] = Bound([0], b, p, c)

charmm_bound = False 
charmm_line = 0
charmm_plane = []
charmm_plane_point = []
with open(filename) as f:
    print("Opened file " + filename)

    for line in f:

        data_point = line.strip().split()
        if data_point[0].startswith('NEWBOUND'):
            charmm_bound = True
            charmm_plane = tuple([float(x) for x in data_point[2:5]])
        elif data_point[0].startswith('NEWPOINT'):
            charmm_plane_point = tuple([float(x) for x in data_point[2:4]])
        elif data_point[0].startswith("CENTROID"):
            charmm_centroid = tuple([float(x) for x in data_point[2:4]])
            print('adding bound', charmm_plane)
            if charmm_plane not in bounds_set:
                bounds_set.add(tuple(charmm_plane))
                #add start time
                time = [int(data_point[1])]
                bounds_dict[charmm_plane] = Bound(time, charmm_plane)
                bounds_dict[charmm_plane] = Bound(time, charmm_plane, charmm_plane_point, charmm_centroid)
        elif data_point[0].startswith("NewBound"):
            bound = tuple(ast.literal_eval(data_point[1]))
            point = tuple(ast.literal_eval(data_point[2]))
            centroid = tuple(ast.literal_eval(data_point[3]))
            print('adding bound', bound)
            if bound not in bounds_set:
                bounds_set.add(tuple(bound))
                #add start time
                bounds_dict[bound] = Bound([int(times[-1])], bound, point, centroid)
        elif data_point[0].startswith('KILLBOUN'):
            bound = tuple([float(x) for x in data_point[2:5]])
            print('removing bound', bound)
            if bound in bounds_set:
                #add removal time
                bounds_dict[bound].time.append(int(data_point[1]))
        else:
            try:
                time = int(line.strip().split()[0])
                times.append(time)
                f_x.append(float(data_point[xcol]))
                f_y.append(float(data_point[ycol]))
                if len(data_point) > ndim + 1:
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
fig = plt.figure(figsize=(10, 10), dpi=1600)

ax = fig.add_subplot(111)


# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Along the same vein, make sure your axis labels are large
# enough to be easily read as well. Make them slightly larger
# than your axis tick labels so they stand out.
ax.set_xlabel(xlabel, fontsize=26)
ax.set_ylabel(ylabel, fontsize=26)
ax.tick_params(axis='x', labelsize='24')
ax.tick_params(axis='y', labelsize='24')

y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(y_formatter)

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
#ax.set_ylim(0, 90)

#ax.set_xlim(0.5,3.0)
#ax.set_ylim(0.5,3.0)
print("Plotting")

i = color_id
lines = []
plot_x = data_x[start_step_index:end_step_index:every_N]
plot_y = data_y[start_step_index:end_step_index:every_N]

x_min = min(plot_x)
x_min = x_min - 0.05*x_min
x_max = max(plot_x)
x_max = x_max + 0.01*x_max
y_min = min(plot_y)
y_min = y_min - 0.01*y_min
y_max = max(plot_y)
y_max = y_max + 0.05*y_max

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


line, = ax.plot(plot_x, plot_y, ls=' ', color=tableau20[i], marker='o', ms=10, alpha=0.2)

#code to plot last point in trajectory range
plot_last_point = False
if plot_last_point:
    last_point_plot, = ax.plot([plot_x[-1]], [plot_y[-1]],
                               color='k', marker='x', lw=1,
                               mew=4, ms=6)  # marker thicknesses
else:
    last_point_plot, = ax.plot([], [])
lines.append(line)
lines.append(last_point_plot)
print("Number of lines", len(lines))
bound_lines_index = len(lines)


bound_length = 0.5
#code to select a specific bound
#bounds_list_full = list(bounds_set)
#bounds_list = []
#bounds_list.append(bounds_list_full[9])
#bounds_list.append(bounds_list_full[1])
#b_index = 0

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

    if start_step <= bound_steps[0] or bound_steps[0] <= end_step:
        print("plotting bound ", b)
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
                    print(norm_perp)
                    point_np = np.array(point)
                    print(point_np)
                    x1 = (bound_length * 0.5 * norm_perp + point_np)[0]
                    print(x1)
                    x2 = (point_np - bound_length * 0.5 * norm_perp)[0]
                    print(x2)
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
                vx = [-0.08*b[0]]
                vy = [-0.08*b[1]]
                print(px, py, vx, vy)
    bline, = ax.plot(x, y, "-", lw=10.0, color="black", alpha=1.0)
    ax.quiver(px, py, vx, vy, angles='xy', scale_units='xy', scale=1, color=tableau20[8])
    vplot, = ax.plot(cx+px, cy+py, "--", lw=5.0, color="black")
    vplot, = ax.plot(cx+px, cy+py, "--", marker="o", ms=15, lw=3.0, color=tableau20[8], alpha=0.8)
    #lines.append(bline)
    lines.append(vplot)
    #lines.append(vplot)
# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.
ax.set_title(title, y=1.06, fontsize=22)
# Finally, save the figure as a PNG and display it.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
#print "Saving to " + output_name
if(args.output):
    plt.savefig(args.output, bbox_inches="tight");


# Animation section

animate_steps = 200


def animate(i):
    start = max(0, i - int(animate_steps))
    #start = 0
    print(i)
    plot_x = data_x[start:(i+1)]
    plot_y = data_y[start:(i+1)]
    line.set_xdata(plot_x)
    line.set_ydata(plot_y)
    last_point_plot.set_xdata([plot_x[-1]])
    last_point_plot.set_ydata([plot_y[-1]])
    j = bound_lines_index
    for b in bounds_set:
        cx = []
        cy = []
        px = []
        py = []
        cent_x = []
        cent_y = []
        x = []
        y = []
        centroid = bounds_dict[b].centroid
        point = bounds_dict[b].point
        skip = False
        bound_steps = bounds_dict[b].time
        if times[i] > bound_steps[0]:
            if len(bound_steps) > 1:
                if times[i] > bound_steps[1]:
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
                        print(x1)
                        x2 = (point_np - bound_length * 0.5 * norm_perp)[0]
                        print(x2)
                        lower = min(x1,x2)
                        upper = max(x1,x2)   
                        lower = 0.0
                        x = np.array(np.linspace(lower, upper))   
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
                    cent_x = cx + px
                    cent_y = cy + py

        lines[j].set_data(x, y)
        lines[j+1].set_data(cent_x, cent_y)
        j += 2
    return lines,


if animate_plot:
    s = start_step_index
    e = end_step_index
    print("animating")
    ani = animation.FuncAnimation(fig, animate, list(range(s, e, every_N)), interval=60,blit=False)

    if save_animation:
        print("Saving movie")
        ani.save('animation.gif', writer='imagemagick', fps=10);
        #ani.save(mov_output_name, extra_args=['-vcodec', 'h264','-pix_fmt', 'yuv420p'])
    else:
        plt.show()
else:
    print("Displaying plot")
    plt.show()





