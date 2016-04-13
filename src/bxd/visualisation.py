import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import bxd.utils as utils
from bxd.tableau import Tableau
from matplotlib.colors import LogNorm


def plot2D(x, y, outputfile, xlabel, ylabel, xlimits=None, show_plot=False,
           ylimits=None, ylines=None, linecolor=Tableau.tableau20[0], **kwargs):
    """
    Useful little plotter. Any arguments to ax.scatter can be added to kwargs
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
    if error_bar:
        ax.errorbar(x, y, yerr=err, **error_kwargs)
    ax.plot(x, y, lw=1.5, alpha=1.0, color=linecolor)
    # maintain x and y limits, as scatter seems to break them.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.scatter(x, y, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if ylines is not None:
        y = ax.get_ylim()
        for x_val in ylines:
            x = [x_val] * len(y)
            ax.plot(x, y, color="black", ls='--')
    if 'label' in kwargs:
        ax.legend()

    plt.savefig(outputfile, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plotDecay(decay_array, outputfile, label, color_id):
    keylist = list(decay_array.keys())
    keylist.sort()
    values = []
    for k in keylist:
        if decay_array[k] > 0:
            values.append(math.log(decay_array[k]))
        else:
            values.append(0)

    plot2D(keylist, values, outputfile, "FPT", "ln R(t)", label=label,
           linecolor=Tableau.tableau20[color_id], s=8)


def plotBoxPlot(boxes, labels, xlabel, ylabel, output, **kwargs):
    fig, axarr = plt.subplots(2)

    for ax, box, label in zip(axarr, boxes, labels):
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.set_xlabel(xlabel, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='x', labelsize='22')
        ax.tick_params(axis='y', labelsize='22')

        # y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
        # ax.yaxis.set_major_formatter(y_formatter)
        ax.set_yscale('log')
        ax.boxplot(box, labels=[label], showmeans=True, **kwargs)

    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plotHistogramBins(planes, bin_centers, plane_points, hist_bools, output):
    """
    Plots the planes used for histogram binning, along with the line
    through the centers of the bins used to define the reaction coordinate
    for the free energy plot
    """

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='x', labelsize='20')
    ax.tick_params(axis='y', labelsize='20')

    x_min = min([x[0] for x in ((bin_centers + plane_points))])
    y_min = min([y[1] for y in ((bin_centers + plane_points))])
    x_max = max([x[0] for x in ((bin_centers + plane_points))])
    y_max = max([y[1] for y in ((bin_centers + plane_points))])
    x_min = x_min - 0.05 * abs(x_max - x_min)
    x_max = x_max + 0.05 * abs(x_max - x_min)
    y_min = y_min - 0.05 * abs(y_max - y_min)
    y_max = y_max + 0.05 * abs(y_max - y_min)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    bound_length = 0.4
    box_id = 0
    for b, point, hist in zip(planes, plane_points, hist_bools):
        if abs(b[1]) < 0.01:
            lower = max(y_min, point[1] - bound_length * 0.5)
            upper = min(y_max, point[1] + bound_length * 0.5)
            y = np.array(np.linspace(lower, upper))
            x = [-(b[2] / b[0])] * len(y)
        else:
            norm_perp = np.array([-b[1], b[0]])
            point_np = np.array(point)
            x1 = (bound_length * 0.5 * norm_perp + point_np)[0]
            x2 = (point_np - bound_length * 0.5 * norm_perp)[0]
            lower = min(x1, x2)
            upper = max(x1, x2)
            # lower = 0.0
            x = np.array(np.linspace(lower, upper))
            y = [(-b[0] * v - b[2]) / b[1] for v in x]
        if hist:
            bline, = ax.plot(x, y, ls=":", lw=1.2, color="black", alpha=0.8)
        else:
            bline, = ax.plot(x, y, "-", lw=1.2, color="black", alpha=1.0)
            box_id += 1

    c_x = [x[0] for x in bin_centers]
    c_y = [y[1] for y in bin_centers]
    ax.plot(c_x, c_y, "--", marker="o", lw=1.0, ms=5, color=Tableau.tableau20[14])
    points_x = [x[0] for x in plane_points]
    points_y = [y[1] for y in plane_points]
    ax.plot(points_x, points_y, marker="o", lw=1.0, ms=10, color=Tableau.tableau20[14])

    ax.set_title("Histogram Bins", y=1.06, fontsize=22)
    plt.savefig(output, bbox_inches="tight")
    plt.close()

def plot_all_box_2d_hist(cv_x, cv_y, bounds, plane_points, xlabel, ylabel, output):
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='x', labelsize='20')
    ax.tick_params(axis='y', labelsize='20')

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    ax.tick_params(axis='x', labelsize='22')
    ax.tick_params(axis='y', labelsize='22')

    plt.hold(True)
    x = []
    y = []
    for i in range(len(bounds)-1):
        lower_bound = bounds[i]
        upper_bound = bounds[i+1]
        x += cv_x[i]
        y += cv_y[i]

    ax.hist2d(x, y, bins=200, norm=LogNorm())
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    bound_length = 0.5
    for (b, point) in zip(bounds, plane_points):
        if b[1] == 0.0:
            lower = point[1] - bound_length*0.5
            upper = point[1] + bound_length*0.5
            y = np.array(np.linspace(lower, upper))
            x = [-(b[2] / b[0])] * len(y)
        else:
            #calculate line based on vector perpendicular to norm b[1] is x direction of plane.
            lower = point[0] -0.5*bound_length*b[1]
            upper = point[0] +0.5*bound_length*b[1]
            print("bound: ", b, " point: ", point)
            print("lower: ", lower, " upper", upper)
            x = np.array(np.linspace(lower, upper))
            y = [(-b[0] * v - b[2]) / b[1] for v in x]
        ax.plot(x, y, "-", lw=6.0, color="black", alpha=1.0)

    plt.savefig(output, bbox_inches="tight");
    plt.close()


def Box2DHistogram(cv_x, cv_y, lower_bound, upper_bound, box_id, xlabel, ylabel, output):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis='x', labelsize='20')
    ax.tick_params(axis='y', labelsize='20')

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    ax.tick_params(axis='x', labelsize='22')
    ax.tick_params(axis='y', labelsize='22')


    x_min = min(lower_bound[0], upper_bound[0])
    x_min *=0.95
    x_max = max(lower_bound[0], upper_bound[0])
    x_max *=1.05
    y_min = min(lower_bound[1], upper_bound[1])
    y_min *=0.95
    y_max = max(lower_bound[1], upper_bound[1])
    y_max *=1.05
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    plt.hist2d(cv_x, cv_y, bins=100, norm=LogNorm())
    plt.colorbar()

    for b in [lower_bound, upper_bound]:
        if b[1] == 0.0:
            y = np.array(np.linspace(y_min, y_max))
            x = [-(b[2] / b[0])] * len(y)
        else:
            x = np.array(np.linspace(x_min, x_max))
            y = [(-b[0] * v - b[2]) / b[1] for v in x]
        ax.plot(x, y, "-", lw=6.0, color="black", alpha=1.0)

    plt.savefig(output, bbox_inches="tight");
    plt.close()


def HistogramFPTs(Upper, Lower, bin, output_dir):
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
    utils.make_sure_path_exists(FPT_hist_path)
    plt.savefig(
            FPT_hist_path + "/" + str(bin).zfill(2) + ".png", bbox_inches="tight")
    plt.close()
