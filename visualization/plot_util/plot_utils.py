# hcm/visualization/plot_util/plot_colors.py
""" plot utility file """
import numpy as np
import matplotlib.pyplot as plt
import os


def get_ct_bin_centers(bin_type):
    bin_centers = None
    if bin_type == '12bins':
        bin_centers = np.arange(7, 33, 2).astype(int)
    return bin_centers


def get_ct_bins_xticks_labels(bin_type):
    xticks, xticklabels, xlabel = None, None, None
    if bin_type == '3cycles':
        xticks = [1, 2, 3]
        xticklabels = ['24H', 'DC', 'LC']
        xlabel = "cycle"
    elif bin_type == '12bins':
        xticks = get_ct_bin_centers(bin_type) - 1
        xticklabels = [str(x) if x < 25 else str(x - 24) for x in np.arange(6, 30, 2).astype(int)]
        xlabel = "circadian time"
    elif bin_type == '4bins':
        xticks = range(1, 4)
        xticklabels = [str(x) if x < 25 else str(x - 24) for x in np.arange(6, 30, 6).astype(int)]
        xlabel = "circadian time"

    return xticks, xticklabels, xlabel


def plot_vertical_lines_at_ct_12_24(ax, xticks=(12, 24)):
    for x in xticks:
        ax.axvline(x=x, color='.3', linestyle='--', lw=.5, zorder=0)


def add_data_source(fig, xpad=-0.1, ypad=-0.05, captionsize=6, source="HCM data, Tecott Lab UCSF",
                    author="GOnnis, Tecott Lab", affiliation="UCSF"):
    from datetime import datetime
    today = datetime.today()
    fig.text(.125 + xpad, ypad, "source: {}\nby: {} {}\n{}/{}".format(
        source, author, affiliation, today.month, today.year),
             fontsize=captionsize, ha='left', va='center')


def save_figure(experiment, fig, subdir, filename, dpi=300, PS=False):
    format_ = 'ps' if PS else 'pdf'
    filename = os.path.join(experiment.path_to_results(subdir), filename)
    filename = "{}.{}".format(filename, format_)
    print "saving.."
    fig.savefig(filename, bbox_inches='tight', dpi=dpi)
    print "figure saved to:\n{}".format(filename)
    plt.close()


def add_figtitle(fig, figtitle, y=1, fontsize=10, xpad=0, ypad=0.05, captionsize=6):
    fig.suptitle(figtitle, y=y, fontsize=fontsize, va='bottom')
    add_data_source(fig, xpad, ypad, captionsize)
    return fig


def set_subplots_labels(fig, ncols, row_labels, col_labels, xlabel="", ylabel="", fontsize_r=12, fontsize_c=12, rxpad=0,
                        cypad=0, xlpad=0, ylpad=0, ROWS=True):
    for cnt, ax in enumerate(fig.axes):
        row_num = cnt // ncols
        col_num = cnt % ncols
        if col_num == 0 and ROWS:  # rows
            ax.text(-0.5 + rxpad, 0.5, row_labels[row_num], fontsize=fontsize_r, ha='right', va='center',
                    transform=ax.transAxes)
        if row_num == 0:  # columns
            ax.text(0.5, 1.2 + cypad, col_labels[col_num], fontsize=fontsize_c, ha='center', va='bottom',
                    transform=ax.transAxes)
    fig.text(0.5, 0 + xlpad, xlabel, ha='center')
    fig.text(0.1 + ylpad, 0.5, ylabel, va='center', rotation=90)


def set_facetgrid_labels(g, fontsize_r=12, fontsize_c=12, rxpad=0, cypad=0, ROWS=True):
    ncols = len(g.col_names)
    for cnt, ax in enumerate(g.axes.flat):
        row_num = cnt // ncols
        col_num = cnt % ncols
        if col_num == 0 and ROWS:  # rows
            ax.text(-0.5 + rxpad, 0.5, g.row_names[row_num], fontsize=fontsize_r, ha='right', va='center',
                    transform=ax.transAxes)
        if row_num == 0:  # columns
            ax.text(0.5, 1.2 + cypad, g.col_names[col_num], fontsize=fontsize_c, ha='center', va='bottom',
                    transform=ax.transAxes)