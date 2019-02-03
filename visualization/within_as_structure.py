# hcm/visualization/viz/within_as_structure.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from util.file_utils import progress
from core.keys import act_to_actlabel, obs_period_to_days
from visualization.plot_util.plot_utils import add_figtitle, save_figure, set_subplots_labels
from visualization.plot_util.plot_colors_old import fcols


def figsubplots(num_items):
    figsize, nrows, ncols = None, None, None
    if num_items <= 9:
        figsize = (12, 8)
        nrows, ncols = 3, 3
    elif 9 < num_items <= 12:
        figsize = (8, 6)
        nrows, ncols = 4, 3
    elif 12 < num_items <= 16:
        figsize = (12, 10)
        nrows, ncols = 4, 4
    elif 16 < num_items <= 20:
        figsize = (12, 12)
        nrows, ncols = 4, 5
    return figsize, nrows, ncols


def set_layout(ax, xmax, num_as):
    ax.set_xlim((0, xmax))
    ax.set_ylim((0, num_as + 5))
    ax.set_xticks(np.arange(0, xmax + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)


def draw_structure(ax, data):
    ev_types = ['F', 'D', 'L', 'O'][:-1]
    colors = [fcols[act_to_actlabel[ev_type]][1] for ev_type in ev_types]

    num_as = len(data)
    k = 0
    for array in data:
        ypos = num_as - k
        col = 0
        for arr in array[:-1]:
            for x1, x2 in arr / 60.:
                pat = patches.Rectangle(xy=(x1, ypos), width=x2 - x1, height=1, lw=0, fc=colors[col])
                ax.add_patch(pat)
            col += 1
        progress(k, num_as)
        k += 1


def within_as_structure(experiment, obs_period, num_mins=15):
    res_subdir = os.path.join('within_as_structure')
    days = obs_period_to_days[experiment.name][obs_period]
    xmax = num_mins if num_mins is not None else 200

    for group in experiment.groups:
        figsize, nrows, ncols = figsubplots(group.tot_mice)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        m = 0
        for mouse in group.mice:
            if not mouse.is_ignored:
                print "drawing within_as_structure for: {}".format(mouse)
                # fig, ax = plt.subplots()
                ax = axes.flatten()[m]
                data = mouse.create_within_as_structure(days, num_mins)
                draw_structure(ax, data)
                set_layout(ax, xmax, num_as=len(data))
                m += 1

        row_labels = ['{}:\n{}'.format(k, v) for k, v in days_dict.items() if k != 'acclimated']
        col_labels = [g.name for g in experiment.groups]
        xlabel = "AS time from onset [min]"
        ylabel = "active state counts"
        set_subplots_labels(fig, ncols, row_labels, col_labels, xlabel, ylabel)
        title = "{}\nwithin active state structure\ngroup{}: {}\n{} days:\n{}"\
            .format(str(experiment), group.number, group.name, obs_period, days)
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        add_figtitle(fig, title)
        # save
        fname = '{}_within_as_structure_group{}_{}_{}_days_{}mins_test3' \
            .format(experiment.name, group.number, group.name, obs_period, num_mins)
        save_figure(experiment, fig, res_subdir, fname)