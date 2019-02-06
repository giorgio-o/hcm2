# hcm/visualization/viz/breakfast_days_comparison.py
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import os

from core.keys import act_to_actlabel, obs_period_to_days, to_seaborn_ci
from util.utils import days_to_obs_period
from visualization.plot_util.plot_utils import add_figtitle, save_figure, set_facetgrid_labels
from visualization.plot_util.plot_colors import get_features_palette


def create_legends(palettes, labels):
    from matplotlib.lines import Line2D
    legends = list()
    for n, palette in enumerate(palettes):
        elements = list()
        for color, label in zip(palette, labels):
            elements.append(Line2D([0], [0], color=color, lw=2, label=label)),
        legends.append(elements)
    return legends


def set_legends(g, palettes, hue):
    labels = g.data[hue].unique()
    title = None
    if hue == 'group':
        title = 'group'
    elif hue == 'obs_period':
        title = 'day'
    legends = create_legends(palettes, labels)
    ys = [0.68, 0.28]
    for y, elements in zip(ys, legends):
        ax = g.fig.add_axes([0.98, y, 0.05, 0.25])
        ax.legend(handles=elements, frameon=False, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes, title=title)
        ax.axis('off')


def set_layout(g, xvar, yvar, palettes, hue, lw=1.5):
    g.set_axis_labels(xvar, yvar)\
        .set(xticks=np.linspace(0, 30, 5 + 1)) \
        .set(xticklabels=range(0, 15 + 1, 3)) \
        .set_titles('')
    set_facetgrid_labels(g)  # outer labels

    # change lines and colors
    for cnt, axes in enumerate(g.axes):
        palette = palettes[cnt]
        num_items = len(g.data[hue].unique())
        for ax in axes:
            # lines
            lines = ax.get_lines()
            k = len(lines) // num_items
            for n, line in enumerate(lines):
                color = palette[n / k]
                line.set_color(color)
                line.set_lw(lw)


def facets_groups_comparison_hue_day(experiment, obs_period, htype, timepoint, err_type, ignore):

    days = obs_period_to_days[experiment.name][obs_period]
    ci = to_seaborn_ci[err_type]
    # load data
    fname = '{}_breakfast_{}_binary_counts_tbinsize30s.json'.format(experiment.name, timepoint)
    df_all = experiment.load_json_data(days, akind='breakfast', fname=fname, subakind=timepoint, sub_index1='as_num',
                                       sub_index2='event', PROB=True, ignore=ignore)
    # tidy form
    xvar = "AS time from {} [min]".format(timepoint)
    yvar = "probability"
    dft = pd.melt(df_all.reset_index(), id_vars=['group', 'mouse', 'day', 'event'], var_name=xvar, value_name=yvar)
    tbin_size = dft[xvar].min()
    dft[xvar] = dft[xvar] / 60.  # convert to minutes
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

    row = 'event'
    col = 'group'
    hue = 'obs_period'
    num_periods = len(dft.obs_period.unique())
    height = 2
    palettes = [get_features_palette(act_to_actlabel[x], num_periods) for x in 'FD']
    kws1 = dict(height=height, aspect=1.5, ci=ci, sharey='row', legend=False,
                markers=[''] * len(dft[xvar].unique()))

    # draw
    if htype == 'groups':
        g = sns.catplot(kind='point', data=dft, x=xvar, y=yvar, row=row, col=col, hue=hue, **kws1)
        # layout
        set_layout(g, xvar, yvar, palettes, hue=hue)
        set_legends(g, palettes, hue=hue)
        # title
        title = "{}\nprobability of event from AS {}\ngroup averages$\pm${}\ndays comparison:\n{}".format(
            str(experiment), timepoint, err_type, days)
        add_figtitle(g.fig, title, y=1.05, ypad=-0.05)
        plt.subplots_adjust(wspace=0.3)
        # save
        res_subdir = experiment.path_to_results(subdir=os.path.join('breakfast'))
        fname = "{}_breakfast_{}_tbinsize{}s_groups_{}_days_comparison_hue_day".format(
            experiment.name, timepoint, tbin_size, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)


def facets_groups_comparison_hue_group(experiment, obs_period, htype, timepoint, err_type, ignore):

    days = obs_period_to_days[experiment.name][obs_period]
    ci = to_seaborn_ci[err_type]
    # load data
    fname = '{}_breakfast_{}_binary_counts_tbinsize30s.json'.format(experiment.name, timepoint)
    df_all = experiment.load_json_data(days, akind='breakfast', fname=fname, subakind=timepoint, sub_index1='as_num',
                                       sub_index2='event', PROB=True, ignore=ignore)
    # tidy form
    xvar = "AS time from {} [min]".format(timepoint)
    yvar = "probability"
    dft = pd.melt(df_all.reset_index(), id_vars=['group', 'mouse', 'day', 'event'], var_name=xvar, value_name=yvar)
    tbin_size = dft[xvar].min()
    dft[xvar] = dft[xvar] / 60.  # convert to minutes
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

    row = 'event'
    col = 'obs_period'
    hue = 'group'
    num_groups = len(dft.group.unique())
    height = 2
    palettes = [get_features_palette(act_to_actlabel[x], num_groups) for x in 'FD']
    kws1 = dict(height=height, aspect=1.5, ci=ci, sharey='row', legend=False,
                markers=[''] * len(dft[xvar].unique()))

    # draw
    if htype == 'groups':
        g = sns.catplot(kind='point', x=xvar, y=yvar, row=row, col=col, hue=hue, data=dft, **kws1)
        # layout
        set_layout(g, xvar, yvar, palettes, hue)
        set_legends(g, palettes, hue)
        # title
        title = "{}\nprobability of event from AS {}\ngroup averages$\pm${}\ndays comparison:\n{}".format(
            str(experiment), timepoint, err_type, days)
        add_figtitle(g.fig, title, y=1.05, ypad=-0.05)
        plt.subplots_adjust(wspace=0.3)
        # save
        res_subdir = experiment.path_to_results(subdir=os.path.join('breakfast'))
        fname = "{}_breakfast_{}_tbinsize{}s_groups_{}_days_comparison_hue_group".format(
            experiment.name, timepoint, tbin_size, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)