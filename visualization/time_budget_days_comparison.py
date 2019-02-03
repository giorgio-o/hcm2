# hcm/visualization/viz/time_budget_days_comparison.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    xs = np.arange(0.15, 1.16, 0.2)
    for x, elements in zip(xs, legends):
        ax = g.fig.add_axes([x, 0.75, 0.05, 0.25])
        ax.legend(handles=elements, frameon=False, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes, title=title)
        ax.axis('off')


def set_layout(g, xvar, yvar, palettes, hue, fontsize):
    g.set_axis_labels(xvar, yvar) \
        .set_titles('')
    set_facetgrid_labels(g)  # outer labels

    # change colors, add annotations
    for axes in g.axes:
        num_items = len(g.data[hue].unique())
        for cnt, ax in enumerate(axes):
            palette = palettes[cnt]
            # rectangles
            patches = ax.patches
            k = len(patches) // num_items
            for n, patch in enumerate(patches):
                color = palette[n / k]
                patch.set_facecolor(color)
                # display bar values
                w, h = patch.get_width(), patch.get_y()
                x, y = w + ax.get_xlim()[1] / 15, h + patch.get_height() / 2
                ax.text(x, h, str(round(w, 2)), fontsize=fontsize, color='dimgrey', ha='left', va='top')


def budget_type(row):
    return 'active state' if row['timebin'].startswith('AS') else '24 hours'


def time_budgets(experiment, obs_period, htype, bin_type, err_type, ignore):
    days = obs_period_to_days[experiment.name][obs_period]
    res_subdir = experiment.path_to_results(subdir=os.path.join('time_budget'))
    # load data
    fname = "{}_time_budget_{}.json".format(experiment.name, bin_type)
    df_all = experiment.load_json_data(days, akind='time_budget', fname=fname, subakind=bin_type, sub_index1='timebin',
                                       ignore=ignore)

    # percents
    df = df_all.apply(lambda row: 100. * row / row['total time'], axis=1).drop(columns=['active state', 'total time'])

    # tidy form
    xvar = "percent"
    yvar = "timebin"
    dft = pd.melt(df.reset_index(), id_vars=['group', 'mouse', 'day', 'timebin'], var_name='event', value_name=xvar)
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)
    # dft['budget type'] = dft.apply(budget_type, axis=1)
    hue = 'group'
    col = 'event'
    num_groups = len(dft.group.unique())

    height = 5 if experiment.name in ['StrainSurvey'] else 4
    fontsize = 4 if experiment.name in ['StrainSurvey'] else 6
    ci = to_seaborn_ci[err_type]
    act_types = ['F', 'D', 'L', 'O', 'IS']
    palettes = [get_features_palette(act_to_actlabel[x], num_groups) for x in act_types]
    kws1 = dict(height=height, aspect=1, ci=ci, sharex='col', legend=False)

    if htype == 'groups':
        g = sns.catplot(kind='bar', x=xvar, y=yvar, row='obs_period', col=col, hue=hue, data=dft, **kws1)

        set_layout(g, xvar, yvar, palettes, hue, fontsize)
        set_legends(g, palettes, hue)
        title = "{}\ntime budgets\ngroup averages$\pm${}\ndays: {}: {}" \
            .format(str(experiment), err_type, obs_period, days)
        add_figtitle(g.fig, title, y=1.15, xpad=-0.1, ypad=-0.05)
        plt.subplots_adjust(wspace=0.3)
        # save
        fname = "{}_time_budget_{}_groups_{}_days_test8" \
            .format(experiment.name, bin_type, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)


def facets_groups_days_comparison_hue_group(experiment, obs_period, htype, bin_type, err_type, ignore):

    days = obs_period_to_days[experiment.name][obs_period]
    res_subdir = experiment.path_to_results(subdir=os.path.join('time_budget'))
    # load data
    fname = "{}_time_budget_{}.json".format(experiment.name, bin_type)
    df_all = experiment.load_json_data(days, akind='time_budget', fname=fname, subakind=bin_type, sub_index1='timebin',
                                       ignore=ignore)

    # percents
    df = df_all.apply(lambda r: 100. * r / r['total time'], axis=1).drop(columns=['active state', 'total time'])

    # tidy form
    xvar = "percent"
    yvar = "timebin"
    dft = pd.melt(df.reset_index(), id_vars=['group', 'mouse', 'day', 'timebin'], var_name='event', value_name=xvar)
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

    row = 'obs_period'
    col = 'event'
    hue = 'group'
    num_groups = len(dft.group.unique())
    height = 3
    fontsize = 6
    ci = to_seaborn_ci[err_type]
    act_types = ['F', 'D', 'L', 'O', 'IS']
    palettes = [get_features_palette(act_to_actlabel[x], num_groups) for x in act_types]
    kws1 = dict(height=height, aspect=1, ci=ci, sharex='col', legend=False)

    if htype == 'groups':
        g = sns.catplot(kind='bar', x=xvar, y=yvar, row=row, col=col, hue=hue, data=dft, **kws1)

        set_layout(g, xvar, yvar, palettes, hue, fontsize)
        set_legends(g, palettes, hue)
        title = "{}\ntime budgets\ngroup averages$\pm${}\ndays: {}: {}" \
            .format(str(experiment), err_type, obs_period, days)
        add_figtitle(g.fig, title, y=1.1, xpad=-0.1, ypad=-0.05)
        plt.subplots_adjust(wspace=0.3)
        # save
        fname = "{}_time_budget_{}_groups_{}_days_days_comparison_hue_group_test1" \
            .format(experiment.name, bin_type, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)


def facets_groups_days_comparison_hue_day(experiment, obs_period, htype, bin_type, err_type, ignore):

    days = obs_period_to_days[experiment.name][obs_period]
    res_subdir = experiment.path_to_results(subdir=os.path.join('time_budget'))
    # load data
    fname = "{}_time_budget_{}.json".format(experiment.name, bin_type)
    df_all = experiment.load_json_data(days, akind='time_budget', fname=fname, subakind=bin_type, sub_index1='timebin',
                                       ignore=ignore)

    # percents
    df = df_all.apply(lambda r: 100. * r / r['total time'], axis=1).drop(columns=['active state', 'total time'])

    # tidy form
    xvar = "percent"
    yvar = "timebin"
    dft = pd.melt(df.reset_index(), id_vars=['group', 'mouse', 'day', 'timebin'], var_name='event', value_name=xvar)
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

    row = 'group'
    col = 'event'
    hue = 'obs_period'
    num_periods = len(dft.obs_period.unique())
    height = 4
    fontsize = 6
    ci = to_seaborn_ci[err_type]
    act_types = ['F', 'D', 'L', 'O', 'IS']
    palettes = [get_features_palette(act_to_actlabel[x], num_periods) for x in act_types]
    kws1 = dict(height=height, aspect=1, ci=ci, sharex='col', legend=False)

    if htype == 'groups':
        g = sns.catplot(kind='bar', x=xvar, y=yvar, row=row, col=col, hue=hue, data=dft, **kws1)

        set_layout(g, xvar, yvar, palettes, hue, fontsize)
        set_legends(g, palettes, hue)
        title = "{}\ntime budgets\ngroup averages$\pm${}\ndays: {}: {}" \
            .format(str(experiment), err_type, obs_period, days)
        add_figtitle(g.fig, title, y=1.1, xpad=-0.1, ypad=-0.05)
        plt.subplots_adjust(wspace=0.3)
        # save
        fname = "{}_time_budget_{}_groups_{}_days_days_comparison_hue_day_test1" \
            .format(experiment.name, bin_type, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)