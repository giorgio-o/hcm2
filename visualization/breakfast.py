# hcm/visualization/viz/breakfast.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from core.keys import act_to_actlabel, obs_period_to_days, to_seaborn_ci
from visualization.plot_util.plot_utils import add_figtitle, save_figure, set_facetgrid_labels
from visualization.plot_util.plot_colors import fcolors


def create_legend(palette, labels):
    from matplotlib.lines import Line2D
    legend = list()
    for color, label in zip(palette, labels):
        legend.append(Line2D([0], [0], color=color, lw=2, label=label)),
    return legend


def set_legend(g, palette, hue):
    labels = g.data[hue].unique()
    legend = create_legend(palette, labels)
    title = None
    if hue == 'event':
        title = hue
    elif hue == 'obs_period':
        title = 'days'

    ax = g.fig.add_axes([0.98, 0.6, 0.05, 0.25])
    ax.legend(handles=legend, frameon=False, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes, title=title)
    ax.axis('off')


def set_layout(g, xvar, yvar):
    g.set_axis_labels(xvar, yvar)\
        .set(xticks=np.linspace(0, 30, 5 + 1)) \
        .set(xticklabels=range(0, 15 + 1, 3)) \
        .set_titles('')
    set_facetgrid_labels(g, ROWS=False)  # outer labels


def breakfast(experiment, obs_period, htype='groups', timepoint='onset', err_type='sem', ignore=True):
    days = obs_period_to_days[experiment.name][obs_period]
    res_subdir = experiment.path_to_results(subdir=os.path.join('breakfast'))
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

    col = 'group'
    hue = 'event'
    height = 2
    palette = sns.color_palette([fcolors[act_to_actlabel[x]][1] for x in 'FD'])
    ci = to_seaborn_ci[err_type]
    col_wrap = 4 if experiment.name == 'StrainSurvey' else None
    kws1 = dict(height=height, aspect=1.5, ci=ci, palette=palette, sharey='row', legend=False, col_wrap=col_wrap,
                markers=[''] * len(dft[xvar].unique()))
    if htype == 'groups':
        # draw
        g = sns.catplot(kind='point', x=xvar, y=yvar, col=col, hue=hue, data=dft, **kws1)

        set_layout(g, xvar, yvar)
        set_legend(g, palette, hue='event')
        title = "{}\nprobability of event from AS {}\ngroup averages$\pm${}\ndays: {}: {}" \
            .format(str(experiment), timepoint, err_type, obs_period, days)
        add_figtitle(g.fig, title, y=1.05, ypad=-0.03)
        plt.subplots_adjust(wspace=0.3)
        # save
        fname = "{}_breakfast_{}_tbinsize{}s_groups_{}_days" \
            .format(experiment.name, timepoint, tbin_size, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)
