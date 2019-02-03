# hcm/visualization/viz/features_vs_ct_as_groups_avg.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from core.keys import to_seaborn_ci, act_to_actlabel, obs_period_to_days
from visualization.plot_util.plot_utils import get_ct_bins_xticks_labels, set_facetgrid_labels, add_figtitle
from visualization.plot_util.plot_utils import save_figure
from visualization.plot_util.plot_colors import get_features_palette


def create_as_legend(palette, labels):
    from matplotlib.lines import Line2D
    legend = list()
    for color, label in zip(palette, labels):
        legend.append(Line2D([0], [0], color=color, lw=2, label=label)),
    return legend


def set_as_legend(g, palette, hue):
    labels, title = None, None
    if hue == 'group':
        labels, title = g.data.group.unique(), 'group'
    elif hue == 'day':
        labels, title = g.data.day.unique(), 'day'
    elif hue == 'obs_period':
        labels, title = g.data.obs_period.unique(), 'days'

    legend = create_as_legend(palette, labels)
    ax = g.fig.add_axes([0.98, 0.6, 0.05, 0.25])
    ax.legend(handles=legend, frameon=False, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes, title=title)
    ax.axis('off')


def show_background_dark_cycle_box(g):
    for ax in g.axes.flat:
        ax.axvspan(xmin=2.5, xmax=8.5, color='0.9', zorder=0)


def set_linewidth(g, lw):
    for ax in g.axes.flat:
        lines = ax.get_lines()
        for line in lines:
            line.set_lw(lw)


def set_as_layout(g, bin_type, labelsize=8, lw=1.5):
    _, xlabels, xlabel = get_ct_bins_xticks_labels(bin_type)
    g.set_axis_labels('', '')\
        .set_xticklabels(xlabels, fontsize=labelsize)\
        .set_titles('')\
        .set_xlabels(xlabel)
    set_facetgrid_labels(g, ROWS=False)  # outer labels
    if bin_type in ['12bins', '4bins']:
        set_linewidth(g, lw)
        g.set(xticks=g.axes.flat[0].get_xticks() - 0.5)  # move xtick to bin center
        show_background_dark_cycle_box(g)


def draw_facets(dft, bin_type, err_type, palette, height, row, col, hue, num_items, add_kws=None):
    ci = to_seaborn_ci[err_type]
    kws1 = dict(height=height, aspect=1.5, ci=ci, sharey=False, palette=palette, legend=False)
    kws2 = dict()
    kind = None
    if bin_type in ['3cycles', '4bins']:
        kind = 'bar'
    elif bin_type in ['12bins', '24bins']:
        kind = 'point'
        kws2 = dict(markers=[''] * num_items)
    kws1.update(kws2)
    add_kws = dict() if add_kws is None else add_kws
    kws1.update(add_kws)
    g = sns.catplot(kind=kind, data=dft, x='timebin', y='value', row=row, col=col, hue=hue, **kws1)
    return g


def load_tidy_data(experiment, obs_period, bin_type, feature_type=None, ignore=False):
    from core.keys import feature_keys_short
    days = obs_period_to_days[experiment.name][obs_period]
    fname = "{}_features_{}.json".format(experiment.name, bin_type)
    df = experiment.load_json_data(days, akind='features', fname=fname, subakind=bin_type, sub_index1='timebin',
                                   ignore=ignore)
    # change dataframe layout
    ordered_levels = ['group', 'mouse', 'day', 'timebin', 'feature_type']
    if feature_type is not None:
        df2 = None
        if feature_type == 'AS':
            df2 = df[feature_keys_short['AS']]
            df2.columns = ['probability', 'rate', 'duration']
            df2 = pd.concat([df2], keys=['active state'], names=['feature_type'])
        elif feature_type == 'FDL':
            new_cols = ['total amount', 'AS intensity', 'AS bout rate', 'bout size', 'bout duration', 'bout intensity']
            new_levels = ['feature_type', 'group', 'mouse', 'day', 'timebin']
            dfs = list()
            for activity in 'FDL':
                df1 = df[feature_keys_short[activity]]
                df1.columns = new_cols
                dfs.append(df1)
            df2 = pd.concat(dfs, keys=[act_to_actlabel[x] for x in 'FDL'], names=new_levels)
        df2 = df2.reorder_levels(ordered_levels)
        # tidy form
        id_vars = ['group', 'mouse', 'day', 'timebin', 'feature_type']
        dft = pd.melt(df2.reset_index(), id_vars=id_vars, var_name='feature', value_name='value')
    else:
        dft = None
    return dft, df


def facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore):
    """ plots features vs CT (bins) and 24H/DC/LC (3cycles) """
    # load data
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type='AS', ignore=ignore)

    res_subdir = os.path.join('features', 'groups')
    days = obs_period_to_days[experiment.name][obs_period]
    days_label = obs_period.replace('-', '_').replace(' ', '_')
    bins_label = '24H, DC and LC' if bin_type == '3cycles' else bin_type

    row = 'feature_type'
    col = 'feature'
    hue = 'group'
    height = 2
    groups = dft.group.unique()
    palette = get_features_palette(ftype='active state', num_items=len(groups))
    # draw
    g = draw_facets(dft, bin_type, err_type, palette, height, row=row, col=col, hue=hue, num_items=len(groups))
    # layout
    set_as_layout(g, bin_type)
    set_as_legend(g, palette, hue='group')
    title = "{}\nactive state features, {}\ngroup averages $\pm$ {}\n{} days:\n{}"\
        .format(str(experiment), bins_label, err_type, days_label, days)
    add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.03)
    plt.subplots_adjust(wspace=0.3)  # necessary, or custom legend will screw the layout up
    # save
    fname = "{}_active_state_features_vs_CT_{}_grp_avg_{}_{}_days"\
        .format(experiment.name, bin_type, err_type, days_label)
    save_figure(experiment, g.fig, res_subdir, fname)
