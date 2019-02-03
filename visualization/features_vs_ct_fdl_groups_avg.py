# hcm/visualization/viz/features_vs_ct_fdl_groups_avg.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

from core.keys import to_seaborn_ci, act_to_actlabel, obs_period_to_days
from visualization.plot_util.plot_utils import get_ct_bins_xticks_labels, set_facetgrid_labels, add_figtitle
from visualization.plot_util.plot_utils import save_figure
from visualization.plot_util.plot_colors import get_features_palette
from features_vs_ct_as_groups_avg import load_tidy_data, show_background_dark_cycle_box


def create_fdl_legends(palettes, labels):
    from matplotlib.lines import Line2D
    legends = list()
    for n, palette in enumerate(palettes):
        elements = list()
        for color, label in zip(palette, labels):
            elements.append(Line2D([0], [0], color=color, lw=2, label=label)),
        legends.append(elements)
    return legends


def set_fdl_legends(g, palettes, hue):
    labels, title = None, None
    if hue == 'group':
        labels, title = g.data.group.unique(), 'group'
    elif hue == 'day':
        labels, title = g.data.day.unique(), 'day'
    legends = create_fdl_legends(palettes, labels)
    ys = [0.7, 0.4, 0.1]
    for y, elements in zip(ys, legends):
        ax = g.fig.add_axes([0.98, y, 0.05, 0.25])
        ax.legend(handles=elements, frameon=False, bbox_to_anchor=(1.5, 1), bbox_transform=ax.transAxes, title=title)
        ax.axis('off')


def set_fdl_layout(g, bin_type, palettes, labelsize=8, lw=1.5):
    # xticks and labels
    _, xlabels, xlabel = get_ct_bins_xticks_labels(bin_type)
    g.set_axis_labels('', '')\
        .set_xticklabels(xlabels, fontsize=labelsize)\
        .set_titles('')\
        .set_xlabels(xlabel)
    set_facetgrid_labels(g, ROWS=False)  # outer labels

    # change lines and colors
    for cnt, axes in enumerate(g.axes):
        palette = palettes[cnt]
        num_groups = len(g.data.group.unique())
        for ax in axes:
            if bin_type in ['3cycles', '4bins']:
                num_cycles = len(g.data.timebin.unique())
                for n, patch in enumerate(ax.patches):
                    color = palette[n / num_cycles]
                    patch.set_color(color)

            elif bin_type in ['12bins', '24bins']:
                g.set(xticks=g.axes.flat[0].get_xticks() - 0.03)  # move xtick to bin edge. do not know why -0.025 here
                # lines
                lines = ax.get_lines()
                k = len(lines) // num_groups
                for n, line in enumerate(lines):
                    color = palette[n / k]
                    line.set_color(color)
                    line.set_lw(lw)
                show_background_dark_cycle_box(g)


def draw_facets(dft, bin_type, err_type, height, row, col, hue, num_items, add_kws=None):
    ci = to_seaborn_ci[err_type]
    kws1 = dict(height=height, aspect=1.5, ci=ci, sharey=False, legend=False)
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


def facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore):
    """ plots features vs CT (bins) and 24H/DC/LC (3cycles) """
    # load data
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type='FDL', ignore=ignore)

    res_subdir = os.path.join('features', 'groups')
    days = obs_period_to_days[experiment.name][obs_period]
    days_label = obs_period.replace('-', '_').replace(' ', '_')
    bins_label = '24H, DC and LC' if bin_type == '3cycles' else bin_type

    row = 'feature_type'
    col = 'feature'
    hue = 'group'
    height = 2
    groups = dft.group.unique()
    palettes = [get_features_palette(act_to_actlabel[x], len(groups)) for x in 'FDL']
    # draw
    g = draw_facets(dft, bin_type, err_type, height, row=row, col=col, hue=hue, num_items=len(groups))
    # layout
    set_fdl_layout(g, bin_type, palettes)
    set_fdl_legends(g, palettes, hue='group')
    title = "{}\nfeeding, drinking and locomotion features, {}\ngroup averages $\pm$ {}\n{} days:\n{}"\
        .format(str(experiment), bins_label, err_type, days_label, days)
    add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.03)
    plt.subplots_adjust(wspace=0.3)  # necessary, or custom legend will screw the layout up
    # save
    fname = "{}_feeding_drinking_locomotion_features_vs_CT_{}_grp_avg_{}_{}_days"\
        .format(experiment.name, bin_type, err_type, days_label)
    save_figure(experiment, g.fig, res_subdir, fname)
