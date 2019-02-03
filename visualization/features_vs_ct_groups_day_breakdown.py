# hcm/visualization/viz/features_vs_ct_groups_day_breakdown.py
import matplotlib.pyplot as plt
import os

from core.keys import obs_period_to_days
from visualization.plot_util.plot_utils import add_figtitle, save_figure
from visualization.plot_util.plot_colors import get_features_palette
from features_vs_ct_as_groups_avg import load_tidy_data, draw_facets, set_as_layout, set_as_legend


def facets_groups_day_breakdown(experiment, obs_period, bin_type, err_type, ignore):
    res_subdir = os.path.join('features', 'groups', 'days_breakdown')
    days = obs_period_to_days[experiment.name][obs_period]
    days_label = obs_period.replace('-', '_').replace(' ', '_')
    bins_label = '24H, DC and LC' if bin_type == '3cycles' else bin_type
    height = 2

    # load AS data
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type='AS', ignore=ignore)
    groups = dft.group.unique()
    palette = get_features_palette(ftype='active state', num_items=len(groups))
    # draw
    g = draw_facets(dft, bin_type, err_type, palette, height, row='feature', col='day', hue='group',
                    num_items=len(groups), add_kws=dict(sharey='row'))
    # layout
    set_as_layout(g, bin_type)
    set_as_legend(g, palette, hue='group')
    title = "{}\nactive state features, {}\ngroup averages $\pm$ {}\n{} days breakdown:\n{}"\.format(str(experiment), bins_label, err_type, days_label, days)
    add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
    plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
    # save
    fname = "{}_active_state_features_vs_CT_{}_grp_avg_{}_{}_days_breakdown"\
        .format(experiment.name, bin_type, err_type, days_label)
    save_figure(experiment, g.fig, res_subdir, fname)

    # load FDL data
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type='FDL', ignore=ignore)
    row = 'feature'
    col = 'day'
    hue = 'group'
    groups = dft.group.unique()
    for ftype, dfg in dft.groupby('feature_type'):
        palette = get_features_palette(ftype, num_items=len(groups))
        # draw
        g = draw_facets(dfg, bin_type, err_type, palette, height, row=row, col=col, hue=hue, num_items=len(groups),
                        add_kws=dict(sharey='row'))
        # layout
        set_as_layout(g, bin_type)
        set_as_legend(g, palette, hue='group')
        title = "{}\n{} features, {}\ngroup averages $\pm$ {}\n{} days breakdown:\n{}" \
            .format(str(experiment), ftype, bins_label, err_type, days_label, days)
        add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
        plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
        # save
        fname = "{}_{}_features_vs_CT_{}_grp_avg_{}_{}_days_breakdown" \
            .format(experiment.name, ftype, bin_type, err_type, days_label)
        save_figure(experiment, g.fig, res_subdir, fname)
