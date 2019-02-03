# hcm/visualization/viz/features_vs_ct_as_groups_avg_days_comparison.py
import matplotlib.pyplot as plt
import os

from core.keys import obs_period_to_days
from util.utils import days_to_obs_period
from visualization.plot_util.plot_utils import add_figtitle, save_figure
from visualization.plot_util.plot_colors import get_features_palette
from features_vs_ct_as_groups_avg import load_tidy_data, draw_facets, set_as_layout, set_as_legend


def facets_groups_comparison_hue_day(experiment, obs_period, bin_type, err_type, ignore):
        res_subdir = os.path.join('features', 'groups', 'days_comparison_hue_day')
        days = obs_period_to_days[experiment.name][obs_period]
        bins_label = '24H, DC and LC' if bin_type == '3cycles' else bin_type
        height = 2

        # load AS data
        feature_type = 'AS'
        dft, _ = load_tidy_data(experiment, obs_period, bin_type=bin_type, feature_type=feature_type, ignore=ignore)
        dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

        row = 'feature'
        col = 'group'
        hue = 'obs_period'
        num_periods = len(dft.obs_period.unique())
        palette = get_features_palette(ftype='active_state', num_items=num_periods)

        # draw
        g = draw_facets(dft, bin_type, err_type, palette, height, row=row, col=col, hue=hue, num_items=16,
                        add_kws=dict(sharey='row'))
        # layout
        set_as_layout(g, bin_type, lw=1)
        set_as_legend(g, palette, hue)
        # title
        title = "{}\nactive state features, {}\ngroup averages $\pm$ {}\ndays comparison:\n{}" \
            .format(str(experiment), bins_label, err_type, days)
        add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
        plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
        # save
        fname = "{}_active_state_features_vs_CT_{}_grp_avg_{}_days_comparison" \
            .format(experiment.name, bin_type, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)

        # load FDL data
        feature_type = 'FDL'
        dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type=feature_type, ignore=ignore)
        # label dataframe in groups-days
        dft[hue] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

        for ftype, dfg in dft.groupby('feature_type'):
            palette = get_features_palette(ftype, num_items=num_periods)

            # draw
            g = draw_facets(dfg, bin_type, err_type, palette, height, row=row, col=col, hue=hue, num_items=16,
                            add_kws=dict(sharey='row'))
            # layout
            set_as_layout(g, bin_type, lw=1)
            set_as_legend(g, palette, hue)

            title = "{}\n{} features, {}\ngroup averages $\pm$ {}\ndays comparison:\n{}" \
                .format(str(experiment), ftype, bins_label, err_type, days)
            add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
            plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
            # save
            fname = "{}_{}_features_vs_CT_{}_grp_avg_{}_days_comparison_hue_day" \
                .format(experiment.name, ftype, bin_type, err_type)
            save_figure(experiment, g.fig, res_subdir, fname)


def facets_groups_comparison_hue_group(experiment, obs_period, bin_type, err_type, ignore):

    res_subdir = os.path.join('features', 'groups', 'days_comparison_hue_group')
    days = obs_period_to_days[experiment.name][obs_period]
    bins_label = '24H, DC and LC' if bin_type == '3cycles' else bin_type
    height = 2

    # load AS data
    feature_type = 'AS'
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type=feature_type, ignore=ignore)
    dft['obs_period'] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)

    row = 'feature'
    col = 'obs_period'
    hue = 'group'
    num_groups = len(dft.group.unique())
    palette = get_features_palette(ftype='active_state', num_items=num_groups)

    # draw
    g = draw_facets(dft, bin_type, err_type, palette, height, row=row, col=col, hue=hue,
                    num_items=16, add_kws=dict(sharey='row'))
    # layout
    set_as_layout(g, bin_type, lw=1)
    set_as_legend(g, palette, hue)
    # title
    title = "{}\nactive state features, {}\ngroup averages $\pm$ {}\ndays comparison:\n{}" \
        .format(str(experiment), bin_type, err_type, days)
    add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
    plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
    # save
    fname = "{}_active_state_features_vs_CT_{}_grp_avg_{}_days_comparison" \
        .format(experiment.name, bin_type, err_type)
    save_figure(experiment, g.fig, res_subdir, fname)

    # load FDL data
    feature_type = 'FDL'
    dft, _ = load_tidy_data(experiment, obs_period, bin_type, feature_type=feature_type, ignore=ignore)
    # label dataframe in groups-days
    dft[col] = dft.apply(days_to_obs_period, args=(experiment,), axis=1)
    all_days = dft.obs_period.unique()

    for ftype, dfg in dft.groupby('feature_type'):
        palette = get_features_palette(ftype, num_items=len(all_days))

        # draw
        g = draw_facets(dfg, bin_type, err_type, palette, height, row=row, col=col, hue=hue, num_items=16,
                        add_kws=dict(sharey='row'))
        # layout
        set_as_layout(g, bin_type, lw=1)
        set_as_legend(g, palette, hue)

        title = "{}\n{} features, {}\ngroup averages $\pm$ {}\ndays comparison:\n{}" \
            .format(str(experiment), ftype, bins_label, err_type, days)
        add_figtitle(g.fig, title, y=1.1, xpad=-0.05, ypad=-0.02)
        plt.subplots_adjust(wspace=0.2)  # necessary, or custom legend will screw the layout up
        # save
        fname = "{}_{}_features_vs_CT_{}_grp_avg_{}_days_comparison_hue_group" \
            .format(experiment.name, ftype, bin_type, err_type)
        save_figure(experiment, g.fig, res_subdir, fname)