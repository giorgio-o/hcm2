# hcm/visualization/viz/position_density.py
"""Module for drawing position density plots with occupancy times. """
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import df_utils
from core.keys import homebase_1dto2d, obs_period_to_days, cycle_timebins
from visualization.plot_util.plot_utils import add_figtitle, save_figure, set_facetgrid_labels


def annotate_position_density(data, fontsize=8):
    ax = plt.gca()
    ybins, xbins = data.shape
    for y in xrange(ybins):
        for x in xrange(xbins):
            ax.text(x, y + 0.2, np.floor(data[y, x] * 1000) / 10, ha='center', va='center', color='w',
                    fontsize=fontsize)


def draw_homebase(data):
    rect, obs = data.detected.unique().tolist()[0], data.observed.unique().tolist()[0]
    rect_hb = [homebase_1dto2d[r] for r in rect]
    obs_hb = [homebase_1dto2d[r] for r in obs]
    ax = plt.gca()
    for cell in rect_hb:
        yc, xc = cell
        ax.plot(xc - 0.2, yc - 0.2, 'v', color='r', markersize=10, label='coded hb')
    if sorted(rect_hb) != sorted(obs_hb):  # np.array([(r==b) for r in rect_hb for b in obs_hb]).sum()
        for cell in obs_hb:
            yb, xb = cell
            ax.plot(xb + 0.2, yb - 0.2, 'v', color='y', markersize=10, label='obs hb')


def colorbar(g):
    axcbar = g.fig.add_axes([.95, 0.35, .02, 0.33])  # [left, bottom, width, height]
    im = g.axes[0, 0].get_images()[0]
    xbins, ybins = im.get_array().data.T.shape
    cbar = None
    if (xbins, ybins) == (2, 4):
        cbar = g.fig.colorbar(im, cax=axcbar, cmap='viridis')
    elif (xbins, ybins) == (12, 24):
        cbar = g.fig.colorbar(im, cax=axcbar, cmap='viridis_r', format=mpl.ticker.LogFormatterMathtext())
    cbar.ax.set_ylabel('Occupancy', rotation=270, labelpad=10)


def draw_imshow(*args, **kwargs):
    from matplotlib.colors import LogNorm
    data = kwargs.pop('data')
    xbins, ybins = args
    values = data['value'].values.reshape(ybins, xbins)
    vmin, vmax = (0.01, 1) if (xbins, ybins) == (2, 4) else (0.0001, 1)
    plt.set_cmap("viridis_r")
    if (xbins, ybins) == (2, 4):
        plt.imshow(values, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax))
        if kwargs['hb']:
            # rect, obs = data.detected.unique().tolist()[0], data.observed.unique().tolist()[0]
            # rect_hb = [homebase_1dto2d[r] for r in rect]
            # obs_hb = [homebase_1dto2d[r] for r in obs]
            # draw_homebase(plt.gca(), rect_hb, obs_hb)
            draw_homebase(data)
        if kwargs['annotate']:
            annotate_position_density(values)
    elif (xbins, ybins) == (12, 24):
        plt.imshow(values, interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), aspect='equal')


def load_data(experiment, days, bin_type, xbins, ybins, occupancy, hb, ignore):
    fname_in = "{}_position_{}_xbins{}_ybins{}_bin_times.json".format(experiment.name, bin_type, xbins, ybins)
    df = experiment.load_json_data(days, akind='position', fname=fname_in, subakind=bin_type, sub_index1='timebin',
                                   occupancy=occupancy, hb=hb, ignore=ignore)
    return df


def write_to_csv(experiment, obs_period, bin_type, xbins, ybins, ignore):
    res_subdir = os.path.join('position_density', 'csv_files')
    days = obs_period_to_days[experiment.name][obs_period]
    days_label = obs_period.replace('-', '_').replace(' ', '_')
    text2 = '' if ignore else 'ALL_'
    for occupancy in [True, False]:
        text = 'occupancy' if occupancy else 'bin_times'
        df = load_data(experiment, days, bin_type, xbins, ybins, occupancy, hb=True, ignore=ignore)

        for timebin, dfg in df.groupby('timebin'):
            # mousedays
            fname = "{}_position_{}_{}_xbins{}_ybins{}_mousedays_{}{}_days".format(
                experiment.name, text, timebin, xbins, ybins, text2, days_label)
            df_utils.save_dataframe_to_csv(experiment, dfg, os.path.join(res_subdir, "mousedays"), fname)
            # mice
            grouped = dfg.groupby(['group', 'mouse'])
            fname2 = "{}_position_{}_{}_xbins{}_ybins{}_mice_avg_{}_days".format(
                experiment.name, text, timebin, xbins, ybins, days_label)
            df_utils.save_dataframe_to_csv(experiment, grouped.mean(), os.path.join(res_subdir, "mice"), fname2)
            # groups
            grouped = dfg.groupby(['group'])
            fname3 = "{}_position_{}_{}_xbins{}_ybins{}_groups_avg_{}_days".format(
                experiment.name, text, timebin, xbins, ybins, days_label)
            df_utils.save_dataframe_to_csv(experiment, grouped.mean(), os.path.join(res_subdir, "groups"), fname3)

    # homebase
    df_hb = experiment.load_homebase_data(days)
    fname = '{}_homebase_location_mousedays'.format(experiment.name)
    df_utils.save_dataframe_to_csv(experiment, df_hb, res_subdir, filename=fname)


def position_density(experiment, obs_period, htype='group', bin_type='7cycles', xbins=2, ybins=4, ignore=False,
                     csv_file=False):
    """Draws position density plots. """
    days = obs_period_to_days[experiment.name][obs_period]
    timebins = cycle_timebins[bin_type]
    res_subdir = os.path.join('position_density', htype)
    var_name = 'xy_bin'
    value_name = 'value'
    kwargs = dict(xticks=[], yticks=[], title='', xlabel='', ylabel='')
    days_label = obs_period.replace('-', '_').replace(' ', '_')

    if csv_file:
        write_to_csv(experiment, obs_period, bin_type, xbins, ybins, ignore)

    else:
        # load data
        hb = True if htype == 'mousedays' else False
        df_all = load_data(experiment, days, bin_type, xbins=xbins, ybins=ybins, occupancy=True, hb=hb, ignore=ignore)

        if htype == 'groups':
            # figure1: group averages
            # tidy form
            id_vars = ['group', 'timebin']
            dfm = df_all.groupby(id_vars).agg(np.mean)
            dft = pd.melt(dfm.reset_index(), id_vars=id_vars, var_name=var_name, value_name=value_name)
            # figure
            g = sns.FacetGrid(dft, row='group', col='timebin', col_order=timebins, height=2,
                              gridspec_kws={'wspace': 0.05})
            g = g.map_dataframe(draw_imshow, xbins, ybins, hb=hb, annotate=True).set(**kwargs)
            # layout
            colorbar(g)
            set_facetgrid_labels(g)  # rows and cols labels
            title = "{}\nposition density {}x{}, {}\ngroup averages\n{} days:\n{}" \
                .format(str(experiment), xbins, ybins, bin_type, obs_period, days)
            add_figtitle(g.fig, title)
            # save
            fname = "{}_position_{}_xbins{}_ybins{}_groups_{}_days" \
                .format(experiment.name, bin_type, xbins, ybins, days_label)
            save_figure(experiment, g.fig, res_subdir, filename=fname)
            plt.close()

            # figure2: group averages, day by day
            res_subdir = os.path.join(res_subdir, 'days_breakdown')
            id_vars = ['group', 'day', 'timebin']
            dfm = df_all.groupby(id_vars).agg(np.mean)
            dft = pd.melt(dfm.reset_index(), id_vars=id_vars, var_name=var_name, value_name=value_name)
            for timebin, dfg in dft.groupby(['timebin']):
                # figure
                g = sns.FacetGrid(dfg, row='group', col='day', height=2, gridspec_kws={'wspace': 0.05})
                g = g.map_dataframe(draw_imshow, xbins, ybins, hb=hb, annotate=True).set(**kwargs)
                # layout
                colorbar(g)
                set_facetgrid_labels(g)  # rows and cols labels
                title = "{}\nposition density {}x{}, {}\ngroup averages, days\n{} days breakdown:\n{}".format(
                    str(experiment), xbins, ybins, timebin, obs_period, days)
                add_figtitle(g.fig, title)
                # save
                fname = "{}_position_{}_xbins{}_ybins{}_{}_grp_avg_{}_days_breakdown" \
                    .format(experiment.name, bin_type, xbins, ybins, timebin, days_label)
                save_figure(experiment, g.fig, res_subdir, fname)
                plt.close()

        elif htype == 'mice':
            id_vars = ['group', 'mouse', 'timebin']
            dfm = df_all.groupby(id_vars).agg(np.mean)
            dft = pd.melt(dfm.reset_index(), id_vars=id_vars, var_name=var_name, value_name=value_name)
            for group_name, dfg in dft.groupby(['group']):
                group_number = experiment.group_number(group_name)
                # figure
                g = sns.FacetGrid(dfg, row='timebin', col='mouse', height=2, gridspec_kws={'wspace': 0.05})
                g = g.map_dataframe(draw_imshow, xbins, ybins, hb=False, annotate=True).set(**kwargs)
                # layout
                colorbar(g)
                set_facetgrid_labels(g)  # rows and cols labels
                title = "{}\nposition density {}x{}, {}\ngroup{}: {}\n{} days:\n{}" \
                    .format(str(experiment), xbins, ybins, bin_type, group_number, group_name, obs_period, days)
                add_figtitle(g.fig, title, y=0.95)
                # save
                fname = "{}_position_{}_xbins{}_ybins{}_grp{}_{}_avg_{}_days" \
                    .format(experiment.name, bin_type, xbins, ybins, group_number, group_name, days_label)
                save_figure(experiment, g.fig, res_subdir, filename=fname)
                plt.close()

        elif htype == 'mousedays':
            res_subdir = os.path.join(res_subdir, 'xbins{}_ybins{}'.format(xbins, ybins))
            id_vars = ['group', 'mouse', 'day', 'timebin', 'detected', 'observed']
            dft = pd.melt(df_all.reset_index(), id_vars=id_vars, var_name=var_name, value_name=value_name)
            for name, dfg in dft.groupby(['group', 'mouse']):
                group_name, mouse_name = name
                group_number = experiment.group_number(group_name)
                # figure
                g = sns.FacetGrid(dfg, row='timebin', col='day', height=2, gridspec_kws={'wspace': 0.05})
                g = g.map_dataframe(draw_imshow, xbins, ybins, hb=hb, annotate=True).set(**kwargs)
                # layout
                colorbar(g)
                set_facetgrid_labels(g)  # rows and cols labels
                title = "{}\nposition density {}x{}, {}\ngroup{}: {}, {}\n{} days:\n{}".format(
                    str(experiment), xbins, ybins, bin_type, group_number, group_name, mouse_name, obs_period, days)
                add_figtitle(g.fig, title, y=0.95)
                # save
                fname = "{}_position_{}_xbins{}_ybins{}_grp{}_{}_{}_{}_days".format(
                    experiment.name, bin_type, xbins, ybins, group_number, group_name, mouse_name, days_label)
                save_figure(experiment, g.fig, res_subdir, filename=fname)
                plt.close()