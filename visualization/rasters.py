# hcm/visualization/viz/rasters.py
""" Module for drawing mice raster plots. """
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visualization.plot_util import plot_utils
from visualization.plot_util.plot_colors_old import fcols
from core.keys import obs_period_to_days

tset_keys = dict(AS_timeset=dict(offset=3.5, height=2.6, color=fcols['active_state'][1]),
                 IS_timeset=dict(offset=2.5, height=0.6, color='0.5'),
                 LB_timeset=dict(offset=5.8, height=1.5, color=fcols['locomotion'][1]),
                 FB_timeset=dict(offset=7.5, height=1.5, color=fcols['feeding'][2]),
                 DB_timeset=dict(offset=9.2, height=1.5, color=fcols['drinking'][2]))

subkeys = ['offset', 'height', 'color']

load_keys = ['IS_timeset', 'AS_timeset', 'LB_timeset', 'FB_timeset', 'DB_timeset']


def set_layout(ax, num_days, labelsize=6, height=4):
    ymax = 10 * num_days + 5
    ax.set_xlim((5, 30.5))
    ax.set_ylim((0, ymax))
    xticks, xticklabels, xlabel = plot_utils.get_ct_bins_xticks_labels(bin_type='12bins')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(labelsize=labelsize)
    ax.tick_params(axis='x', which='minor')
    for pos in ['right', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

    plot_utils.plot_vertical_lines_at_ct_12_24(ax)

    # add dark cycle rectangle
    ax.add_patch(patches.Rectangle((12, ymax), 12, height=height, linewidth=0.5, facecolor='0.5', clip_on=False,
                                   rasterized=True))


def draw_timesets(ax, md, y_offset, AS_only=False):
    """ Draw raster element for timesets. """
    cnt = 0
    kvs = [(k, v) for k, v in tset_keys.iteritems()]
    kvs2 = [(k, v) for k, v in tset_keys.iteritems() if k in ['AS_timeset', 'IS_timeset']]
    for key, val in kvs if not AS_only else kvs2:
        tset = md.data['preprocessing'][key] / 3600 - 7
        offset, height, color = [val[y] for y in subkeys]
        for x1, x2 in tset:
            # xy lower left corner
            pat = patches.Rectangle(xy=(x1, y_offset - offset), width=x2 - x1, height=height, lw=0.001, fc=color,
                                    ec=color)
            ax.add_patch(pat)
        cnt += 1


def figsubplots(num_items):
    # if 6 < num_items <= 12:
    #     figsize = (12, 8)
    # elif 12 < num_items <= 16:
    #     figsize = (12, 10)
    # elif 16 < num_items <= 20:
    #     figsize = (12, 12)
    figsize = (12, 2.5 * (1 + (num_items-1) // 4))
    nrows, ncols = 1 + (num_items - 1) // 4, num_items if num_items < 4 else 4
    return figsize, nrows, ncols


def group_rasters(experiment, obs_period=(), write_days=True, as_only=False):
    """Draws individual rasters in a panel for all individuals in a group """
    days = obs_period_to_days[experiment.name][obs_period]
    num_days = len(days)
    suffix = obs_period.replace('-', '_').replace(' ', '_')

    for group in experiment.groups:
        num_mice = len(list(group.valid_mice))
        figsize, nrows, ncols = figsubplots(num_mice)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        m = 0
        for mouse in group.mice:
            ax = axes.flatten()[m]
            if not mouse.is_ignored:
                cnt = 0
                for md in mouse.mousedays(days):
                    if not md.is_ignored:
                        print "plotting {}".format(md)
                        y_offset = (num_days - cnt) * 10
                        md.load_npy_data(load_keys)
                        draw_timesets(ax, md, y_offset, as_only)
                        if write_days:
                            ax.text(3.5, y_offset-5, 'day{}'.format(md.day), ha='left', va='bottom', fontsize=4)
                    cnt += 1

                set_layout(ax, num_days)
                ax.set_title("{}".format(mouse.name), y=1.15)
                m += 1

        # turn ignored mice axis off
        for ax in axes.flatten()[len(group.valid_mice):]:
            ax.axis('off')

        figtitle = "{}\n{}\n{} days: {}".format(str(experiment), str(group), suffix, days)
        plot_utils.add_figtitle(fig, figtitle, y=0.97)
        plt.subplots_adjust(hspace=0.4)
        text = '' if not as_only else "AS_only"
        filename = "{}_{}_raster_group{}_{}_{}_days".format(experiment.name, text, group.number, group.name, suffix)
        plot_utils.save_figure(experiment, fig, subdir='rasters', filename=filename)
        plt.close()


def mouse_raster(experiment, obs_period=(), mouse_label=None, write_days=True, as_only=False):
    """Draws a raster plot for one individual. """
    days = obs_period_to_days[experiment.name][obs_period]
    num_days = len(days)
    suffix = obs_period.replace('-', '_').replace(' ', '_')
    mouse = experiment.mouse_object(mouse_label)
    fig, ax = plt.subplots(figsize=(12, 8))

    for cnt, md in enumerate(mouse.mousedays(days)):
        print "plotting {}".format(md)
        y_offset = (num_days - cnt) * 10
        md.load_npy_data(load_keys)
        draw_timesets(ax, md, y_offset, as_only)
        if write_days:
            ax.text(3.5, y_offset-5, 'day{}'.format(md.day), ha='left', va='bottom', fontsize=8)

    set_layout(ax, num_days, labelsize=10, height=2)

    figtitle = "{}\n{}\n{} days: {}".format(str(experiment), str(mouse), suffix, days)
    plot_utils.add_figtitle(fig, figtitle, y=0.97)
    text = '' if not as_only else "AS_only"
    filename = "{}_{}_raster_group{}_{}_indv{}_{}_{}_days".format(experiment.name, text, mouse.group.number,
                                                                  mouse.group.name, mouse.number, mouse.name, suffix)
    plot_utils.save_figure(experiment, fig, subdir='rasters/mice', filename=filename)
    plt.close()


def all_mice_rasters(experiment, obs_period=(), write_days=True, as_only=False):
    """Draws individual rasters for all mice. """
    for g in experiment.groups:
        for m in g.mice:
            mouse_raster(experiment, obs_period, mouse_label=m.label, write_days=write_days, as_only=as_only)
