# hcm/visualization/viz/raster_distance_mouseday.py
"""Module for drawing raster plots for single mousedays. """
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
from pandas.plotting import table

from visualization.plot_util import plot_utils
from util import utils
from visualization.plot_util.plot_colors_old import fcols


tset_keys = dict(AS_timeset=dict(offset=60, height=4, color=fcols['active_state'][1]),
                 IS_timeset=dict(offset=61.5, height=0.5, color=fcols['inactive_state'][0]),
                 LB_timeset=dict(offset=54, height=2, color=fcols['locomotion'][1]),
                 FB_timeset=dict(offset=-7, height=2, color=fcols['feeding'][2]),
                 DB_timeset=dict(offset=-7, height=2, color=fcols['drinking'][2]),
                 F_timeset=dict(offset=-13, height=4, color=fcols['feeding'][3]),
                 D_timeset=dict(offset=-13, height=4, color=fcols['drinking'][3]))

tstamp_keys = dict(timestamps_out_hb=dict(offset=48, height=4, color=fcols['locomotion'][3]),
                   timestamps_at_hb=dict(offset=48, height=4, color=fcols['locomotion'][4]))

subkeys = ['offset', 'height', 'color']

load_keys = ['IS_timeset', 'AS_timeset', 'LB_timeset', 't', 'x', 'y', 'idx_timestamps_out_hb', 'idx_timestamps_at_hb',
             'FB_timeset', 'DB_timeset', 'F_timeset', 'D_timeset', 'velocity']


def align_yaxis(ax1, v1, ax2, v2):
    """Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1. """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def set_twinx_layout(ax):
    ax.set_ylim(0, 300)
    yticks = range(0, 201, 20)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.tick_params(labelsize=8)
    for pos in ['right', 'bottom', 'left', 'top']:
        ax.spines[pos].set_visible(False)


def get_xaxis_boundaries(md_data, as_num):
    as_set = md_data['AS_timeset'][as_num-1]
    dx = None
    if np.diff(as_set) > 30 * 60:
        dx = 3  # min
    elif 5 < np.diff(as_set) < 30 * 60:
        dx = 2
    elif np.diff(as_set) < 5 * 60:
        dx = 0.4
    elif np.diff(as_set) < 2 * 60:
        dx = 0.2
    xlims = (as_set[0] // 60 - dx) / 60 - 7, (as_set[1] // 60 + dx) / 60 - 7
    return xlims


def draw_rectangle_around_active_state(ax, tstart, tend):
    pat = patches.Rectangle(xy=(tstart, -12), width=tend - tstart, height=75, lw=1, fc='none', ec='0.5', zorder=0)
    ax.add_patch(pat)


def get_tstep(tstart, tend):
    dt = tend - tstart
    tstep = 0.01
    if dt > 2:
        tstep = 10 / 60.
    elif 1 > dt > 0.5:
        tstep = 5 / 60.
    elif 0.5 > dt > 0.12:
        tstep = 2 / 60.
    elif 0.12 > dt > 3 / 60.:
        tstep = 0.5 / 60.
    elif dt < 3 / 60.:
        tstep = 0.05
    return tstep


def show_durations(ax, tset, offset, key, color):
    for c, x in enumerate(tset):
        xpos = x[0] + (x[1] - x[0]) / 2.
        dt = int((x[1] - x[0]) * 3600)
        mins, sec = int((dt // 60)), int((dt % 60))
        if key == 'AS_timeset':
            ypos = offset + 5
            ax.text(xpos, ypos, "{}\' {:02d}\"".format(mins, sec), fontsize=8, ha='center', va='bottom')
        elif 'IS' not in key:
            text = '{}'.format(dt)
            if key == 'LB_timeset':  # loco bouts
                ypos, fontsize = offset + 2, 3
            elif 'B' in key:  # feed, drink bouts
                text = '{}\"'.format(dt)
                ypos, fontsize = offset + 2, 4
            else:  # events
                ypos, fontsize = offset + 4, 3
            if sec > 0.99:  # arbitrary
                ax.text(xpos, ypos, text, color=color, fontsize=fontsize, ha='center', va='bottom')


def show_timeset_durations(ax, md_data, tstart, tend):
    for key, val in tset_keys.iteritems():
        tset_in_bin = utils.intersect_timeset_with_timebin(md_data[key] / 3600 - 7, (tstart, tend))
        offset, height, color = [val[x] for x in subkeys]
        show_durations(ax, tset_in_bin, offset, key, color)


def zoom_in_xaxis(ax, tstart, tend, tstep=None):
    ax.set_xlim(tstart, tend)
    tstep = get_tstep(tstart, tend) if tstep is None else tstep
    xticks = np.arange(tstart, tend, tstep)
    xticklabels = [utils.hcm_time_to_ct_string(xtick) for xtick in (xticks + 7) * 3600]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=90)


def plot_velocity(ax, md_data, lw=0.3):
    ax2 = ax.twinx()
    ax2.plot(md_data['t'] / 3600 - 7, md_data['velocity'], color='0.5', lw=lw, zorder=0)
    set_twinx_layout(ax2)
    align_yaxis(ax, 0, ax2, 0)
    ax2.set_ylabel('velocity [cm/s]', fontsize=8, color='0.5', rotation=270, labelpad=15)


def plot_active_state_zoom_in(md, as_num, vel=False):
    print "plotting active state #{} zoom-in ..".format(as_num)
    md_data = md.data['preprocessing']
    tstart, tend = get_xaxis_boundaries(md_data, as_num)
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_md(ax, md_data, lw=1)
    # function for xticks nice
    zoom_in_xaxis(ax, tstart, tend)
    draw_feeder_lickometer_position(ax, label_xpos=tend)  # tstart)
    show_timeset_durations(ax, md_data, tstart, tend)
    if vel:
        plot_velocity(ax, md_data)
    return fig


def vel_index(as_tset):
    """Returns False for active states longer than 10mins """
    return False if np.diff(as_tset).T[0] / 60 > 10 else True


def draw_sctive_state_durations(ax, md_data):
    key = 'AS_timeset'
    tset = md_data[key] / 3600 - 7
    offset, height, color = [tset_keys[key][x] for x in subkeys]
    for c, x in enumerate(tset):
        xpos = x[0] + (x[1] - x[0]) / 2.
        ypos = offset + 5
        mins, sec = int((x[1] - x[0]) * 3600. // 60), int((x[1] - x[0]) * 3600. % 60)
        mm, ss = "{}\'".format(mins), "{:02d}\"".format(sec)
        text = '{}\n{}'.format(mm, ss) if mins < 45 else '{} {}'.format(mm, ss)
        ax.text(xpos, ypos, text, fontsize=6, ha='center', va='bottom')


def draw_feeder_lickometer_position(ax, label_xpos):
    ys = [2, 12, 37]
    loc = ['at lickometer', 'at feeder', ' in niche']
    for n, y in enumerate(ys):
        ax.plot([label_xpos - 0.5, label_xpos + 0.5], [y, y], linestyle='--', lw=0.5, color='0.4')
        ax.text(label_xpos, y + 0.2, loc[n], fontsize=6, color='0.4', ha='left', va='bottom')


def draw_raster_labels(ax):
    ax.text(30, 63, 'active vs inactive state', fontsize=6, color='0.4', ha='left', va='center')
    ax.text(30, 54, 'locomotion bouts', fontsize=6, color='0.4', ha='left', va='center')
    ax.text(30, 49, 'and events', fontsize=6, color='0.4', ha='left', va='center')
    ax.text(30, -5, 'feeding/drinking bouts', fontsize=6, color='0.4', ha='left', va='center')
    ax.text(30, -10, 'and events', fontsize=6, color='0.4', ha='left', va='center')


def distance_from_origin_at_lickometer(md_data):
    from util.cage import Cage
    # distance data
    c = Cage()
    keys = ['t', 'x', 'y', 'idx_timestamps_at_hb', 'idx_timestamps_out_hb']
    t, x, y, idx_hb, idx_out = [md_data[x] for x in keys]

    lick_x, lick_y = c.xy_o
    d_origin = np.sqrt(np.power(x - lick_x, 2) + np.power(y - lick_y, 2))

    d_hb, d_out = d_origin.copy(), d_origin.copy()
    d_hb[idx_out] = np.nan
    d_out[idx_hb] = np.nan

    t = utils.hcm_time_to_ct(t)  # circadian time on x-axis
    t_hb, t_out = t.copy(), t.copy()
    t_hb[idx_out] = np.nan
    t_out[idx_hb] = np.nan
    return t_out, t_hb, d_hb, d_out


def set_layout(ax):
    xticks, xticklabels, xlabel = plot_utils.get_ct_bins_xticks_labels(bin_type='12bins')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.xaxis.tick_top()
    ax.set_xlim(5, 30.5)
    ax.set_ylim((-20, 75))
    yticks = range(0, 45, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel('distance from\nlickometer [cm]', fontsize=8, labelpad=5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', which='minor')
    for pos in ['right', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

    plot_utils.plot_vertical_lines_at_ct_12_24(ax)


def draw_timesets(ax, md_data):
    """ Draw raster element for timesets. """
    cnt = 0
    for key, val in tset_keys.iteritems():
        tset = md_data[key] / 3600 - 7
        offset, height, color = [val[y] for y in subkeys]
        for x1, x2 in tset:
            # xy lower left corner
            pat = patches.Rectangle(xy=(x1, offset), width=x2 - x1, height=height, lw=0.001, fc=color, ec=color)
            ax.add_patch(pat)
        cnt += 1


def draw_distance_and_loco_events(ax, md_data, lw):
    """ Draw raster element for events. """
    t_out, t_hb, d_hb, d_out = distance_from_origin_at_lickometer(md_data)
    lkeys = ['timestamps_out_hb', 'timestamps_at_hb']
    lcolors = [tstamp_keys[x]['color'] for x in lkeys]
    ax.plot(t_out, d_out, color=lcolors[0], lw=lw, zorder=5)
    ax.plot(t_hb, d_hb, color=lcolors[1], lw=lw, zorder=5)

    # loco events
    for key, val in tstamp_keys.iteritems():
        timestamps = md_data[key] / 3600 - 7
        offset, height, color = [val[y] for y in subkeys]
        ax.eventplot(timestamps, orientation='horizontal', lineoffsets=offset, linelengths=height, linewidths=0.01,
                     colors=color)


def plot_md(ax, md_data, lw):
    """Draws elements for a single mouseday and sets plot the layout. """
    draw_distance_and_loco_events(ax, md_data, lw)
    draw_timesets(ax, md_data)
    set_layout(ax)


def plot_entire_day(md):
    md.load_npy_data(keys=load_keys)
    md_data = md.data['preprocessing']
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_md(ax, md_data, lw=0.1)
    draw_raster_labels(ax)
    draw_feeder_lickometer_position(ax, label_xpos=30)
    draw_sctive_state_durations(ax, md_data)
#     draw_rectangle_around_active_state(ax, tstart, tend, as_num)
    figtitle = 'Experiment: {}, group{}: {}\nindv{}: {}, day: {}'.format(md.experiment.name, md.group.number,
                                                                         md.group.name, md.mouse.number,
                                                                         md.mouse.name, md.day)
    plot_utils.add_figtitle(fig, figtitle, ypad=-0.05)
    return fig


def plot_all_mds(experiment, obs_period):
    """Plots all rasters for mousedays. """
    from core.keys import obs_period_to_days
    days = obs_period_to_days[experiment.name][obs_period] or experiment.days
    subdir = 'rasters/mousedays'
    for md in experiment.mousedays(days):
        print "plotting raster:", md
        fig = plot_entire_day(md)
        plot_utils.save_figure(md.experiment, fig, subdir, md.filename_long)


def plot_as(md, as_num, vel=None):
    # active state data
    as_tset = md.data['preprocessing']['AS_timeset'][as_num - 1]
    vel = vel or vel_index(as_tset)
    fig = plot_active_state_zoom_in(md, as_num, vel)
    figtitle = 'Experiment: {}\ngroup: {}, {}, day: {}\nActive State #{}'.format(
        md.experiment.name, md.group.name, md.mouse.name, md.day, as_num)
    plot_utils.add_figtitle(fig, figtitle, y=1.12)
    return fig


def plot_manual_zoomin(md, tstart, dt, tstep):
    md_data = md.data['preprocessing']
    tend = tstart + dt
    ts, te = [utils.hcm_time_to_ct_string((x+7)*3600) for x in (tstart, tend)]
    print "plotting manual Zoom-In: CT{} to CT{} ..".format(ts, te)
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_md(ax, md_data, lw=1)
    plot_velocity(ax, md_data)
    zoom_in_xaxis(ax, tstart, tend, tstep)
    show_timeset_durations(ax, md_data, tstart, tend)
    figtitle = 'Experiment: {}\ngroup: {}, {}, day: {}\nZoom-In: CT{} to CT{}'.format(
        md.experiment.name, md.group.name, md.mouse.name, md.day, ts, te)
    plot_utils.add_figtitle(fig, figtitle, y=1.12)
    return fig, ts, te


def add_raw_table(fig, df, mouse_name, day):
    tab = df.xs((mouse_name, day), level=[1, 2], drop_level=False)
    tab.reset_index(level=['event_type', 'error_code'], inplace=True)
    yb, h = -0.3, 0.3
    tot = tab.instances.values.max()
    if tot > 5:
        yb -= 0.1 * (tot / 5.)
        h += 0.1 * (tot / 5.)

    ax2 = fig.add_axes([0.2, yb, 0.5, 0.3])
    table(ax2, tab, bbox=(0.2, 0, 0.8, 1))
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    # # write CT times
    # notes = tab.notes
    # for row in xrange(len(tab)):
    #     ax2.text(0, yb-0.05, notes.values[row])


def draw_mouseday(experiment, df=None, mouse_name=None, day=None, subtype=0, as_num=None, tstart=12, dt=1, tstep=0.1,
                  res_subdir='rasters', f_suffix=None, raw_table=False):
    """Draws raster ploit for a single mouseday. """ 
    md = experiment.mouseday_object(mouse=mouse_name, day=day)
    md.load_npy_data(keys=load_keys)

    f_suffix = '' if f_suffix is None else '_{}'.format(f_suffix)

    print "plotting", md
    fig, fname = None, None
    if subtype == 0:  # entire day
        fig = plot_entire_day(md)
        fname = md.filename_long + f_suffix
        if df is not None and not df.empty and raw_table:  # add_table
            df = df[df.columns[:-1]]
            add_raw_table(fig, df, mouse_name, day)

    elif subtype == 1:  # active state
        if as_num is not None:
            fig = plot_as(md, as_num)
            fname = md.filename_long + '_AS{}{}'.format(as_num, f_suffix)
        else:
            as_set = md.data['preprocessing']['AS_timeset']
            for as_num in xrange(1, len(as_set) + 1):
                fig = plot_as(md, as_num)
                fname = md.filename_long + '_AS{}{}'.format(as_num, f_suffix)
                plot_utils.save_figure(md.experiment, fig, res_subdir, fname)
            return

    elif subtype == 2:  # manual zoom in
        tbin = [utils.ct_to_hcm_time(tstart), utils.ct_to_hcm_time(tstart+dt)]
        md.cut_md_data(tbin)
        fig, ts, te = plot_manual_zoomin(md, tstart, dt, tstep)
        ts, te = [x.replace(':', '') for x in (ts, te)]
        fname = md.filename_long + '_zoomin_CT{}_CT{}{}'.format(ts, te, f_suffix)

    plot_utils.save_figure(md.experiment, fig, res_subdir, fname)
