#/hcm2/visualization/paper_figures/SS2_fig1.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.model.experiment import Experiment
from visualization.raster_distance_mouseday import load_keys, draw_distance_and_loco_events, draw_timesets, \
    set_layout, distance_from_origin_at_lickometer, draw_feeder_lickometer_position, draw_sctive_state_durations, \
    draw_rectangle_around_active_state, zoom_in_xaxis, show_timeset_durations, draw_feeder_lickometer_position
from visualization.plot_util import plot_utils
from visualization.plot_util.plot_colors_old import fcols
from util import utils

tset_keys = dict(AS_timeset=dict(offset=60, height=4, color=fcols['active_state'][1]),
                 IS_timeset=dict(offset=61.5, height=0.5, color=fcols['inactive_state'][0]),
                 LB_timeset=dict(offset=50, height=4, color=fcols['locomotion'][1]),
                 FB_timeset=dict(offset=-10, height=4, color=fcols['feeding'][2]),
                 DB_timeset=dict(offset=-10, height=4, color=fcols['drinking'][2]))


# # mouseday 1
group_name, mouse_name, day = "129S1", "M5104", 10
tstart = 17 + 32/60.  # CT
dt = 28. / 60  # hours
tstep = 0.05  # min

# # mouseday 2
# group_name, mouse_name, day = "WSB", "M2111", 11
# tstart = 14 + 48/60.  # CT
# dt = 28. / 60  # hours
# tstep = 0.05  # min


experiment = Experiment(name="StrainSurvey")
res_subdir = experiment.path_to_results(subdir="SS2_paper_figures")
f_suffix = ""# "_test2"
tsets = ["AS_timeset", "IS_timeset", "FB_timeset", "DB_timeset", "LB_timeset"]
subkeys = ['offset', 'height', 'color']

# load mouseday
md = experiment.mouseday_object(label=(group_name, mouse_name, day))
md.load_npy_data(keys=load_keys)
md_data = md.data['preprocessing']

# all day
fig, ax = plt.subplots(figsize=(10, 3))
# draw distance
t_out, t_hb, d_hb, d_out = distance_from_origin_at_lickometer(md_data)
ax.plot(t_out, d_out, color=fcols['locomotion'][3], lw=0.1, zorder=5)
ax.plot(t_hb, d_hb, color=fcols['locomotion'][4], lw=0.1, zorder=5)
# draw bouts
for key, val in tset_keys.iteritems():
    tset = md_data[key] / 3600 - 7
    offset, height, color = [val[y] for y in subkeys]
    for x1, x2 in tset:
        # xy lower left corner
        pat = patches.Rectangle(xy=(x1, offset), width=x2 - x1, height=height, lw=0.1, fc=color, ec=color)
        ax.add_patch(pat)

set_layout(ax)
draw_sctive_state_durations(ax, md_data)
draw_rectangle_around_active_state(ax, tstart, tstart + dt)

fname = "SSpaper2_fig1_{}".format(md.filename_long + f_suffix)
plot_utils.save_figure(md.experiment, fig, res_subdir, fname)


# # zoomin
tbin = [utils.ct_to_hcm_time(tstart), utils.ct_to_hcm_time(tstart+dt)]
md.cut_md_data(tbin)
tend = tstart + dt
ts, te = [utils.hcm_time_to_ct_string((x + 7) * 3600) for x in (tstart, tend)]
print "plotting manual Zoom-In: CT{} to CT{} ..".format(ts, te)
fig, ax = plt.subplots(figsize=(10, 3))
draw_distance_and_loco_events(ax, md_data, lw=1)
draw_timesets(ax, md_data)
set_layout(ax)
zoom_in_xaxis(ax, tstart, tend, tstep, label_rotation=0)
show_timeset_durations(ax, md_data, tstart, tend)

ts, te = [x.replace(':', '') for x in (ts, te)]
# save figure
fname2 = "SSpaper2_fig1_{}_zoomin_CT{}_CT{}{}".format(md.filename_long, ts, te, f_suffix)
plot_utils.save_figure(md.experiment, fig, res_subdir, fname2)
