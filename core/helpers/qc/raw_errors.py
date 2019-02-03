import numpy as np
import os
import pandas as pd

from util import utils, file_utils
from util.intervals import Intervals
from core.keys import hcm_raw_error_code, act_to_actlabel


def check_concurrency_data(tset_errors):
    start_times = [utils.hcm_time_to_ct_string(t) for t in tset_errors[:, 0]]
    durations = np.diff(tset_errors).T[0]
    code, instances, threshold, msg = (6, len(durations), None, "")
    for t, d in zip(start_times, durations):
        msg += "CT{}:({:02.2f}min)\n".format(t, d / 60)
    return code, instances, threshold, msg


def error_starts_durations(tset, long_idx):
    longest_idx = np.where(long_idx > 0)[0]
    starts = [x for x in tset[longest_idx, 0]]
    start_times = [utils.hcm_time_to_ct_string(t) for t in starts]
    durations = [x for x in (tset[longest_idx, 1] - tset[longest_idx, 0])]
    return starts, start_times, durations


def check_long_breaks(tset, recording_start_stop, threshold=10):
    """ check for long time w/o events, according to threshold (hours) """
    tset_breaks = Intervals(tset).complement().intersect(Intervals(recording_start_stop)).intervals
    durations = np.diff(tset_breaks).T[0]
    long_idx = durations > 3600 * threshold
    code, instances, msg = 0, 0, ""
    if long_idx.any():
        code = 5
        _, start_times, durations = error_starts_durations(tset_breaks, long_idx)
        instances, threshold, msg = len(durations), "{}hrs".format(threshold), ""
        for t, d in zip(start_times, durations):
            msg += "CT{}:({:02.2f}hrs)\n".format(t, d / 3600)
    return code, instances, threshold, msg


def check_long_events(tset, threshold=10):
    """check too long events, according to threshold [min] """
    durations = np.diff(tset).T[0]
    long_idx = durations > 60 * threshold
    code, instances, msg, window = 0, 0, "", list()
    if long_idx.any():
        code = 4
        starts, start_times, durations = error_starts_durations(tset, long_idx)
        instances, threshold, msg = len(durations), "{}min".format(threshold), ""
        for n, (t, d) in enumerate(zip(start_times, durations)):
            msg += "CT{}:({:02.2f}min);\n".format(t, d / 60)
            t1 = utils.hcm_time_to_ct(starts[n] - 60)
            dt = (d + 60) / 3600.
            window.append([t1, dt])
    return code, instances, threshold, msg, window


def check_no_events(tset):
    return (0, 0) if len(tset) else (3, 1)


def check_velocity(data, threshold=300):
    t = data['preprocessing']['t']
    vel = data['preprocessing']['velocity']
    idx = vel > threshold  # cm/s
    code, instances, msg, window = 0, 0, "", list()
    if idx.any():
        code = 2
        instances = idx.sum()
        for tt in t[idx]:
            msg += "CT{};\n".format(utils.hcm_time_to_ct_string(tt))
            t1 = utils.hcm_time_to_ct(tt - 600)
            dt = 600 / 3600.
            window.append([t1, dt])
    return code, instances, "{}cm/s".format(threshold), msg, window


def check_platform_drift(data, max_zero_cells=0.1):
    """ checks for too many cells with zero occupancy times, signaling tilted/failed platform,
        in a 12x24 cage discretization
    """
    bin_times = data['preprocessing']['xbins12_ybins24_24H_bin_times']
    always_zero = 14  # at least 14 cells are always zero (niche walls)
    tot_bins = bin_times.size - always_zero
    net_zeros = tot_bins - np.count_nonzero(bin_times) + always_zero
    return (0, net_zeros, ">{}% null cells".format(100*max_zero_cells)) if net_zeros / tot_bins < max_zero_cells \
        else (1, net_zeros, ">{}% null cells".format(100*max_zero_cells))


def raw_errors(experiment, days, remove_code_6=True):
    days = days or experiment.days
    keys = ['xbins12_ybins24_24H_bin_times', 't', 'velocity', 'F_timeset', 'D_timeset', 'recording_start_stop_time']
    md_values = list()
    path = os.path.join(experiment.path_to_binary(experiment, subdir='preprocessing'), keys[0])
    for md in experiment.mousedays_from_path(path, ext='npy'):
        if md.day in days:
            md_label = (md.group.name, md.mouse.name, md.day)
            data = md.load_npy_data(keys)

            # check position density
            code1, inst1, thresh1 = check_platform_drift(data)
            if code1:
                vals = tuple(['L', code1, hcm_raw_error_code[code1], inst1, thresh1, "", list()])
                md_values.append(md_label + vals)

            # check velocity
            code2, inst2, thresh2, msg2, window = check_velocity(data)
            if code2:
                vals = tuple(['L', code2, hcm_raw_error_code[code2], inst2, thresh2, msg2, window])
                md_values.append(md_label + vals)

            # check devices
            for ev_type in ['F', 'D']:
                recording_start_stop = data['preprocessing']['recording_start_stop_time']
                tset = data['preprocessing']['{}_timeset'.format(ev_type)]
                ev_label = act_to_actlabel[ev_type]
                # no events
                code3, inst3 = check_no_events(tset)
                if code3:
                    vals = tuple([ev_label, code3, hcm_raw_error_code[code3], inst3, None, "", list()])
                    md_values.append(md_label + vals)
                # long events
                code4, inst4, thresh4, msg4, window = check_long_events(tset)
                if code4:
                    vals = tuple([ev_label, code4, hcm_raw_error_code[code4], inst4, thresh4, msg4, window])
                    md_values.append(md_label + vals)
                # long breaks
                code5, inst5, thresh5, msg5 = check_long_breaks(tset, recording_start_stop)
                labels_so_far = [(s[0], s[1], s[2], s[3]) for s in md_values]
                if code5 and md_label + (ev_label, ) not in labels_so_far:
                    vals = tuple([ev_label, code5, hcm_raw_error_code[code5], inst5, thresh5, msg5, list()])
                    md_values.append(md_label + vals)

                # position device firing concurrency
                key = '{}_timeset_position_errors'.format(ev_type)
                path = experiment.path_to_binary(experiment, subdir=os.path.join('preprocessing', key))
                for fname in file_utils.find_files(path, ext='npy'):
                    md_label_ = file_utils.mouseday_label_from_filename(fname)
                    if md_label == md_label_:
                        md.load_npy_data([key])
                        tset_errors = md.data['preprocessing'][key]
                        code6, inst6, thresh6, msg6 = check_concurrency_data(tset_errors)
                        if code6:
                            vals = tuple([ev_label, code6, hcm_raw_error_code[code6], inst6, thresh6, msg6, list()])
                            md_values.append(md_label + vals)

    columns = ['group', 'mouse', 'day', 'event_type', 'error_code', 'description', 'instances', 'threshold', 'notes',
               'window']
    df = pd.DataFrame(md_values, columns=columns)
    index = ['group', 'mouse', 'day', 'event_type', 'error_code']
    df.set_index(index, inplace=True)
    mds_list = None
    if remove_code_6:  # remove error_code = 6
        df = df.ix[~(df.index.get_level_values('error_code') == 6)]
        md_values_ = [x for x in md_values if 6 not in x]
        mds_list = [(s[1], s[2], s[4], s[9]) for s in md_values_]  # mouse, day, error_code, window
    return df, mds_list