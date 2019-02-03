# hcm/core/model/mouseday.py
"""
G. Onnis, 11.2017
revised: 11.2018

HCM Experiment Class.

Copyright (c) 2018. All rights reserved.

Tecott Lab UCSF
"""
import numpy as np
import logging
import os

from util.utils import get_timebins
from util.intervals import Intervals
from core.keys import cycle_timebins, act_to_actlabel

logger = logging.getLogger(__name__)


class MouseDay(object):
    """ HCM mouseday class """

    def __init__(self, mouse=None, day=1):
        self.experiment = None or mouse.experiment
        self.group = None or mouse.group
        self.mouse = mouse
        self.day = day
        self.data = dict(experiment=self.experiment.name, group=self.group.name, mouse=self.mouse.name, day=self.day,
                         preprocessing=dict(), position=dict(), features=dict(), events=dict(), breakfast=dict())

    def __str__(self):
        return "group{}: {}, indv{}: {}, day: {}".format(self.group.number, self.group.name, self.mouse.number,
                                                         self.mouse.name, self.day)

    @property
    def label(self):
        """ returns tuple (group_name, mouse_number, day) """
        return self.group.name, self.mouse.name, self.day

    @property
    def is_ignored(self):
        """ returns True if mouseday is ignored """
        return self.label in self.experiment.all_ignored_mousedays()

    @property
    def label_short(self):
        """ returns tuple (group_number, mouse_number, day) """
        return self.group.number, self.mouse.number, self.day

    @property
    def label_long(self):
        """ returns tuple (group_number, group_name, individual_number, mouse_number, day) """
        return self.group.number, self.group.name, self.mouse.number, self.mouse.name, self.day

    @property
    def filename_long(self):
        """ returns string for filename """
        return 'group{}_{}_indv{}_{}_d{}'.format(*self.label_long)

    @property
    def active_states(self):
        """ returns (N, 2) array, N number of active states, (active_state_start_time, active_state_stop_time)
            time is seconds after midnight (HCM time) 
        """
        return self.load_npy_data(keys=['AS_timeset'])[0]

    @property
    def inactive_states(self):
        """ same as active_states(self) for inactive states """
        return self.load_npy_data(keys=['IS_timeset'])[0]

    def bouts(self, ev_type):
        """ same as active_states(self) for bouts, given event_type ('F', 'D', 'L') """
        return self.load_npy_data(keys=['{}B_timeset'.format(ev_type)])[0]

    def events(self, ev_type):
        """ same as active_states(self) for events, given event_type ('F', 'D', 'L') """
        return self.load_npy_data(keys=['{}_timeset'.format(ev_type)])[0]

    @property
    def coordinates(self):
        """ returns list of arrays, components:
            - 0: HCM time t
            - 1: x corrected
            - 2: y corrected
        """
        return self.load_npy_data(['t', 'x', 'y'])

    @property
    def active_states_counts(self):
        """ return number of active states"""
        arr = self.active_states
        return arr.shape[0] if arr is not None else 0

    # load/save data
    def load_raw_data(self):
        """ loads photobeam, lickometer and movement from PE, LE, ME and MSI files """
        from ..helpers import load_raw
        mouse_number = int(self.mouse.name.strip('M'))
        msi_dict = self.experiment.msi_dict[self.group.name][mouse_number][self.day]
        msi_dict['exp_name'] = self.experiment.name
        msi_dict['mouseNumber'] = mouse_number
        self.data['preprocessing'] = load_raw.from_raw(msi_dict)

    def save_npy_data(self, keys=None):
        """ saves preprocessed data as npy stored in mouseday data dictionary """
        data = self.data['preprocessing']
        keys = keys or data.keys()
        for key in keys:
            if not key.startswith('_'):
                path = self.experiment.path_to_binary(os.path.join('preprocessing', key))
                fname = os.path.join(path, '{}.npy'.format(self.filename_long))
                np.save(fname, data[key])

    def load_npy_data(self, keys=None):
        """ loads variables in "keys" into mouseday data dictionary """
        keys = keys or self.experiment.get_preprocessing_data_keys(subdir='preprocessing')
        arrays = list()
        for key in keys:
            path = self.experiment.path_to_binary(subdir=os.path.join('preprocessing', key))
            fname = os.path.join(path, self.filename_long + '.npy')
            try:
                arr = np.load(fname)
            except IOError:
                print "Skipping {} due to: missing .npy file: {}".format(self, fname)
                arr = None
            self.data['preprocessing'][key] = arr
            arrays.append(arr)

        idx_keys = ['idx_timestamps_at_hb', 'idx_timestamps_out_hb']
        if idx_keys[0] in keys or idx_keys[1] in keys:
            # add timestamps for position at/out homebase
            idx_at, idx_out = [self.data['preprocessing'][x] for x in idx_keys]
            arr1, arr2 = [self.data['preprocessing']['t'][x] for x in [idx_at, idx_out]]
            self.data['preprocessing']['timestamps_at_hb'] = arr1
            self.data['preprocessing']['timestamps_out_hb'] = arr2
            arrays.append(arr1)
            arrays.append(arr2)
        return arrays

    def cut_md_data(self, tbin):
        """ intersects mouseday data with tbin = (t_start, t_end) """
        from util.utils import intersect_timeset_with_timebin, timestamps_index_to_timebin
        cut_data = dict()
        for key, arr in self.data['preprocessing'].iteritems():
            if 'timeset' in key:
                cut_data[key] = intersect_timeset_with_timebin(arr, tbin)  # todo: try intersect with interval
            elif key in ['t', 'timestamps_out_hb', 'timestamps_at_hb']:
                idx = timestamps_index_to_timebin(arr, tbin)
                cut_data[key] = arr[idx]
            elif key in ['x', 'y', 'idx_timestamps_out_hb', 'idx_timestamps_at_hb', 'velocity']:
                idx = timestamps_index_to_timebin(self.data['preprocessing']['t'], tbin)
                cut_data[key] = arr[idx]
        self.data['preprocessing'] = cut_data
        return cut_data

    # raw data preprocessing methods
    #  device events
    def raw_ingestion_timeset(self, ev_type):
        """ returns uncorrected PE vs. LE events """
        key = None
        if ev_type == 'F':
            key = '_photobeam_data'
        elif ev_type == 'D':
            key = '_lickometer_data'
        ev = self.data['preprocessing'][key]
        ev[:, 1] += ev[:, 0]
        return ev

    def process_device_data(self):
        """ computes and stroes in mouseday data dictionary:
            - raw (uncorrected) events timesets (intervals)
            - timestamps index at Feeder and Lickometer
            - timesets (intervals) at Feeder and Lickometer
            - timesets for events at device where mouse is not at device cell, if any (2x4 grid discretization)
            - events timesets corrected for position at device
        """
        from util.utils import timeset_at_location, check_ingestion_position_concurrency
        m = [self.data['preprocessing'][x] for x in ['t', 'x', 'y']]
        for ev_type in ['F', 'D']:
            # timesets
            tset_raw = self.raw_ingestion_timeset(ev_type)
            idx, tset_at_loc = timeset_at_location(m, loc=ev_type)
            logger.debug("{}: {} events".format(act_to_actlabel[ev_type], tset_raw.shape[0]))
            # correct position at device
            tset_corrected, tset_errors, error_msg = check_ingestion_position_concurrency(tset_raw, tset_at_loc,
                                                                                          loc=ev_type)
            if len(error_msg):
                logger.error(error_msg)
                # add errors to dict
                self.data['preprocessing']['{}_timeset_position_errors'.format(ev_type)] = tset_errors
                logger.debug("{}: {} events, corrected for position at device".format(act_to_actlabel[ev_type],
                                                                                      tset_corrected.shape[0]))
            # add to dict
            keys = [x.format(ev_type) for x in ['{}_timeset', 'idx_timestamps_at_{}', 'timeset_at_{}']]
            values = [tset_corrected, idx, tset_at_loc]
            for key, val in zip(keys, values):
                self.data['preprocessing'][key] = val

    # position and homebase
    def designate_homebase(self):
        """ returns homebase (single or domino-2 cells) as list of tuples (ybins, xbins)
            based on 24H occupancy times for 2 x 4 discretization of cage.
            coordinates system:
             - (0, 0) top-left niche,
             - (3, 1) bottom-right lickometer
            Most mice will return (0, 0)
        """
        from util.utils import max_domino
        bin_times = self.data['preprocessing']['xbins2_ybins4_24H_bin_times']
        rect = np.unravel_index(bin_times.argmax(), bin_times.shape)  # rectangle with largest time
        if rect == (0, 0):  # and (max_time / tot.sum()) > .5:
            designated = [rect]
            logger.debug("homebase: {}, in Niche".format(rect))  # if largest time at niche
        else:  # if not niche, use domino
            designated, text = max_domino(bin_times)
            logger.debug("homebase: {}, {}, off niche".format(designated, text))
        return designated

    def process_homebase_data(self):
        """designates homebase and computes timesets at homebase """
        from util.cage import Cage
        from core.keys import homebase_2dto1d
        from util.utils import timeset_at_homebase
        c = Cage()
        nestx_3x7 = self.data['preprocessing']['_nest_loc_x']
        nesty_3x7 = self.data['preprocessing']['_nest_loc_y']
        obs_hb = c.map_ethel_obs_to_cage2x4_grid((nestx_3x7, nesty_3x7))
        rect_hb = self.designate_homebase()
        m = [self.data['preprocessing'][x] for x in ['t', 'x', 'y']]  # timestamp data
        idx_hb, tset_at_hb, tset_out_hb, msg = timeset_at_homebase(m, rect_hb)
        logger.debug(msg)
        # map homebase to integer
        rects = np.array([homebase_2dto1d[r] for r in rect_hb])
        obs = np.array([homebase_2dto1d[r] for r in obs_hb])
        # add to dict
        keys = ['rect_hb', 'obs_hb', 'idx_timestamps_at_hb', 'idx_timestamps_out_hb', 'timeset_at_hb', 'timeset_out_hb']
        values = [rects, obs, idx_hb, ~idx_hb, tset_at_hb, tset_out_hb]
        for key, val in zip(keys, values):
            self.data['preprocessing'][key] = val

    def process_platform_data(self):
        """returns delta times between consecutive displacements, related distances traveled and velocity
            todo: angle resp. turning angle
        """
        t, x, y = [self.data['preprocessing'][x] for x in ['t', 'x', 'y']]  # timestamp data
        dt = t[1:] - t[:-1]  # delta_t, s
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))  # total distance, cm
        vel = dist / dt  # velocity, cm/s
        keys = ['delta_t', 'distance', 'velocity']
        values = [dt, dist, vel]
        for key, val in zip(keys, values):
            # pad with zero, value assigned at i (vs. np.insert(var, 0, 0))  # at i+1)
            self.data['preprocessing'][key] = np.append(val, 0)
        logger.debug("movement: {} events".format(t.shape[0]))

    def position_times_binned(self, bin_type, xbins, ybins):
        """ returns dataframe with occupancy times data (timebin, cage_grid_cells)
            timebin based on bin_type:
            - bin_type='7cycles': 24H/DC/LC/AS24H/ASDC/ASLC
            - bin_type='12bins': bin0/bin1/..../bin11
        """
        import pandas as pd
        from util.utils import position_time_xbins_ybins
        keys = cycle_timebins[bin_type]
        num_cells = xbins * ybins
        m = np.vstack(self.coordinates)
        as_tset, is_tset = self.active_states, self.inactive_states
        bin_times = None
        if bin_type == '7cycles':
            bin_times = np.zeros((len(keys), num_cells))
            vals = None
            for k, bintype in enumerate(keys):
                if bintype in ['24H', 'DC', 'LC']:
                    vals = position_time_xbins_ybins(m, xbins, ybins, bin_type=bintype)
                elif bintype in ['AS24H', 'ASDC', 'ASLC']:
                    vals = position_time_xbins_ybins(m, xbins, ybins, bin_type=bintype, as_tset=as_tset)
                elif bintype == 'IS':
                    vals = position_time_xbins_ybins(m, xbins, ybins, bin_type='IS', as_tset=is_tset)
                bin_times[k] = vals.ravel()
            # test
            h24, dc, lc, as24, asdc, aslc = [bin_times[x] / 100 for x in range(6)]
            try:
                np.testing.assert_array_almost_equal(h24, (dc + lc), decimal=1)
            except AssertionError, msg:
                logger.error(msg)
            try:
                np.testing.assert_array_almost_equal(as24, asdc + aslc, decimal=1)
            except AssertionError, msg:
                logger.error(msg)  # arr = bin_times

        elif bin_type in ['12bins', '4bins', '24bins']:
            bin_times = position_time_xbins_ybins(m, xbins, ybins, bin_type=bin_type)
            # test
            h24 = position_time_xbins_ybins(m, xbins, ybins, bin_type='24H')
            try:
                np.testing.assert_almost_equal(bin_times.sum(0), h24, decimal=1)
            except AssertionError, msg:
                logger.error(msg)
            bin_times = bin_times.reshape(len(keys), -1)

        elif bin_type == 'DC2':
            bin_times = position_time_xbins_ybins(m, xbins, ybins, bin_type=bin_type)
            bin_times = bin_times.reshape(len(keys), -1)

        df = pd.DataFrame(bin_times, columns=range(0, num_cells))
        df['timebin'] = keys
        return df

    def create_position(self, bin_type=None, xbins=None, ybins=None, binary=True):
        """ creates position data """
        if binary:
            from util.utils import position_time_xbins_ybins
            for (xbins, ybins) in [(2, 4), (12, 24)]:
                m = np.vstack([self.data['preprocessing'][x] for x in ['t', 'x', 'y']])
                bin_times = position_time_xbins_ybins(m, xbins, ybins, bin_type='24H')  # 24H data
                key = 'xbins{}_ybins{}_24H_bin_times'.format(xbins, ybins)
                self.data['preprocessing'][key] = bin_times  # add to dict as numpy array
                msg = "Total {} bin times [min], xbins:{}, ybins:{}\n{}".format(bin_type, xbins, ybins, bin_times / 60.)
                logger.debug(msg)
                if bin_times.sum() == 0:
                    msg = "No activity during {}".format(bin_type)
                    logger.warning(msg)
        else:
            df = self.position_times_binned(bin_type, xbins, ybins)
            df['group'], df['mouse'], df['day'] = self.label
            self.experiment.data['position'][self.label] = df

    def process_raw_data(self):
        """ adds ingestion, position and homebase data as (key, value) to mouseday data dictionary """
        self.create_position(binary=True)
        self.process_platform_data()
        self.process_homebase_data()
        self.process_device_data()  # if REMOVE: #self.remove_few_drinking_events()

    def create_ingestion_bouts(self, FBD_min=2):
        """ creates ingestion bouts and adds data as (key, value) to mouseday data dictionary.
            hardcoded FBD minimum duration [seconds] and BTT bout connection time threshold
        """
        from util.utils import trim_short_bouts, check_ingestion_bouts_overlap
        for ev_type in ['F', 'D']:
            # get bout thresholds
            bt, btt = self.experiment.hyper_params['{}BT'.format(ev_type)], self.experiment.hyper_params['BTT']
            keys = ['{}_timeset'.format(ev_type), 'timeset_at_{}'.format(ev_type)]
            ev_set, tset_at_device = [self.data['preprocessing'][x] for x in keys]
            b = Intervals()
            for k, x in enumerate(tset_at_device):
                y = Intervals(x).intersect(Intervals(ev_set))  # be at device
                b = b.union(y.connect_gaps(eps=bt))  # apply bout threshold [sec]
            b_btt = b.connect_gaps(eps=btt)  # apply short break time threshold
            num_short = len((b - b_btt).intervals)
            if num_short:
                text_log = "{}: connected {} bout short breaks (threshold={:2.2f}s)".format(act_to_actlabel[ev_type],
                                                                                            num_short, btt)
                logger.debug(text_log)

            bt_set = b_btt.intervals
            if ev_type == 'F':  # work out F short bouts
                text_log = "{}: {} bouts from {} events (threshold={}s)".format(act_to_actlabel[ev_type], len(bt_set),
                                                                                len(ev_set), int(bt))
                logger.debug(text_log)
                # trim short bouts, flag corresponding photobeam events
                bt_set, idx, text_log = trim_short_bouts(ev_set, bt_set, FBD_min)
                logger.debug(text_log)
                # flag corresponding photobeam events
                self.data['preprocessing']['idx_FBD_min_F_timeset'.format(ev_type)] = idx
                text_log = "{}: {} bouts from {} events".format(act_to_actlabel[ev_type], len(bt_set), idx.sum())
                logger.debug(text_log)
            elif ev_type == 'D':
                text_log = "{}: {} bouts from {} events (threshold={}s)".format(act_to_actlabel[ev_type], len(bt_set),
                                                                                len(ev_set), int(bt))
                logger.debug(text_log)
            self.data['preprocessing']['{}B_timeset'.format(ev_type)] = bt_set

            # feeding and licking coefficient data
            coeff_name = 'FC' if ev_type == 'F' else 'LC'
            coeff = 1000 * self.get_ingestion_coefficient(ev_type)
            self.data['preprocessing'][coeff_name] = coeff_name.tolist() if type(coeff_name) is list else coeff_name
            logger.debug("{}: {}={:.4f} mg/s".format(act_to_actlabel[ev_type], coeff_name, coeff))

        # check overlap of feeding vs drinking bouts
        check_ingestion_bouts_overlap(self.data['preprocessing']['FB_timeset'],
                                      self.data['preprocessing']['DB_timeset'])

    def create_locomotion_bouts(self):
        """ creates locomotion bouts and adds data as (key, value) to mouseday data dictionary """
        from util.utils import index_ingestion_bout_and_homebase_timestamps, find_nonzero_runs
        keys = ['t', 'x', 'y', 'distance', 'velocity', 'idx_timestamps_out_hb', 'FB_timeset', 'DB_timeset']
        t, x, y, distance, velocity, idx_out_hb, fbouts, dbouts = [self.data['preprocessing'][x] for x in keys]
        lbvt, lbdt, btt = [self.experiment.hyper_params[h] for h in ['LBVT', 'LBDT', 'BTT']]
        # timestamps not in feeding/drinking bouts and/or homebase
        mask, log_text = index_ingestion_bout_and_homebase_timestamps(t, idx_out_hb, fbouts, dbouts)
        logger.debug(log_text)
        # velocity and distance thresholds
        bout_list = list()
        idx_new = np.zeros(t.shape, dtype=bool)
        # velocity threshold: slower than 1cm/s, it's slow
        idx = (velocity > lbvt) & mask
        b_chunks = find_nonzero_runs(idx)
        for idx_on, idx_off in b_chunks:
            x0, y0 = x[idx_on], y[idx_on]
            x1, y1 = x[idx_off], y[idx_off]
            d01 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            # distance threshold
            if d01 > lbdt:
                idx_new[idx_on: idx_off] = True
                bout_list.append([t[idx_on], t[idx_off]])
            else:
                arrx, arry = x[idx_on: idx_off], y[idx_on: idx_off]
                d1 = np.sqrt((arrx - x0) ** 2 + (arry - y0) ** 2)
                d2 = np.sqrt((arrx - x1) ** 2 + (arry - y1) ** 2)
                if (d1 > lbdt).any() or (d2 > lbdt).any():
                    idx_new[idx_on: idx_off] = True
                    bout_list.append([t[idx_on], t[idx_off]])
                    # # has to leave cell condition  # cells = np.array(cell_ids[idx_on : idx_off])
                    # changed_cell = np.count_nonzero(cells[1:] - cells[:-1]) > 0

        logger.debug("locomotion: {} bouts from {} events".format(len(bout_list), mask.sum()))
        # connect short breaks
        loco_bouts = Intervals(bout_list).connect_gaps(eps=btt).intervals
        if len(loco_bouts) < len(bout_list):
            logger.debug("locomotion: connected {} bout short breaks (threshold={:02.1f}s)".format(
                len(bout_list) - len(loco_bouts), btt))
            # get loco_bouts timestamps index
            lchunk = np.nonzero(np.in1d(t, loco_bouts[:, 0]))[0]
            rchunk = np.nonzero(np.in1d(t, loco_bouts[:, 1]))[0]
            idx_new = np.zeros(t.shape, dtype=bool)
            for l, r in zip(lchunk, rchunk):
                idx_new[l: r] = True

        # test
        assert len(loco_bouts) == len(find_nonzero_runs(idx_new))
        logger.debug("locomotion: {} bouts from {} events".format(len(loco_bouts), mask.sum()))
        # add to dict
        keys = ['LB_timeset', 'idx_timestamps_loco_bouts']
        values = [loco_bouts, idx_new]
        for key, val in zip(keys, values):
            self.data['preprocessing'][key] = val

    def create_active_states(self):
        """ creates active states and adds data as (key, value) to mouseday data dictionary """
        pdata = self.data['preprocessing']
        recording_start_stop = pdata['recording_start_stop_time']
        fb, db, lb = [pdata[x] for x in ['FB_timeset', 'DB_timeset', 'LB_timeset']]
        all_bouts = Intervals(fb) + Intervals(db) + Intervals(lb)
        tot_bouts = all_bouts.intervals.shape[0]
        # interpause = all_bouts.complement().intersect(Intervals(recording_start_stop))
        # apply ist, in seconds
        as_interval = all_bouts.connect_gaps(60 * self.experiment.hyper_params['IST'])
        pdata['AS_timeset'] = as_interval.intervals
        pdata['IS_timeset'] = Intervals(recording_start_stop).intersect(~as_interval).intervals
        logger.debug("active states: {} states from {} feeding, drinking and locomotion bouts\n".format(
            as_interval.intervals.shape[0], tot_bouts))

    # features methods
    # bouts
    def bout_numbers_in_timebin(self, ev_type, tbin):
        """ returns number of bouts in timebin, strict sense """
        from util.utils import timeset_onsets_in_bin
        return timeset_onsets_in_bin(tset=self.data['preprocessing']['{}B_timeset'.format(ev_type)], tbin=tbin)

    def bout_rate_in_timebin(self, ev_type, tbin):
        """ returns rate (1/sec) of bouts in timebin, onset-based """
        cnt = self.bout_numbers_in_timebin(ev_type, tbin)
        return cnt / np.diff(np.asarray(tbin)).sum()

    def bout_durations_in_timebin(self, ev_type, tbin):
        """ returns time spent in bout in timebin, strict sense """
        from util.utils import timeset_durations_in_timebin
        durs = timeset_durations_in_timebin(tset=self.data['preprocessing']['{}B_timeset'.format(ev_type)], tbin=tbin)
        return durs if len(durs) else np.zeros(1)

    def active_state_bout_rate_in_timebin(self, ev_type, tbin):
        """ returns number of bouts per active state time [1/AS.s] in timebin, strict sense """
        as_times = self.active_state_durations_in_timebin(tbin)
        cnt = self.bout_numbers_in_timebin(ev_type, tbin)
        return cnt / as_times.sum() if as_times.sum() else 0

    def active_state_bout_probability_in_timebin(self, ev_type, tbin):
        """ returns ratio of time spent in bout over time spent in active state, or probability, in timebin, strict sense
        """
        as_durs = self.active_state_durations_in_timebin(tbin)
        bt_durs = self.bout_durations_in_timebin(ev_type, tbin)
        return bt_durs.sum() / as_durs.sum() if as_durs.sum() else 0

    def bout_intensities_in_timebin(self, ev_type, tbin, intens_option=1):
        """ returns average bout intensity [g/sec] in timebin, strict sense """
        ref_tset_name = '{}B_timeset'.format(ev_type)
        durs = self.bout_durations_in_timebin(ev_type, tbin)
        amounts, intensity = None, None
        if ev_type in ['F', 'D']:
            amounts = self.ingestion_amounts_in_reference_timeset(ev_type, tbin, ref_tset_name)
        elif ev_type == 'L':
            amounts = self.locomotion_amounts_in_reference_timeset(tbin, ref_tset_name=ref_tset_name)
        if intens_option == 1:  # way 1
            intensity = amounts.sum() / durs.sum() if durs.sum() else 0
        elif intens_option == 2:  # way2
            intensity = amounts / durs if (amounts.sum() and durs.sum()) else np.zeros(1)
        return intensity

    def average_bout_features_binned(self, bin_type, ev_type, intens_option=1):
        """ computes bout features in timebins and adds data as (key, value) to mouseday data dictionary:
            - Bout Numbers, or counts [-]
            - Bout Rate [1/hr]
            - Bout Time [s]
            - AS Bout Rate [1/AS.hr]
            - AS Bout Probability [-]
            - Bout Size [mg], [cm]
            - Bout Duration [s]
            - Bout Intensity [mg/s], [cm/s]
            bin_type can be ['3cycles', '12bins', '4bins', '24bins']
        """
        tbins = get_timebins(bin_type)
        keys = [x.format(ev_type) for x in ['{}BN', '{}BR', 'AS{}BR', '{}BT', 'AS{}BP', '{}BS', '{}BD', '{}BI']]
        # numbers, rates, AS bout rates, timem AS bout probability
        values = [[self.bout_numbers_in_timebin(ev_type, tbin) for tbin in tbins],
                  [self.bout_rate_in_timebin(ev_type, tbin) * 3600 for tbin in tbins],
                  [self.active_state_bout_rate_in_timebin(ev_type, tbin) * 3600 for tbin in tbins],
                  [self.bout_durations_in_timebin(ev_type, tbin).sum() for tbin in tbins],  # sum
                  [self.active_state_bout_probability_in_timebin(ev_type, tbin) for tbin in tbins]]

        # durations
        avg_durs = [self.bout_durations_in_timebin(ev_type, tbin).mean() for tbin in tbins]

        # amounts and intensities
        avg_amounts, avg_intens = None, None
        if ev_type in ['F', 'D']:
            avg_amounts = [self.ingestion_amounts_in_reference_timeset(ev_type, tbin,
                                                                       ref_tset_name='{}B_timeset'.format(
                                                                           ev_type)).mean() * 1000 for tbin in tbins]
            if intens_option == 1:
                avg_intens = [self.bout_intensities_in_timebin(ev_type, tbin, intens_option) * 1000 for tbin in tbins]
            elif intens_option == 2:
                avg_intens = [self.bout_intensities_in_timebin(ev_type, tbin, intens_option).mean() * 1000 for tbin in
                              tbins]
        elif ev_type == 'L':
            avg_amounts = [self.locomotion_amounts_in_reference_timeset(tbin, ref_tset_name='LB_timeset').mean() for
                           tbin in tbins]
            if intens_option == 1:
                avg_intens = [self.bout_intensities_in_timebin(ev_type, tbin, intens_option) for tbin in tbins]
            elif intens_option == 2:
                avg_intens = [self.bout_intensities_in_timebin(ev_type, tbin, intens_option).mean() for tbin in tbins]

        values.extend(
            [avg_amounts, avg_durs, avg_intens])  # add average amounts, durations and intensities to list of values
        for key, val in zip(keys, values):  # add to dict
            self.data['features'][key] = val

    # active state
    def active_state_intensities_in_timebin(self, ev_type, tbin, intens_opt=1):
        """ returns ingestion and locomotion active state intensities [mg/s resp. cm/s] in time bin, in strict sense """
        durs = self.active_state_durations_in_timebin(tbin)
        amounts, intensity = None, None
        if ev_type in ['F', 'D']:
            amounts = self.ingestion_amounts_in_reference_timeset(ev_type, tbin, ref_tset_name='AS_timeset')
        elif ev_type == 'L':
            amounts = self.locomotion_amounts_in_reference_timeset(tbin, ref_tset_name='AS_timeset')
        if intens_opt == 1:  # way1: ignore single contributions, sum them up and divide
            intensity = amounts.sum() / durs.sum() if durs.sum() else 0
        elif intens_opt == 2:  # way2: compute single contributions, then average
            intensity = amounts / durs if (amounts.sum() and durs.sum()) else np.zeros(1)
        return intensity

    def active_state_effective_durations_in_timebin(self, tbin):
        """ returns average duration [sec] of active states in timebin, onset-based """
        from util.utils import effective_timeset_in_timebin
        durs_eff, _ = effective_timeset_in_timebin(tset=self.data['preprocessing']['AS_timeset'], tbin=tbin)
        return np.asarray(durs_eff) if len(durs_eff) else np.zeros(1)

    def active_state_rate_in_timebin(self, tbin):
        """ returns rate (1/sec) of active states in timebin, onset-based """
        cnt = self.active_state_numbers_in_timebin(tbin)
        return cnt / np.diff(np.asarray(tbin)).sum()

    def active_state_numbers_in_timebin(self, tbin):
        """ returns numbers, or counts, of active states in timebin, onset-based """
        from util.utils import timeset_onsets_in_bin
        return timeset_onsets_in_bin(tset=self.data['preprocessing']['AS_timeset'], tbin=tbin)

    def active_state_probability_in_timebin(self, tbin):
        """ returns ratio of time spent in active state over timebin duration, or probability , strict sense
        """
        durs = self.active_state_durations_in_timebin(tbin)
        return durs.sum() / np.diff(np.asarray(tbin)).sum()

    def active_state_durations_in_timebin(self, tbin):
        """ returns time spent [sec] in active state in timebin, strict sense """
        from util.utils import timeset_durations_in_timebin
        return timeset_durations_in_timebin(tset=self.data['preprocessing']['AS_timeset'], tbin=tbin)

    def average_active_state_features_binned(self, bin_type, intens_opt=1):
        """ computes active state features in timebins and adds data as (key, value) to mouseday data dictionary:
            - Time [min]
            - Probability [-]
            - Numbers, or counts [-]
            - Rate [1/hr]
            - Duration [min]
            - active state intensities for Feed [mg/s], Drink [mg/s] and Locomotion [cm/s]
            bin_types can be ['3cycles', '12bins', '4bins', '24bins']
        """
        tbins = get_timebins(bin_type)
        keys = ['AST', 'ASP', 'ASN', 'ASR', 'ASD', 'ASFI', 'ASDI', 'ASLI']
        values = [[self.active_state_durations_in_timebin(tbin).sum() / 60 for tbin in tbins],  # minutes
                  [self.active_state_probability_in_timebin(tbin) for tbin in tbins],
                  [self.active_state_numbers_in_timebin(tbin) for tbin in tbins],
                  [self.active_state_rate_in_timebin(tbin) * 3600 for tbin in tbins],
                  [self.active_state_effective_durations_in_timebin(tbin).mean() / 60. for tbin in tbins]]

        if intens_opt == 1:
            values += [
                [self.active_state_intensities_in_timebin(ev_type='F', tbin=tbin, intens_opt=intens_opt) * 1000 for tbin
                 in tbins],
                [self.active_state_intensities_in_timebin(ev_type='D', tbin=tbin, intens_opt=intens_opt) * 1000 for tbin
                 in tbins],
                [self.active_state_intensities_in_timebin(ev_type='L', tbin=tbin, intens_opt=intens_opt) for tbin in
                 tbins]]

        elif intens_opt == 2:
            values += [
                [self.active_state_intensities_in_timebin(ev_type='F', tbin=tbin, intens_opt=intens_opt).mean() * 1000
                 for tbin in tbins],
                [self.active_state_intensities_in_timebin(ev_type='D', tbin=tbin, intens_opt=intens_opt).mean() * 1000
                 for tbin in tbins],
                [self.active_state_intensities_in_timebin(ev_type='L', tbin=tbin, intens_opt=intens_opt).mean() for tbin
                 in tbins]]

        if bin_type == '3cycles':  # test
            np.testing.assert_almost_equal(np.asarray(values[0][0]), np.asarray(values[0][1] + values[0][2]), decimal=2)
            np.testing.assert_almost_equal(np.asarray(values[2][0]), np.asarray(values[2][1] + values[2][2]), decimal=2)

        for key, val in zip(keys, values):  # add to dict
            self.data['features'][key] = val

    # total amounts
    def locomotion_amounts_in_reference_timeset(self, tbin, loco_qty='distance', ref_tset_name='AS_timeset'):
        """ returns movement total amounts [cm] in timebin.
            reference tset is a 2d array, either AS_timeset or LB_timeset.
            All movements included (at *and* out_of homebase)
            tstamps is a 1d array with t timestamps
            loco_qty is a (tstamps related) 1d array, e.g. {xy, velocity, distance}
            way 2 used here: compute quantity first, then average (to be clarified)
        """
        from util.utils import intersect_timeset_with_timebin, timestamps_index_to_timebin
        from util.utils import array_positions_to_reference_array
        all_tstamps = self.data['preprocessing']['t']
        all_lqty = self.data['preprocessing'][loco_qty]
        ref_arr = intersect_timeset_with_timebin(tset=self.data['preprocessing'][ref_tset_name], tbin=tbin)
        idx = timestamps_index_to_timebin(all_tstamps, tbin)  # intersect timestamps with timebin
        tstamps = all_tstamps[idx]
        lqty = all_lqty[idx]

        all_amounts = list()
        if len(tstamps) and len(ref_arr):
            pos = array_positions_to_reference_array(tstamps, ref_arr)
            all_amounts = [lqty[x[0]:x[1] + 1] for x in pos]
        res_arr = [np.array(all_amounts[n]).sum() for n in xrange(len(all_amounts))]
        return np.array(res_arr) if len(res_arr) else np.zeros(1)

    def device_firing_durations_in_reference_timeset(self, ev_type, tbin, ref_tset_name='AS_timeset'):
        """ returns device firing durations.
            other_tset is a 2d array (timeset), e.g. {FDL}B_timeset, {FD}_timeset
        """
        from util.utils import intersect_timeset_with_timebin, array_positions_to_reference_array
        arr = intersect_timeset_with_timebin(tset=self.get_ingestion_timeset(ev_type), tbin=tbin)
        ref_arr = intersect_timeset_with_timebin(tset=self.data['preprocessing'][ref_tset_name], tbin=tbin)
        all_durs = list()
        if len(arr) and len(ref_arr):
            pos = array_positions_to_reference_array(arr, ref_arr)
            ev_durs = np.diff(arr).T[0]
            all_durs = [ev_durs[pos == p] for p in xrange(len(ref_arr))]
        durs = [all_durs[n].sum() for n in xrange(len(all_durs))]
        return np.array(durs)

    def get_ingestion_coefficient(self, ev_type):
        """ returns feed (resp. Lick Coefficient) for a given day in [g/sec] """
        ev_set, total_consumed = None, None
        if ev_type == 'F':
            ev_set = self.trim_fbd_min_f_timeset()
            total_consumed = self.data['preprocessing']['food_consumed_grams']
        elif ev_type == 'D':
            ev_set = self.data['preprocessing']['D_timeset']
            total_consumed = self.data['preprocessing']['liquid_consumed_grams']
        total_time = np.diff(ev_set).sum()
        return total_consumed / total_time if total_time else 0

    def trim_fbd_min_f_timeset(self):
        """ exclude events in short (<FBD_min) bouts """
        tset = self.data['preprocessing']['F_timeset']
        idx = self.data['preprocessing']['idx_FBD_min_F_timeset']
        return tset[idx] if len(idx) else np.array([])

    def get_ingestion_timeset(self, ev_type):
        """ returns ingestion timeset as (N, 2) array, N number of events, (event_start_time, event_stop_time)
            time is seconds after midnight (HCM time) """
        return self.data['preprocessing']['D_timeset'] if ev_type == 'D' else self.trim_fbd_min_f_timeset()

    def ingestion_amounts_in_reference_timeset(self, ev_type, tbin, ref_tset_name='AS_timeset'):
        """ returns ingestion amounts relative to a timeset (N, 2) array of (start_time, end_time) .
            timesets_name : AS_timeset, {FD}B_timeset
        """
        coeff = self.get_ingestion_coefficient(ev_type)
        durs = self.device_firing_durations_in_reference_timeset(ev_type, tbin, ref_tset_name)
        return coeff * durs if len(durs) else np.zeros(1)

    def average_amounts_features_binned(self, bin_type):
        """ loops through timebins computing total amounts for features in bin for Food [g], Drink [g],
            Locomotion (distance) [m]
        """
        if bin_type in ['3cycles', '12bins', '4bins']:
            tbins = get_timebins(bin_type)
            # ingestion
            for ev_type in ['F', 'D']:
                ivalues = [self.ingestion_amounts_in_reference_timeset(ev_type, tbin).sum() for tbin in tbins]
                # test
                if bin_type == '3cycles':
                    np.testing.assert_almost_equal(ivalues[0], ivalues[1] + ivalues[2], decimal=1)
                self.data['features']['T{}'.format(ev_type)] = ivalues  # add to dict
            # movement
            mvalues = [self.locomotion_amounts_in_reference_timeset(tbin).sum() / 100 for tbin in tbins]  # m
            if bin_type == '3cycles':  # test
                np.testing.assert_almost_equal(mvalues[0], mvalues[1] + mvalues[2], decimal=1)
            self.data['features']['TL'] = mvalues  # add to dict

    def create_features(self, bin_type=None, intens_option=1):
        """ creates a dataframe with features and adds total amounts, active states and bouts features as (key, value)
            to mouseday data dictionary.
            Intensity option refers to the way intensity averages are computed:
            - way 1: amounts.sum() / durations.sum() # go with this one for now
            - way 2: (amounts / durations).mean()
        """
        import pandas as pd
        from core.keys import all_features
        logger.debug("intens_option: {}".format(intens_option))
        self.load_npy_data()  # load all keys
        self.average_amounts_features_binned(bin_type)  # totals  # pass along md object to check numbers
        self.average_active_state_features_binned(bin_type)  # active states
        for ev_type in ['F', 'D', 'L']:  # bouts
            self.average_bout_features_binned(bin_type, ev_type, intens_option)
        # create dataframe
        df = pd.DataFrame(self.data['features'], columns=all_features)
        df['timebin'] = cycle_timebins[bin_type]
        df['group'], df['mouse'], df['day'] = self.label
        self.experiment.data['features'][self.label] = df

    # breakfast
    def create_breakfast(self, timepoint, tbin_size, num_secs):
        """ creates a dataframe with breakfast data and adds to mouseday data dictionary.
            tbin_size in seconds
        """
        import pandas as pd
        num_bins = num_secs / tbin_size
        dfs = list()
        for ev_type in ['F', 'D']:
            evset, ass_set = self.load_npy_data(keys=['{}B_timeset'.format(ev_type), 'AS_timeset'])  # use bout data
            num_as = ass_set.shape[0] if ass_set is not None else 0
            if num_as:
                binary_counts = np.zeros((num_as, num_bins))
                for k in xrange(num_as):
                    if timepoint == 'onset':
                        as_tstart = ass_set[k, 0]  # ONSET
                        as_tend = as_tstart + num_secs
                    elif timepoint == 'offset':
                        as_tend = ass_set[k, 1]
                        as_tstart = as_tend - num_secs

                    as_evset = Intervals(evset).intersect_with_interval(as_tstart, as_tend)
                    tbins = [(i, i + tbin_size) for i in np.linspace(as_tstart, as_tend, num_bins)]
                    binary_counts[k] = [1 if not as_evset.intersect_with_interval(t1, t2).is_empty() else 0 for
                                        b, (t1, t2) in enumerate(tbins)]
            else:
                raise ValueError("no active states this day!")  # binary_counts = np.nan * np.zeros((1, num_bins))
            # create dataframe
            df = pd.DataFrame(np.array(binary_counts).astype(int),
                              columns=range(tbin_size, num_secs + tbin_size, tbin_size))
            df['event'] = act_to_actlabel[ev_type]
            df['as_num'] = ['AS{}'.format(n) for n in range(1, len(binary_counts) + 1)]
            df['group'], df['mouse'], df['day'] = self.label
            dfs.append(df)
        self.experiment.data['breakfast'][self.label] = pd.concat(dfs)

    # within AS structure
    def create_within_as_structure(self, num_mins=15):
        """ returns a generator with 'within active states structure' data """
        keys = ['FB_timeset', 'DB_timeset', 'LB_timeset', 'AS_timeset', 'recording_start_stop_time']
        fb, db, lb, ass, rec = self.load_npy_data(keys=keys)
        bc = (Intervals(fb).union(Intervals(db)).union(Intervals(lb))).complement().intersect(Intervals(rec)).intervals
        for row in ass:
            fa, wa, la, other = [Intervals(row).intersect(Intervals(x)).intervals - row[0] for x in [fb, db, lb, bc]]
            if num_mins is not None:
                fa, wa, la, other = [x[x[:, 0] < 60 * (num_mins + 1)].reshape(-1, 2) if len(x) else x for x in
                                     [fa, wa, la, other]]
            yield ([fa, wa, la, other], np.diff(row)[0])

    # time budget
    def create_time_budget(self, bin_type):
        """ creates a dataframe with breakfast data and adds to mouseday data dictionary """
        import pandas as pd
        from util.utils import get_timebin_in_recording_time
        keys = ['FB_timeset', 'DB_timeset', 'LB_timeset', 'AS_timeset', 'IS_timeset', 'recording_start_stop_time']
        fb, db, lb, ass, iss, rec = self.load_npy_data(keys=keys)

        timebins = cycle_timebins[bin_type]
        res = list()
        for timebin in timebins:
            tbin, tot_time = get_timebin_in_recording_time(timebin, rec, ass)
            ft, dt, lt, asst, isst = [np.diff(Intervals(x).intersect(Intervals(tbin)).intervals).sum() for x in
                                      [fb, db, lb, ass, iss]]
            ot = asst - (ft + dt + lt)
            # test
            if timebin in ['24H', 'DC', 'LC']:
                np.testing.assert_almost_equal(tot_time, asst + isst, decimal=1)
            res.append([ft, dt, lt, ot, asst, isst, tot_time])
        # create dataframe
        columns = [act_to_actlabel[x] for x in ['F', 'D', 'L', 'O', 'AS', 'IS']] + ['total time']
        df = pd.DataFrame(res, columns=columns)
        df['timebin'] = timebins
        df['group'], df['mouse'], df['day'] = self.label
        self.experiment.data['time_budget'][self.label] = df