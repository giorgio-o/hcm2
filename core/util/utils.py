# hcm/util/file_utils.py
""" utilities file """
import numpy as np
from functools import wraps
from time import time

from core.keys import AS_code_lookup, all_feature_keys, act_to_actlabel
from util.intervals import Intervals
from util.cage import Cage


# general
def days_to_obs_period(row, experiment):
    from core.keys import obs_period_to_days
    if experiment.name in ['HiFat2']:
        res = (k for k, vals in obs_period_to_days[experiment.name].items() if
               row['day'] in vals and k not in ['acclimated', 'comparison']).next()
    else:
        res = (k for k, vals in obs_period_to_days[experiment.name].items() if row['day'] in vals).next()
    return res


def flat_out_list(l):
    return [item for sublist in l for item in sublist]


def feature_to_behavior(feature):
    return (key for key, f in all_feature_keys.iteritems() if feature in f).next()


def timing(f):
    """ from: https://codereview.stackexchange.com/questions/169870/decorator-to-measure-execution-time-of-a-function
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        elapsed = time() - start
        mins = elapsed // 60
        text = "{} took {:.2f}s".format(f.__name__, elapsed)
        if mins > 2:
            text += " ({:.0f}m{:.0f}s)".format(mins, elapsed % 60)
        print text
        return result

    return wrapper


# timebins
def get_timebins(bin_type='12bins', shift=0):
    """returns list of time bins start/stop in seconds after midnight
        - 24 hours, DC, LC
        - 12 bins: hours after midnight -- Circadian Time
            bin 0: [ 13.  15.] - CT time: [ 6.  8.]  , military time - 7hrs
            bin 1: [ 15.  17.] - CT time: [ 8.  10.]
            bin 2: [ 17.  19.] - CT time: [ 10.  12.]
            ...
            bin 10:[ 31.  33.] - CT time: [ 24.  2.]
            bin 10:[ 33.  35.] - CT time: [ 2.  4.]
            bin 11:[ 35.  37.] - CT time: [ 4.  6.]
        - 24 bins
        - 4 bins: last 6h LC [6-12], first DC [12-18], last DC [18-24]], first LC [0-6]
    """
    # seconds after midnight, CT6
    starth = 13 * 3600. + shift
    # seconds after midnight, CT30-CT6
    endh = 37 * 3600. + shift
    dc_starth = 19 * 3600.  # CT12
    dc_endh = 31 * 3600.  # CT24
    dc_midh = 25 * 3600.  # CT18
    tbins = None
    if bin_type == '3cycles':  # 24, DC, LC
        tbins = [(starth, endh), (dc_starth, dc_endh), ((starth, dc_starth), (dc_endh, endh))]

    elif bin_type.endswith('bins'):
        digit = int(''.join(c for c in bin_type if c.isdigit()))
        step = 3600 * 24 // digit
        tbins = [(i, i + step) for i in np.arange(starth, endh, step)]
    elif bin_type == '24H':
        tbins = [(starth, endh)]
    elif bin_type == 'DC':
        tbins = [(dc_starth, dc_endh)]
    elif bin_type == 'LC':
        tbins = [[(starth, dc_starth), (dc_endh, endh)]]
    elif bin_type == 'DC2':
        tbins = [(dc_starth, dc_midh), (dc_midh, dc_endh)]
    return tbins


def get_timebin_in_recording_time(timebin, rec, ass):
    arr = None
    if timebin == '24H':
        arr = rec
    elif timebin in ['DC', 'LC']:
        arr = Intervals(get_timebins(timebin)[0]).intersect(Intervals(rec)).intervals
    elif timebin == 'AS24H':
        arr = ass
    elif timebin in ['ASDC', 'ASLC']:
        arr = Intervals(ass).intersect(Intervals(get_timebins(timebin[-2:])[0])).intervals
    return arr, np.diff(arr).sum()


def get_timebins_string(bin_type, b=None):
    """ given a time bin [start, end] converts time after midnight into a string with Circadiam Time"""
    if bin_type in ['12bins', '24bins']:  # , 'ASbins']:
        tbins = np.array(get_timebins(bin_type))
        if bin_type.startswith('AS'):
            starts = ["{:02d}{:02d}".format(*hcm_time_to_ct_string_tuple(t)[:2]) for t in tbins[:, 0]]
            stops = ["{:02d}{:02d}".format(*hcm_time_to_ct_string_tuple(t)[:2]) for t in tbins[:, 1]]
            tbins_string = ["CT{}-{}".format(start, stop) for start, stop in zip(starts, stops)]
        else:  # 12, 24 bins
            bin_step = np.diff(tbins).mean().astype(int) / 3600
            hours = [int(t) if t < 24 else int(t - 24) for t in tbins[:, 0] // 3600 - 7]
            minutes = (tbins[:, 0] % 60).astype(int)
            tbins_string = ["CT{:02d}{:02d}-{:02d}{:02d}".format(h, m, h + bin_step, m) for h, m in zip(hours, minutes)]
    elif bin_type == '3cycles':
        tbins_string = ["CT0600-0600", "CT1200-2400", "CT0600-1200 U CT2400-0600"]
    elif bin_type == '24H':
        tbins_string = "CT0600-0600"
    elif bin_type == 'DC':
        tbins_string = "CT1200-2400"
    elif bin_type == 'LC':
        tbins_string = "CT0600-1200 U CT2400-0600"
    else:
        tbins_string = ""
    return tbins_string if b is None else tbins_string[b]


def recode_inds2(inds, bin_type):
    """ recodes bin boundaries indices at 12 and 24 for code2"""
    if bin_type == '12bins':
        inds[2, 1] = 1  # CT10-12 is LC
        inds[5, 1] = 2  # CT16-18 is DC1
        inds[8, 1] = 3  # CT22-24 is DC
    else:
        pass
    return inds


def recode_boundary_inds1(inds, bin_type):
    """ recodes bin boundaries indices at 12 and 24 for code1 """
    if bin_type == '12bins':
        inds[2, 1] = 1  # CT10-12 is LC
        inds[8, 1] = 2  # CT22-24 is DC
    else:
        pass
    return inds


def recode_light_cycle_indices(inds):
    """ recodes primary code for first part of LC, CT00-CT06
        leaving code 13 intact for AS that span the entire DC
    """
    tups = map(tuple, inds)
    new_tups = list()
    for c, tup in enumerate(tups):
        if tup == (2, 3):
            tup = (2, 1)
        elif tup == (3, 3):
            tup = (1, 1)
        new_tups.append(tup)
    return np.array(new_tups)


def recode_indices1(inds, bin_type):
    new_inds = recode_light_cycle_indices(inds)
    if bin_type == '12bins':
        new_inds = recode_boundary_inds1(new_inds, bin_type)
    return new_inds


def get_timebin_category(tbins, bin_type):
    """ assigns timebin to a category based on start and end times
        each time of day is coded with a number:
        - cat_code1, primary
        - 1 : LC, CT06-12 union CT24-06
        - 2 : DC, CT12-18
        - cat_code2, secondary
        - 1 : CT06-12, or last part of LC, or LC2
        - 2 : CT12-18, or first part of DC, or DC1
        - 3 : CT18-24, or second part of DC, or DC2
        - 4 : CT24-06, or first part of LC, or LC1
        categories are formed by pairing numbers for start and end.
        categories '13' (primary) and '14' (secondary) ought to be rare
        returns cateogry code, descriptions and proportion of time spent in DC and LC
    """
    # seconds after midnight
    dc_start = 19 * 3600.  # CT12
    dc_mid = 25 * 3600.  # CT18
    dc_end = 31 * 3600.  # CT24
    starts, ends = np.array(tbins).T

    # primary, DC and LC
    bins = [0, dc_start, dc_end, 1000000]
    inds1 = np.array([np.digitize(x, bins) for x in [starts, ends]]).T
    inds1 = recode_indices1(inds1, bin_type)
    cat_code1 = [''.join(str(i) for i in ind) for ind in inds1]

    # secondary, day divided in 4 bins
    bins = [0, dc_start, dc_mid, dc_end, 1000000]
    inds2 = np.array([np.digitize(x, bins) for x in [starts, ends]]).T
    if not bin_type.startswith('AS'):
        inds2 = recode_inds2(inds2, bin_type)
    cat_code2 = [''.join(str(i) for i in ind) for ind in inds2]

    # proportions in LC and DC
    dc, lc = [Intervals(get_timebins('3cycles')[x]) for x in [1, 2]]

    return cat_code1, [AS_code_lookup['1'][x] for x in cat_code1], cat_code2, [AS_code_lookup['2'][x] for x in
                                                                               cat_code2], [
               Intervals(x).intersect(dc).measure() / Intervals(x).measure() for x in tbins], [
               Intervals(x).intersect(lc).measure() / Intervals(x).measure() for x in tbins]


# # hcm time conversion
def hcm_time_to_ct(hcm_time):
    """ converts hcm time (seconds after midnight) to circadian time"""
    return (hcm_time / 3600) - 7


def ct_to_hcm_time(ct_time):
    """ converts hcm time (seconds after midnight) to circadian time"""
    return (ct_time + 7) * 3600.


def hcm_time_to_ct_string_tuple(hcm_time, FLOAT=False):
    """ converts hcm time (seconds after midnight) to circadian time
        returns a tuple
    """
    mm, ss = divmod(hcm_time, 60)
    h, m = divmod(mm, 60)
    h -= 7
    if h > 24:
        h -= 24
    return (int(h), int(m), int(ss)) if not FLOAT else (int(h), int(m), ss)


def hcm_time_to_ct_string(hcm_time):
    """ converts hcm time (seconds after midnight) to circadian time
        returns a string
    """
    return "{:02d}:{:02d}:{:02d}".format(*hcm_time_to_ct_string_tuple(hcm_time))


def hcm_time_to_ct_string_float(hcm_time):
    """ converts hcm time (seconds after midnight) to circadian time
        returns a string
    """
    return "{:02d}:{:02d}:{:02.3f}".format(*hcm_time_to_ct_string_tuple(hcm_time, FLOAT=True))


def seconds_to_mins_and_secs_tuple(secs):
    return int(secs) // 60, int(secs % 60)


def seconds_to_mins_and_secs(secs):
    return "{}\'{}\'\'".format(*seconds_to_mins_and_secs_tuple(secs))


# timesets and timebins
def find_nonzero_runs(arr):
    """Returns array that is 1 where arr is nonzero, and pad each end with an extra 0."""
    arr[-1] = False  # avoid problems
    isnonzero = np.concatenate(([0], (np.asarray(arr) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def timeset_onsets_in_bin(tset, tbin):
    cnt = 0  # onsets
    if np.array(tbin).size < 4:  # not LC
        for start, end in tset:
            if tbin[0] < start < tbin[1]:
                cnt += 1
    else:
        for start, end in tset:
            for tb in tbin:
                if tb[0] < start < tb[1]:
                    cnt += 1
    return cnt


def effective_timeset_in_timebin(tset, tbin):
    """ returns effective timeset in tbin and related effective time bin, in an onset-based sense or,
        based on when active states starts (onset) and not when it ends
    """
    tbin_start, tbin_end = None, None
    tset_durs_eff = list()
    if np.array(tbin).size < 4:  # not LC
        cnt = 0  # onsets
        for start, end in tset:
            if tbin[0] < start < tbin[1]:
                tset_durs_eff.append(end - start)
                if cnt < 1:
                    tbin_start = start
                tbin_end = end
                cnt += 1
        tbin_eff = [tbin_start, tbin_end] if cnt else list()
    else:  # LC
        tbin_eff = list()
        for start, end in tset:
            cnt = 0  # onsets
            for tb in tbin:
                if tb[0] < start < tb[1]:
                    tset_durs_eff.append(end - start)
                    if cnt < 1:
                        tbin_start = start
                    tbin_end = end
                    cnt += 1
            if cnt:
                tbin_eff.append([tbin_start, tbin_end])
    return tset_durs_eff, tbin_eff


def intersect_timeset_with_timebin(tset, tbin):
    return Intervals(tset).intersect(
        Intervals(tbin)).intervals  # todo: try: Intervals(tset).intersect_with_interval(tbin[0], tbin[1]  # faster


def timeset_durations_in_timebin(tset, tbin):
    """ return timeset durations in tbin, in a strict sense, or, intersected with bin """
    tset_in_bin = intersect_timeset_with_timebin(tset, tbin)
    tset_durs = np.diff(tset_in_bin).T[0] if len(tset_in_bin) > 0 else np.array([])
    return tset_durs


def array_positions_to_reference_array(arr, ref_arr):
    """ returns position of intervals in 'lower' (hierarchically, model speaking) timeset (tipically bouts resp. events)
        relative to intervals in a 'higher' timeset (typically active states resp. bouts)
    """
    ref_arr_copy = ref_arr.copy()
    posx0 = None
    if arr.ndim == 2:  # 2d arrays, e.g. timeset intervals
        # pos of flattened array
        ref_arr_copy[:, 0] -= 0.1
        ref_arr_copy[:, 1] += 0.1
        pos = np.searchsorted(ref_arr_copy.flatten(), arr)
        posx = pos // 2
        # test all pos are odd numbers, only for valid mousedays - for now
        np.testing.assert_array_equal(pos % 2, np.ones(arr.shape),
                                      err_msg='event in between reference timeset intervals')
        assert np.diff(posx).sum() == 0, 'start end endtime should have same position'
        posx0 = posx[:, 0]  

    elif arr.ndim == 1:  # 1d arrays, e.g. timestamps
        # here, there could be events in between ref_tset intervals, e.g active states (think move at homebase)
        posx0 = np.searchsorted(arr, ref_arr)
    return posx0


def device_firing_durations_in_reference_timeset(other_set, ref_set, tbin, tset_name):
    """ other_tset is a 2d array (timeset), e.g. {FDL}B_timeset, {FD}_timeset """
    arr = intersect_timeset_with_timebin(other_set, tbin)
    ref_arr = intersect_timeset_with_timebin(ref_set, tbin)
    all_durs = list()
    if len(arr) and len(ref_arr):
        pos = array_positions_to_reference_array(arr, ref_arr, tset_name)
        ev_durs = np.diff(arr).T[0]
        all_durs = [ev_durs[pos == p] for p in xrange(len(ref_arr))]
    durs = [all_durs[n].sum() for n in xrange(len(all_durs))]
    return np.array(durs)


def timestamps_index_to_timebin(timestamps, tbin):
    """ intersect timestamp array with timebin
        returns movement timestamps index in timebin.
        All movements included (at and out homebase)
    """
    if np.array(tbin).size < 4:  # 24H, DC, 12bins
        idx = (timestamps >= tbin[0]) & (timestamps < tbin[1])
    else:  # LC
        idx1 = (timestamps >= tbin[0][0]) & (timestamps < tbin[0][1])
        idx2 = (timestamps >= tbin[1][0]) & (timestamps < tbin[1][1])
        idx = idx1 | idx2
    return idx


# Position density
def pull_locom_tseries_subset(m, start_time=0, stop_time=300):
    """
    given an (m x n) numpy array m where the 0th row is array of times [ASSUMED SORTED]
    returns a new array (copy) that is a subset of m corresp to start_time, stop_time

    returns [] if times are not in array
    (the difficulty is that if mouse does not move nothing gets registered
     so we should artificially create start_time, stop_time movement events at boundaries)
    """
    t = m[0]
    idx_start = t.searchsorted(start_time)
    idx_stop = t.searchsorted(stop_time)
    new_m = m[:, idx_start:idx_stop].copy()
    if idx_stop != t.shape[0]:
        if (idx_start != 0) and (t[idx_start] != start_time):
            v = np.zeros(m.shape[0])
            v[1:] = m[1:, idx_start - 1].copy()
            v[0] = start_time
            v = v.reshape((m.shape[0], 1))
            new_m = np.hstack([v, new_m])
        if (t[idx_stop] != stop_time) and (idx_stop != 0):
            v = np.zeros(m.shape[0])
            v[1:] = m[1:, idx_stop - 1].copy()  # find last time registered
            v[0] = stop_time
            v = v.reshape((m.shape[0], 1))
            new_m = np.hstack([new_m, v])
        elif t[idx_stop] == stop_time:
            v = m[:, idx_stop].copy().reshape((m.shape[0], 1))
            new_m = np.hstack([new_m, v])
    else:
        pass
    return new_m


def total_time_rectangle_bins(m, xlims=(0, 1), ylims=(0, 1), xbins=5, ybins=10):
    """
    given an (3 x n) numpy array m where the 0th row is array of times [ASSUMED SORTED]
    returns a new (xbins x ybins) array (copy) that contains PDF of location over time
    """
    xmin, xmax = xlims
    ymin, ymax = ylims
    meshx = xmin + (xmax - xmin) * 1. * np.array(range(1, xbins + 1)) / xbins
    meshy = ymin + (ymax - ymin) * 1. * np.array(range(1, ybins + 1)) / ybins

    cnts = np.zeros((ybins, xbins))
    if m.shape[0] <= 1:
        return cnts

    bin_idx = meshx.searchsorted(m[1, :], side='right')
    bin_idy = meshy.searchsorted(m[2, :], side='right')
    for k in xrange(m.shape[1] - 1):
        if bin_idx[k] == xbins:
            bin_idx[k] -= 1
        if bin_idy[k] == ybins:
            bin_idy[k] -= 1
        cnts[ybins - bin_idy[k] - 1, bin_idx[k]] += m[0, k + 1] - m[0, k]
    return cnts


def time_in_xbins_ybins(m, xbins, ybins, tbin):
    c = Cage()
    xlims, ylims = c.cage_boundaries
    pos_subset = pull_locom_tseries_subset(m, tbin[0], tbin[1])
    return total_time_rectangle_bins(pos_subset, xlims, ylims, xbins, ybins)


def position_time_xbins_ybins(m, xbins, ybins, bin_type, as_tset=None):
    """
    given a cycle ['24H', 'IS', 'DC', 'LC', 'AS24H', 'ASDC', 'ASLC', IS],
    returns total time spent at each grid cell in cycle
    """
    bin_times = np.zeros([ybins, xbins])
    if bin_type in ['24H', 'DC']:
        tbin, = get_timebins(bin_type)
        bin_times = time_in_xbins_ybins(m, xbins, ybins, tbin)
    elif bin_type == 'LC':
        tbins, = get_timebins(bin_type)
        for tbin in tbins:
            bin_times += time_in_xbins_ybins(m, xbins, ybins, tbin)

    elif bin_type in ['AS24H', 'ASDC', 'ASLC', 'IS']:
        bintype = bin_type.strip('AS') if bin_type != 'IS' else '24H'
        tbins, = get_timebins(bin_type=bintype)
        cut_bins = intersect_timeset_with_timebin(as_tset, tbins)
        for tbin in cut_bins:
            bin_times += time_in_xbins_ybins(m, xbins, ybins, tbin)

    elif bin_type in ['12bins', '4bins', 'DC2', '24bins']:
        tbins = get_timebins(bin_type)
        bin_times = np.zeros([len(tbins), ybins, xbins])
        for n, tbin in enumerate(tbins):
            bin_times[n] = time_in_xbins_ybins(m, xbins, ybins, tbin)
    return bin_times


# homebase
def index_move_timestamps_at_homebase(x, y, rect_hb):
    """returns a boolean array for t timestamps under condition 'mouse is at homebase'.
    homebase is given as a rectangle in a 2 x 4 dicretization of cage.
    note: different from method 'index_move_timestamps_at_location; in that niche coordinates
    are slightly corrected accounting for niche walls and enclosure.
    """
    c = Cage()
    idx_list = []
    for rect in rect_hb:
        tl, tr, bl, br = c.map_rectangle_to_cage_coordinates(rect)
        # print "Homebase location in original coordinates top_left, top_right, bot_left, bot_right: "
        # print tl, tr, bl, br
        if rect == c.activity_to_rectangle['N']:
            # these are correct cage boundaries -/+something to allow for match
            # with gridcells in a 2x4 discretization
            less_than_x = c.nestRightX - 1.2
            greater_than_y = c.nestBottomY + 0.7
            # 200x faster
            idx = (x < less_than_x) & (y > greater_than_y)
            idx_list.append(idx)
        else:
            idx1 = (x > tl[0]) & (x < tr[0])
            idx2 = (y < tl[1]) & (y > bl[1])
            idx = idx1 & idx2
            idx_list.append(idx)

    if len(idx_list) == 1:
        # single homebase
        idx = idx_list[0]
    else:
        # domino homebase
        idx = idx_list[0] | idx_list[1]
    msg = "movement: {} events (off-homebase) vs {} (at-homebase)".format(x.shape[0] - sum(idx), sum(idx))
    return idx, msg


def timeset_at_homebase(m, rect_hb=None):
    """returns array of start/stop times at devices (or niche) cell in a (2, 4) cage
    discretization over the 24 hours. rect=None is for Homebase (single or domino)
    """
    t, x, y = m
    idx_hb, msg = index_move_timestamps_at_homebase(x, y, rect_hb)
    idx_runs_hb = find_nonzero_runs(idx_hb)
    list_of_times_hb = [[t[start], t[end]] for start, end in idx_runs_hb]
    idx_runs_out_hb = find_nonzero_runs(~idx_hb)
    list_of_times_out_hb = [[t[start], t[end]] for start, end in idx_runs_out_hb]
    return idx_hb, np.asarray(list_of_times_hb), np.asarray(list_of_times_out_hb), msg


def index_move_timestamps_at_location(x, y, loc='F'):
    """returns a boolean array for t timestamps under condition: 'mouse is at rectangle'
    (Niche/Feeder/Lickometer or other) in a (2, 4) cage discretization
    position given by rectangle coordinates:
            (0, 0)  top-left (niche)
            (3, 0)  feeder
            (3, 1)  lickometer
    """
    c = Cage()
    rect = c.activity_to_rectangle[loc]
    # check if single or domino cell
    idx = None
    if len(rect) == 1:
        tl, tr, bl, br = c.map_rectangle_to_cage_coordinates(rect[0])
        if loc == 'N':  # Niche
            less_than_x = tr[0]
            more_than_y = bl[1]
            idx = (x < less_than_x) & (y < more_than_y)
        elif loc == 'F':  # Feeder
            less_than_x = tr[0]
            less_than_y = tl[1]
            idx = (x < less_than_x) & (y < less_than_y)
        elif loc == 'D':  # Lickometer
            more_than_x = tl[0]
            less_than_y = tl[1]
            idx = (x > more_than_x) & (y < less_than_y)
        else:  # generic (2, 4) cell location
            idx1 = (x > tl[0]) & (x < tr[0])
            idx2 = (y < tl[1]) & (y > bl[1])
            idx = idx1 & idx2

    elif len(rect) == 2:  # domino cell
        idxs = []
        for r in rect:
            tl, tr, bl, br = c.map_rectangle_to_cage_coordinates(r)
            idx1 = (x > tl[0]) & (x < tr[0])
            idx2 = (y < tl[1]) & (y > bl[1])
            idxs.append(idx1 & idx2)
        idx = idxs[0] & idxs[1]
    return idx


def timeset_at_location(m, loc='F'):
    """returns array of start/stop times at devices (or niche) cell in a (2, 4) cage
    discretization over the 24 hours. rect=None is for Homebase (single or domino)
    """
    t, x, y = m
    idx = index_move_timestamps_at_location(x, y, loc)
    idx_runs = find_nonzero_runs(idx)
    return idx, np.asarray([[t[start], t[end]] for start, end in idx_runs])


def max_domino(tot):
    """ given top left is (0, 0) and given a cell idx = (y, x), returns
        three kinds of dominoes and sums in tot: (L)eft, (R)ight, (M)iddle
    """
    ybins, xbins = tot.shape
    # set default values
    L = [(0, 0), (1, 0)]
    R = [(0, 1), (1, 1)]
    M = [(0, 0), (0, 1)]
    L_max = 0
    for y in xrange(ybins - 1):
        domino_tot = tot[y, 0] + tot[y + 1, 0]
        if domino_tot > L_max:
            L = [(y, 0), (y + 1, 0)]
            L_max = domino_tot
    R_max = 0
    for y in xrange(ybins - 1):
        domino_tot = tot[y, 1] + tot[y + 1, 1]
        if domino_tot > R_max:
            R = [(y, 1), (y + 1, 1)]
            R_max = domino_tot
    M_max = 0
    for y in xrange(ybins):
        domino_tot = tot[y, 0] + tot[y, 1]
        if domino_tot > M_max:
            M = [(y, 0), (y, 1)]
            M_max = domino_tot
    all_arr = [L, R, M]
    idx = np.argmax([L_max, R_max, M_max])
    # domino homebase, , coordinates in (2 ,4)
    hb_domino = all_arr[idx]
    domino_times = [tot[hb_domino[0]] / tot.sum(), tot[hb_domino[1]] / tot.sum()]
    id_max = np.argmax(domino_times)  # either 0 or 1
    # largest time in domino
    hb_max = hb_domino[id_max]

    if domino_times[1 - id_max] > 0.25:
        if domino_times[1 - id_max] / domino_times[id_max] > 0.5:
            text = 'domino'
            designated = hb_domino
        else:
            text = 'single'
            designated = [hb_max]
    else:
        text = 'single'
        designated = [hb_max]
    return designated, text


# device events
def check_ingestion_position_concurrency(tset_raw, tset_at_loc, loc='F'):
    """ check device events and position at device concurrency
        returns array with corrected events and errors removed
        this function is not used starting May 8th, 2018
    """
    i_raw = Intervals(tset_raw)
    i_at_device = Intervals(tset_at_loc)
    i_true = i_raw.intersect(i_at_device)
    tset_errors = (i_raw - i_true).intervals  # photobeam firing when mouse not at device
    np.testing.assert_almost_equal(np.diff(tset_raw).sum(),
                                   np.diff(tset_errors).sum() + np.diff(i_true.intervals).sum(), decimal=2)
    num_errors = len(tset_errors)
    tot = i_raw.intervals.shape[0]
    err_time = np.diff(tset_errors).sum() / 60.
    error_msg = ""
    if num_errors:
        # todo: check this is working. plot on rasters
        tset = i_true.intervals
        error_msg = "{}: {}/{} events removed, device firing while mouse not at device " \
                    "(total time involved={:.2f} min)".format(act_to_actlabel[loc], num_errors, tot, err_time)
    else:
        tset = tset_raw
    return tset, tset_errors, error_msg


# bouts
def check_ingestion_bouts_overlap(fbouts, dbouts):
    overlap = Intervals(fbouts).intersect(Intervals(dbouts))
    # fdur, ddur = np.diff(f_bouts).T[0], np.diff(d_bouts).T[0]
    if not overlap.is_empty():
        overlap_diff = np.diff(overlap.intervals).T[0]
        for n in xrange(len(overlap_diff)):
            logger.error("feeding and drinking bouts: found overlap at CT{} for {:02.3f}sec".format(
                hcm_time_to_ct_string_float(overlap.intervals[n, 0]), overlap_diff[n]))


def trim_short_bouts(ev_set, bt_set, FBD_min=2):
    event_is_in_long_bout = np.ones(ev_set.shape, dtype=bool)
    bt_set_trimmed = np.array([])
    text_log = None
    if len(bt_set):
        num1 = len(bt_set)
        # flag events in bout shorter than FBD_min
        is_long_bout = np.diff(bt_set).T[0] > FBD_min
        bt_set_short = bt_set[~is_long_bout]
        # old_pos = np.searchsorted(ev_set[:, 0], bt_set_short[:, 0]) # wrong, misses consecutive events
        # use searchsorted, seems best on contiguous arrays
        # https://stackoverflow.com/questions/15139299/performance-of-numpy-searchsorted-is-poor-on-structured-arrays
        # https://stackoverflow.com/questions/26998223/akind-is-the-difference-between-contiguous-and-non-contiguous-arrays
        pos = np.searchsorted(ev_set.flatten(), bt_set_short)
        posx = pos // 2
        posxf_nested = [[x[0]] if np.diff(x)[0] == 0 else range(x[0], x[1] + 1) for x in posx]
        posxf = [item for sublist in posxf_nested for item in sublist]
        event_is_in_long_bout = np.array([True if i not in posxf else False for i in xrange(len(ev_set))])

        # remove short bouts
        text_log = ""
        bt_set_trimmed = bt_set[is_long_bout]
        num2 = len(bt_set_trimmed)
        if num1 - num2:
            text_log = "{}: removed {} short bouts ({} events, threshold={}sec) "\
                .format(act_to_actlabel['F'], num1 - num2, (~event_is_in_long_bout).sum(), FBD_min)
    return bt_set_trimmed, event_is_in_long_bout, text_log


def index_ingestion_bout_and_homebase_timestamps(t, idx, fbouts, dbouts):
    """returns a mask which excludes timestamps indexes corresponding to ingestion bouts
        and position at homebase
    """
    i_fdb = Intervals(fbouts).union(Intervals(dbouts))
    mask = np.ones(t.shape[0], dtype=bool)
    for timestamp in xrange(t.shape[0]):
        if i_fdb.contains(t[timestamp]):
            mask[timestamp] = False

    mask = mask & idx
    log_text = "locomotion: flagged {} events occurring at homebase and/or during ingestion bouts".format(
        mask.shape[0] - mask.sum())
    return mask, log_text
