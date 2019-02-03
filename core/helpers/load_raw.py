# hcm/core/helpers/load_raw.py
""" loads HCM raw data """

import numpy as np
import os
import re
import shutil
import logging

from util.file_utils import datadir

logger = logging.getLogger(__name__)


def load_am_data(AM_filename):
    """"loads AM file data. supposedly, the format is:
        - 4th element (6 bytes in) is start bin
        - 5th element (8 bytes in) is bin size (=30)
        - 7th element (12 bytes in) is end bin
    """
    try:
        result = np.fromfile(AM_filename, dtype='H', count=7)  # Pull 1st 7 2-byte elements (type short) from AM file
    except IOError:
        result = np.array([])
    return result


def remove_bad_line(filename):
    """ due to some fuck up, last lines are often corrupt. """
    logger.info("Removing last line from %s." % filename)
    shutil.copyfile(filename, filename + ".orig")
    lines = open(filename).readlines()
    while len(lines[-1]) <= 1:
        lines = lines[:-1]
    open(filename, "w").writelines(lines[:-1])


def load_movement_data(data, me_filename):
    """loads uncorrected movement data md.t, md.x, md.y"""
    logger.debug('ME_filename: {}'.format(me_filename))
    movement_data = np.loadtxt(me_filename, delimiter=",", dtype=np.double, ndmin=2)
    # originally this was:
    # AMData = np.fromfile(fullFileAM, dtype='H', count=7)  # Pull 1st 7 2-byte elements (type short) from AM file

    me_shape = movement_data.shape
    extended_me = np.zeros((me_shape[0] + 1, me_shape[1]), dtype=np.double)
    extended_me[:-1, :] = movement_data
    movement_data = extended_me
    # Convert x and y positions to cm
    movement_data /= 1000.0
    # Get the delta x and y data
    delta_x, delta_y = movement_data[:, 1], movement_data[:, 2]

    # Convert to cumulative times and positions
    movement_data = np.cumsum(movement_data, axis=0)

    # recording start / end
    # try to use AMData to get end of recording time
    # self.movement_data[-1, 0] = self.light_data[1] / 1000.0       #does not work
    bin_start = data['_light_data'][3]  # 4th element (6 bytes in) is start bin
    bin_size = data['_light_data'][4]  # 5th element (8 bytes in) is bin size (=30)
    bin_end = data['_light_data'][6]  # 7th element (12 bytes in) is end bin
    recording_end = (int(bin_start) + int(bin_end)) * int(bin_size) - .001
    recording_start = movement_data[0, 0]

    # darren used to set last ME event time equal to end of recording time
    movement_data[-1, 0] = recording_end
    # Convert to unix timestamp
    movement_data[:, 0] += 0
    # TICKET from original code: probably wrong (from Darren)
    data['recording_start_stop_time'] = np.array([recording_start, recording_end], dtype=np.double)

    data['_movement_data'] = movement_data
    data['_delta_x'] = delta_x
    data['_delta_y'] = delta_y
    # Get the uncorrected position data
    data['uncorrected_t'] = movement_data[:, 0]
    data['uncorrected_x'] = movement_data[:, 1]
    data['uncorrected_y'] = movement_data[:, 2]
    # in the absence of a better option, we point to the uncorrected stuff
    data['t'] = movement_data[:, 0]
    data['x'] = movement_data[:, 1]
    data['y'] = movement_data[:, 2]
    return data


def load_nonlocomotor_event_data(filename):
    try:
        # Convert time to seconds since midnight
        data = np.loadtxt(filename, delimiter=",", ndmin=2) / 1000
    except ValueError:
        remove_bad_line(filename)
        data = np.loadtxt(filename, delimiter=",", ndmin=2) / 1000

    filetype = filename[-3:-1]
    name = filetype
    if filetype == 'PE':
        name = 'feeding'
    elif filetype == 'LE':
        name = 'drinking'
    if data.shape[0] < 2:
        logger.error("No {} events on this day! see {} to confirm.".format(name, filename))

    # Convert to cumulative times
    data[:, 0] = np.cumsum(data[:, 0])
    data[:, 0] += 0  # (starttime - datetime.datetime(1970, 1, 1)).total_seconds()
    # first one is just the start of recording, so chuck it
    res = data[1:, :]
    return res


def raw_filenames_strainsurvey(exp_name, mouseNumber, day, round_number):
    # search event files first
    path = os.path.join(datadir(exp_name), 'EventFiles/EventFiles_SSe1r%d' % round_number)
    regex_path = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]" + "e1r%dd%01d" % (round_number, day)
    subdir = list()
    for dirname in os.listdir(path):
        if re.search(regex_path, dirname) is not None:
            subdir.append(dirname)
    assert len(subdir) == 1
    subdir, = subdir
    searchpath = os.path.join(path, subdir)
    extensions = ["AM", "LE", "ME", "PE"]
    result = [None, None, None, None]
    regex = "[0-9][0-9][0-9][0-9]%04d\.%%s" % int(mouseNumber)
    for filename in os.listdir(searchpath):
        for i, ext in enumerate(extensions):
            if re.search(regex % ext, filename) is not None:
                result[i] = os.path.join(searchpath, filename)

    # search am file
    assert result[0] is None  # no AM file found so far
    path = os.path.join(datadir(exp_name), 'AMFiles/AMFiles_SSe1r%d' % round_number)
    searchpath = os.path.join(path, subdir)
    for filename in os.listdir(searchpath):
        if re.search(regex % "AM", filename) is not None:
            result[0] = os.path.join(searchpath, filename)

    return result


def raw_filenames_2cd1(exp_name, mouseNumber, day):
    path = os.path.join(datadir(exp_name), '{}_DayFiles/'.format(exp_name))
    regex_path = "^Day{:01d}".format(day)
    subdir = list()
    for dirname in os.listdir(path):
        if re.search(regex_path, dirname) is not None:
            subdir.append(dirname)
    # assert len(subdir) == 1
    try:
        assert len(subdir) == 1
        subdir, = subdir
    except AssertionError:
        subdir = subdir[0]  # workaround: use only first directory

    searchpath = os.path.join(path, subdir)
    extensions = ["AM", "LE", "ME", "PE"]
    result = [None, None, None, None]
    regex = "[0-9][0-9][0-9][0-9]%04d\.%%s" % int(mouseNumber)
    for filename in os.listdir(searchpath):
        for i, ext in enumerate(extensions):
            if re.search(regex % ext, filename) is not None:
                result[i] = os.path.join(searchpath, filename)

    # # search am file
    assert result[0] is not None  # AM file found
    return result


def raw_filenames(exp_name, mouseNumber, day, round_number=None):
    """AM, PE, LE, ME filenames"""
    result = None
    if exp_name in ['StrainSurvey', '2cD1A2aCRE', '2cD1A2aCRE2']:
        if exp_name == 'StrainSurvey':
            result = raw_filenames_strainsurvey(exp_name, mouseNumber, day, round_number)
        elif exp_name.startswith('2cD1A2aCRE'):
            result = raw_filenames_2cd1(exp_name, mouseNumber, day)
    else:
        extensions = ["AM", "LE", "ME", "PE"]
        result = [None, None, None, None]
        regex = "[0-9][0-9][0-9][0-9]%04d\.%%s" % int(mouseNumber)
        path = None
        if exp_name == '2CFast':
            path = os.path.join(datadir(exp_name), '2CFast_DayFiles/2CFAST_HCMe1r1_D{:02d}/'.format(day))
        elif exp_name == '1ASTRESS':
            path = os.path.join(datadir(exp_name), '1ASTRESS_DayFiles/Day{:01d}/'.format(day))
        elif exp_name == 'Stress_HCMe1r1':
            path = os.path.join(datadir(exp_name), 'Stress_DayFiles/StressHCMe1r1d{:01d}/'.format(day))
        elif exp_name == 'CORTTREAT':
            path = os.path.join(datadir(exp_name), 'CORTTREAT_DayFiles/CORTTREAT_HCMe1r1_d{:02d}/'.format(day))
        elif exp_name == 'HiFat2':
            path = os.path.join(datadir(exp_name), 'HFD2_DayFiles/HFD2_HCMe2r1_d{:01d}/'.format(day))
        elif exp_name == 'HiFat1':
            path = os.path.join(datadir(exp_name),
                                'Round%d/HFD_DayFiles/HFDe1r%dd%01d/' % (round_number, round_number, day))
        elif exp_name.startswith('WR'):
            path = os.path.join(datadir(exp_name), '%s_DayFiles/Day%01d/' % (exp_name, day))

        for filename in os.listdir(path):
            for i, ext in enumerate(extensions):
                if re.search(regex % ext, filename) is not None:
                    result[i] = os.path.join(path, filename)
    return result


def from_raw(msi_dict):
    """loads photobeam, lickometer and movement from PE, LE, ME and MSI file"""
    am_filename, le_filename, me_filename, pe_filename = raw_filenames(exp_name=msi_dict['exp_name'],
                                                                       mouseNumber=msi_dict['mouseNumber'],
                                                                       day=msi_dict['day_number'],
                                                                       round_number=msi_dict['round_number'])
    data = dict()
    data['_am_filename'] = am_filename
    data['_le_filename'] = le_filename
    data['_me_filename'] = me_filename
    data['_pe_filename'] = pe_filename

    if am_filename is not None:
        data['_light_data'] = load_am_data(am_filename)
        data['_photobeam_data'] = load_nonlocomotor_event_data(pe_filename)  # if pe_filename else np.zeros((1, 2))
        data['_lickometer_data'] = load_nonlocomotor_event_data(le_filename)  # if le_filename else np.zeros((1, 2))
        load_movement_data(data, me_filename)

        # add other mouseday attributes
        data['_nest_loc_x'] = msi_dict['nest_loc_x']
        data['_nest_loc_y'] = msi_dict['nest_loc_y']
        data['start_bodyweight_grams'] = msi_dict['start_bodyweight_grams']
        data['end_bodyweight_grams'] = msi_dict['end_bodyweight_grams']
        data['bw_gain_grams'] = msi_dict['end_bodyweight_grams'] - msi_dict['start_bodyweight_grams']
        data['food_consumed_grams'] = msi_dict['chow_in_grams'] - msi_dict['chow_out_grams']
        data['liquid_consumed_grams'] = msi_dict['liquid_in_g'] - msi_dict['liquid_out_g']
    return data


def check_consistency(data):
    data_ = data['preprocessing']
    if data_['_am_filename'] is None:
        raise ValueError("missing AM file data record")

    start_time, end_time = [data_['t'][x] for x in [0, -1]]
    photo_data = data_['_photobeam_data']
    lick_data = data_['_lickometer_data']
    move_data = data_['_movement_data']
    if len(photo_data):
        if photo_data[0, 0] < start_time:
            raise ValueError("Error: firstFeedTime earlier than recordingStartTime")
        if photo_data[-1, 0] + photo_data[-1, 1] > end_time:
            raise ValueError("Error: lastFeedTime past recordingEndTime")
    else:
        raise IndexError

    if len(lick_data):
        if lick_data[0, 0] < start_time:
            raise ValueError("Error: firstLickTime earlier than recordingStartTime")  # %s" % self)
        if lick_data[-1, 0] + lick_data[-1, 1] > end_time:
            raise ValueError("Error: lastLickTime past recordingEndTime")  # %s" % self)

    else:
        raise IndexError

    if not len(move_data):
        raise IndexError
