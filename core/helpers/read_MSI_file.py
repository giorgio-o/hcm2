# hcm/core/helpers/read_MSI_files.py
""" reads HCM MSI files data """

import os
import csv

from util import file_utils


MSI_columns = (("round_number", int),  # mouse
               ("run", int),  # ?
               ("daq_system", int),  # mouse
               ("enclosure", int),  # mouse
               ("day_number", int),  # mouseday
               ("file_date", int),  # mouseday
               ("mouse_number", int),  # mouse
               ("msi_group_number", int),  # no need (group)
               ("dob", int),  # mouse
               ("start_age_days", int),  # mouse
               ("start_bodyweight_grams", int),  # mouse
               ("end_bodyweight_grams", int),  # mouse
               ("lenght_cm", int),  # mouse
               ("chow_in_grams", int),  # mouseday
               ("chow_out_grams", int),  # mouseday
               ("chow_type", int),  # mouseday
               ("liquid_in_g", int),  # mouseday
               ("liquid_out_g", int),  # mouseday
               ("liquid_type", int),  # mouseday
               ("nest_loc_x", int),  # mouseday
               ("nest_loc_y", int),  # mouseday
               ("chow_com", str),  # ?
               ("liq_com", str),  # ?
               ("pe_com", str),  # ?
               ("le_com", str),  # ?
               ("me_com", str),  # ?
               ("nest_com", str),  # ?
               ("ani_com", str),  # ?
               ("sys_com", str))  # ?


def check_groups_for_names(res, path):
    if 'HiFat2' in path:
        res['WT'] = res.pop('WTHF')
        res['2CKO'] = res.pop('2CHF')
    return res


def find_msi_files(path):
    """ Returns a generator over msi files in the path. """
    return (os.path.join(dirpath, f) for dirpath, _, files in os.walk(path) for f in files if f.endswith("MSI_FF.csv"))


def find_group_file(path):
    """ Returns the path of the group mapping file. """
    return (os.path.join(dirpath, f) for dirpath, _, files in os.walk(path) for f in files if
            f.endswith("GrpCodeNames.txt")).next()


def parse_group_file(path):
    """ Returns a dictionary of group number to group name. """
    groups = {}
    # import sys
    # sys.stdout.write(path)
    with open(path) as fp:
        for line in fp.readlines()[1:]:
            tokens = line.strip().split("\t")
            num, name = tokens
            num = int(num.strip())
            groups[num] = name.strip()
    return groups


def parse_msi_row(row):
    """ Returns an object with keys corresponding to select MSI columns. """
    row_data = {}
    for (field, field_type), value in zip(MSI_columns, row):
        try:
            row_data[field] = int(value)
        except ValueError:
            row_data[field] = float(value)
    return row_data


def parse_msi_file(msifile):
    """ Parse an msi csv file and return a dictionary for each row. """
    msi = csv.reader(msifile, delimiter=",")
    msi.next()  # Skip header row
    return (parse_msi_row(row) for row in msi)


def accumulate_results_(rd, groups, res):
    """ Deprecate. """
    rd["group_name"] = groups[int(rd["msi_group_number"])]
    mouse_data = res.setdefault(rd["group_name"], {}).setdefault(rd["mouse_number"], {})
    if rd["day_number"] in mouse_data:
        raise ValueError("Group, mouse, and day triplet occurs on multiple lines from the "
                         "various MSIFiles: ({}, {}, {})".format(rd["msi_group_number"], rd["mouse_number"],
                                                                 rd["day_number"]))
    mouse_data.setdefault(rd["day_number"], rd)


def read_msi_files(exp_name, savepath):
    """ Returns MSI data parsed from the given experiment, saves as cPickle object """
    import cPickle as pickle
    filename = os.path.join(savepath, "{}_msi_data.p".format(exp_name))
    try:
        with open(filename, 'rb') as fp:
            res = pickle.load(fp)
    except IOError:
        path = file_utils.datadir(exp_name)
        msi_filenames = find_msi_files(path)
        print "reading MSI file/s:"
        group_file = find_group_file(path)
        groups = parse_group_file(group_file)
        res = {}
        for fname in msi_filenames:
            print fname
            with open(fname) as msifile:
                for rd in parse_msi_file(msifile):
                    accumulate_results_(rd, groups, res)
        res = check_groups_for_names(res, path)
        with open(filename, 'wb') as fp:
            pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print "msi dict file saved to:\n{}".format(filename)
    return res
