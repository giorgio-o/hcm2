# hcm2/core/model/experiment.py
"""
G. Onnis, 11.2017
revised: 11.2018

HCM Experiment Class.

Copyright (c) 2018. All rights reserved.

Tecott Lab UCSF

"""
import numpy as np
import logging
import inspect
import os
import json

from ..helpers import read_MSI_file
from .. import fixer
from util.utils import timing
from util.file_utils import repo_dir, find_files, mouseday_label_from_filename
from util.df_utils import set_df_indices

logger = logging.getLogger()


class Experiment(object):
    """HCM experiment class """
    hyper_params = dict(IST=20,  # inactive state threshold [min]
                        FBT=30,  # feed bout threshold [sec]
                        DBT=30,  # drink bout threshold [sec]
                        LBVT=1,  # move bout velocity threshold [cm/s]
                        LBDT=5,  # move bout distance threshold [cm]
                        BTT=0.2  # connecting gaps of less than this for all bouts [sec]
                        )

    def __init__(self, name=None, hyper=None):
        print "initializing {} experiment..".format(name)
        self.name = name
        self.hyper_params = hyper or self.hyper_params
        # MSI files
        self.msi_dict = read_MSI_file.read_msi_files(name, self.path_to_binary())
        self.days = self.msi_dict.values()[0].values()[0].keys()  # all experiment days, from day 1
        self.data = dict(position=dict(), features=dict(), events=dict(), breakfast=dict(), time_budget=dict())

    def __str__(self):
        return "Experiment: {}".format(self.name)

    @property
    def groups_ordered(self):
        """Returns list of (ordered) group names """
        from core.keys import groups_ordered
        return groups_ordered[self.name]

    @property
    def groups(self):
        """Returns group objects generator for groups (from MSI file) """
        from group import Group
        return (Group(experiment=self, number=number + 1, name=name) for number, name in enumerate(self.groups_ordered))

    @property
    def mice(self):
        """Returns mouse objects generator for mice (from MSI file) """
        return (m for g in self.groups for m in g.mice)

    def all_mousedays(self, days=()):
        """Returns mouseday objects generator for all mousedays (from MSI file) """
        days = days or self.days
        return (md for group in self.groups for mouse in group.mice for md in mouse.all_mousedays(days))

    def mousedays(self, days=()):
        """Returns mouseday objects generator for mousedays for which preprocessed data is available """
        days = days or self.days
        path = self.path_to_binary(subdir=os.path.join("preprocessing", "AS_timeset"))
        path_labels = [l for l in self.mouseday_labels_from_binary_path(path) if l[2] in days]
        return (md for md in self.all_mousedays(days) if md.label in path_labels)

    def group_object(self, name):
        """Returns a mouse given its name """
        return (g for g in self.groups if g.name == name).next()

    def mouse_object(self, label):
        """Returns a mouse given its name (e.g., (group_name, mouse_number)"""
        return (m for m in self.mice if m.name == label[1]).next()

    def mouseday_object(self, label):
        """Returns a mouse given its name (e.g., (group_name, mouse_number, day)) """
        return (md for md in self.all_mousedays() if label[1] == md.mouse.name and label[2] == md.day).next()

    @property
    def ignored(self):
        """Returns dictionary with ignored mice and mousedays """
        from core.ignore import ignored
        return ignored[self.name]

    @property
    def valid_mice(self):
        """Returns list of tuples (group_name, mouse_number) """
        return [m.label for m in self.mice if not m.is_ignored]

    @property
    def ignored_mice(self):
        """Returns list of tuples (group_name, mouse_number) """
        return self.ignored["mice"]

    def valid_mousedays(self, days=()):
        """Returns list of tuples (group_name, mouse_number, day) for selected days """
        return [md.label for md in self.all_mousedays(days) if not md.is_ignored]

    def ignored_mousedays(self, days=()):
        """Returns list of tuples (group_name, mouse_number, day) for selected days """
        return [md.label for md in self.all_mousedays(days) if md.is_ignored]

    @property
    def group_mice_dict(self):
        """Returns dictionary {group_name: list of mouse numbers} """
        return {name: sorted(self.msi_dict[name].keys()) for name in self.msi_dict.keys()}  # ints

    def groups_numbered(self, reverse=False):
        """Returns dictionary {group_number: group_name} """
        d = {n: v for n, v in zip(range(1, len(self.groups_ordered) + 1), self.groups_ordered)}
        return d if not reverse else dict((v, k) for k, v in d.iteritems())

    def summary(self, obs_period="acclimated"):
        """Returns information summary for experiment for selected observation period (obs_period) """
        from core.keys import obs_period_to_days
        days = obs_period_to_days[self.name][obs_period]
        print "Experiment {} Summary".format(self.name)
        print "days: {}".format(self.days)
        print "experiment day periods:"
        for l, d in obs_period_to_days[self.name].iteritems():
            print "- \"{}\": {}".format(l, d)
        print "mice: {}, valid: {}, ignored: {}".format(len(self.valid_mice) + len(self.ignored_mice),
                                                        len(self.valid_mice), len(self.ignored_mice))
        print "{} mousedays: {}, valid: {} ignored: {}".format(obs_period, len(self.valid_mousedays(days)) + len(
            self.ignored_mousedays(days)), len(self.valid_mousedays(days)), len(self.ignored_mousedays(days)))

        for g in self.groups:
            print "--" * 20
            print "group{}: {}, mice: {}".format(g.number, g.name, len(g.mouse_numbers))
            print "valid: {}, {}".format(len(g.valid_mice), [m[1] for m in g.valid_mice])
            print "ignored: {}, {}".format(len(g.ignored_mice), g.ignored_mice)
            valid, ignored = g.valid_mousedays(days), g.ignored_mousedays(days)
            print "valid mousedays: {}".format(len(valid))
            print "ignored mousedays: {}, {}".format(len(ignored), ignored)

    def all_ignored_mousedays(self, days=None):
        """Returns list of tuples (group_name, mouse_number, day) for selected days """
        days = days or self.days
        mds = [md for md in self.ignored["mousedays"] if md[2] in days]
        return mds + [(g, m, d) for g, m in self.ignored["mice"] for d in days]

    def group_number(self, name):
        """Returns group_number given group_name """
        from core.keys import groups_ordered
        return groups_ordered[self.name].index(name) + 1

    # data preprocessing methods
    def path_to_binary(self, subdir=''):
        """Returns path for npy files (files generated by preprocessing raw data) in: /binary/<exp_name>/<subdir>/
        """
        path = os.path.join(repo_dir(), 'binary', self.name, subdir)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def get_preprocessing_data_keys(self, subdir, EXCLUDE=True):
        """Returns list of variables names found in /binary/<exp_name>/<subdir> """
        path = self.path_to_binary(subdir)
        keys = os.walk(path).next()[1]
        if EXCLUDE:
            exclude = ["uncorrected_t", "uncorrected_x", "uncorrected_y", "delta_t", "F_timeset_position_errors",
                       "D_timeset_position_errors"]
            keys = [a for a in keys if a not in exclude]
        return keys

    @timing
    def process_raw_data(self, days=(), fixers=()):
        """Raw data preprocessing. Creates .npy files """
        import time
        cstart = time.clock()
        fixers = get_fixers() or fixers
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.process_raw_data(days, fixers)

        logger.info("\n")
        logger.info("Pre-processing raw data took {} minutes".format((time.clock() - cstart) / 60))
        logger.info("Created active states and feeding, drinking and locomotion bouts")
        logger.info("Created binary data:")
        variables = self.get_preprocessing_data_keys(subdir="preprocessing", EXCLUDE=False)
        for var in variables:
            logger.info(" -{}".format(var))

    @timing
    def create_position(self, days=(), bin_type=None, xbins=2, ybins=4):
        """Creates json position data files """
        logger.info(str(self))
        logger.info("Creating position density ..")
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.create_position(days, bin_type, xbins, ybins)
        fname = "{}_position_{}_xbins{}_ybins{}_bin_times.json".format(self.name, bin_type, xbins, ybins)
        self.save_json_data(akind='position', fname=fname)

    @timing
    def create_features(self, days=(), bin_type='12bins'):
        """Creates json feature data files, including:
            - Total amounts: Food, Drink, Movement
            - Active states: Probability, Numbers, Durations,
            - Active states' intensities: Food, Drink, Movement
            - Bouts: Bout Active State Rates, Numbers, Size, Durations, Intensities (Food, Drink, Locomotion)
        """
        logger.info(str(self))
        logger.info("Creating {} features ..".format(bin_type))
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.create_features(days, bin_type)
        fname = "{}_features_{}.json".format(self.name, bin_type)
        self.save_json_data(akind="features", fname=fname)

    @timing
    def create_breakfast(self, days=(), timepoint='onset', tbin_size=30, num_secs=900):
        """Creates json breakfast data files """
        logger.info(str(self))
        logger.info("Creating breakfast {} data (tbinsize={}s) ..".format(timepoint, tbin_size))
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.create_breakfast(days, timepoint, tbin_size, num_secs)
        fname = "{}_breakfast_{}_binary_counts_tbinsize{}s.json".format(self.name, timepoint, tbin_size)
        self.save_json_data(akind="breakfast", fname=fname)

    @timing
    def create_within_as_structure(self, days):
        """Creates within active states structure data """
        logger.info(str(self))
        logger.info("Creating within-active state structure data ..")
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.create_within_as_structure(days)

    @timing
    def create_time_budget(self, days, bin_type):
        """Creates json time budget data files """
        logger.info(str(self))
        logger.info("Creating time budget data ..")
        for group in self.groups:
            # for group in list(self.groups)[:1]:
            group.create_time_budget(days, bin_type)
        fname = "{}_time_budget_{}.json".format(self.name, bin_type)
        self.save_json_data(akind="time_budget", fname=fname)

    # load/ save data methods
    @staticmethod
    def mouseday_labels_from_binary_path(path, ext='npy'):
        """Returns a generator over mouseday labels (group_name, mouse_number, day) given filenames in path """
        return (mouseday_label_from_filename(f, ext) for f in find_files(path, ext))
        # return sorted([mouseday_label_from_filename(f, ext) for f in find_files(path, ext)], key=itemgetter(1,2,3))

    def mouseday_labels_tuples(self, days=(), subakind=None, sub_index1=None, sub_index2=None, short=True):
        """Returns list of (group, mouse, day, sub_index) used to build multi-index pandas dataframes """
        from core.keys import all_features, features_short, cycle_timebins, act_to_actlabel
        days = days or self.days
        ev_types = ['F', 'D']
        res = None
        if sub_index1 is None:
            res = [md.label for md in self.mousedays(days)]
        elif sub_index1 == 'as_num':
            if sub_index2 is None:
                res = [md.label + ('AS{}'.format(n),) for md in self.mousedays(days) for n in
                       range(1, md.active_states_counts + 1)]
            elif sub_index2 == 'event':
                res = [md.label + ('AS{}'.format(n), act_to_actlabel[ev_type]) for md in self.mousedays(days) for n in
                       range(1, md.active_states_counts + 1) for ev_type in ev_types]
        elif sub_index1 == 'event':
            res = [md.label + (act_to_actlabel[ev_type],) for md in self.mousedays(days) for ev_type in ev_types]

        elif sub_index1 == 'timebin':
            keys = cycle_timebins[subakind]
            res = [md.label + (n,) for md in self.mousedays(days) for n in keys]
        elif sub_index1 == 'feature':
            features = features_short if short else all_features
            res = [md.label + (n,) for md in self.mousedays(days) for n in features]
        return res

    def path_to_results(self, subdir=''):
        """Returns path for analysis results and visualizations: /results/<exp_name>/<subdir>/ """
        path = os.path.join(repo_dir(), 'results', self.name, subdir)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def save_json_data(self, akind, fname):
        """Saves experiment data to json file in: /binary/<exp_name>/advanced/ """
        import pandas as pd
        import json
        df = pd.concat([v for v in self.data[akind].values()])
        # stop
        path = self.path_to_binary(subdir="advanced")
        filename = os.path.join(path, fname)
        with open(filename, 'w') as fp:
            json.dump(df.to_json(orient='split'), fp)  # , indent=4, sort_keys=True)
        print "{} data saved to:\n{}".format(akind, filename)

    def load_homebase_data(self, days):
        """Loads homebase data from: /binary/<exp_name>/preprocessing/ 
            Returns DataFrame object
        """
        import pandas as pd
        from core.keys import homebase_1dto_string as to_string
        print "loading homebase data.."
        keys = ['obs_hb', 'rect_hb']
        ordered_tups = self.mouseday_labels_tuples(days)
        # path = path_to_binary(self.name, subdir=os.path.join('preprocessing', keys[0]))
        md_list = list()
        cnt = 0
        for md in self.mousedays(days):
            md.load_npy_data(keys=keys)
            rect, obs = [tuple(md.data['preprocessing'][x]) for x in ['rect_hb', 'obs_hb']]
            # percent agreement
            common = set(rect).intersection(set(obs))
            if common:
                cnt += 1
            # homebase location
            values = (rect, to_string[rect], obs, to_string[obs], np.bool(len(rect) - 1), np.bool(common))
            md_list.append(md.label + values)

        print "observed vs. detected homebase agreement: {}/{}".format(cnt, len(md_list))
        columns = ['group', 'mouse', 'day', 'detected', 'd_location', 'observed', 'o_location', 'domino', 'do_agree']
        df = pd.DataFrame(md_list, columns=columns)
        df = set_df_indices(df, ordered_tups)
        return df

    @timing
    def load_json_data(self, days, akind, fname, subakind=None, sub_index1=None, sub_index2=None, occupancy=None,
                       hb=None, PROB=None, ignore=False):
        """Load json data from: /binary/<exp_name>/advanced/. 
            Returns DataFrame object
        """
        import pandas as pd
        ignored_mds = self.all_ignored_mousedays(days)
        ordered_tups = self.mouseday_labels_tuples(days, subakind, sub_index1, sub_index2)
        # load
        path = self.path_to_binary(subdir='advanced')
        filename = os.path.join(path, fname)
        print "loading {} data from:\n{}\ndays: {}".format(akind, filename, days)
        with open(filename, 'r') as fp:
            df = pd.read_json(json.load(fp), orient='split')
        #     # read with: with open(filename, 'r') as fp: df2 = pd.read_json(json.load(fp), orient='split'))
        #     # alternatively: with open(filename, 'w') as fp: json.dump(data.to_dict(), fp)
        #     # read with:  with open(filename, 'r') as fp: df2 = pd.DataFrame.from_dict(json.load(fp))

        df = df[df['day'].isin(days)]
        df = set_df_indices(df, ordered_tups, sub_index1, sub_index2)
        df = df if not ignore else df.drop(ignored_mds)
        if akind == 'position' and occupancy:  # compute position percent occupancy
            df = df.apply(lambda x: x / x.sum(), axis=1) \
                .reindex(self.mouseday_labels_tuples(days, subakind=subakind, sub_index1='timebin'))
            if hb:  # homebase data
                from util.df_utils import merge_position_homebase_dataframes
                df_hb = self.load_homebase_data(days)
                df = merge_position_homebase_dataframes(df, df_hb)

        elif akind == 'breakfast' and PROB:
            # compute probability. takes a bit of time
            df = df.groupby(['group', 'mouse', 'day', 'event']).agg(lambda x: x.sum() / float(len(x)))
            df = df.reindex(self.mouseday_labels_tuples(days, sub_index1='event'))
        return df


def get_fixers():
    """Returns list of fixer classes: backwardfix/devicefix"""
    fixers = list()
    for fixname in dir(fixer):
        fix = getattr(fixer, fixname)
        if inspect.isclass(fix):
            fixers.append(fix)

    return fixers


# # kind of old
# def day_act_to_actlabel(self, day):
#     from core.keys import obs_period_to_days
#     return (k for k, vals in obs_period_to_days[self.name].iteritems() if day in vals and k != 'acclimated').next()
#
# @property
# def mousedays_bout_event_errors_list(self):
#     from core.keys import md_bout_events_remove_errors_dict
#     md_list = md_bout_events_remove_errors_dict[self.name]
#     return list(set([v for val in md_list.values() for v in val]))
#
# def mice_bout_event_errors_list(self, behav):
#     from core.keys import md_bout_events_remove_errors_dict, act_to_actlabel
#     mice_list = md_bout_events_remove_errors_dict[self.name][act_to_actlabel[behav]]
#     return [(s[0], s[1]) for s in mice_list] if len(mice_list) else list()