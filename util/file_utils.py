# hcm/util/file_utils.py
"""File utilities. """
import os
import sys
import logging
import pandas as pd

logger = logging.getLogger(__name__)
pd.set_option('display.width', 1000)


def days_suffix(obs_period):
    return obs_period.replace('-', '_').replace(' ', '_')


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress
    #  #  -bar-in-the-console/27871113#comment50529068_27871113)


def dataroot():
    return "/data/HCM/"


def datadir(exp_name):  # day=None):
    """ TODO: Move this to the data field. """
    daydir = ""
    other_exp = ["2cD1A2aCRE", "2cD1A2aCRE2", "2CFast", "1ASTRESS", "Stress_HCMe1r1", "CORTTREAT", "HiFat2", "HiFat1"]
    if exp_name in other_exp:
        datadir_parent = ""
        data_dir = exp_name
    # if day is not None: daydir = "D%d" % (day + 1)
    # elif exp_name.startswith("SS_Data_051905_FV"):
    #     datadir, expround = self.name.split(":")
    #     datadir_parent = "EventFiles/EventFiles_SSe1r%s" % expround
    #     date = (self.start + datetime.timedelta(days=day)).strftime("%m%d%Y")
    #     daydir = "%se1r%sd%s" % (date, expround, day + 1)
    elif exp_name == "StrainSurvey":
        datadir_parent = ""
        data_dir = "SS_Data_051905_FV"
    elif exp_name.startswith("WR"):
        datadir_parent = "WR"
        data_dir = exp_name
    else:
        raise ValueError("Unknown experiment: {}".format(exp_name))
    return os.path.join(dataroot(), "Experiments", datadir_parent, data_dir, daydir)


def hcm_dir():
    datadir_parent, data_dir = None, None
    if os.uname()[1] == "giorgios-MacBook-Pro.local":
        datadir_parent = "/Users/go"
        data_dir = "Projects/HCM/"
    return os.path.join(datadir_parent, data_dir)


def repo_dir():
    return os.path.join(hcm_dir(), "hcm2")


def find_files(path, ext='npy'):
    """ returns a generator over filenames in path """
    return (os.path.join(dirpath, f) for dirpath, _, files in os.walk(path) for f in sorted(files) if
            f.endswith('.{}'.format(ext)))


def mouseday_label_from_filename(fname, ext="npy"):
    stripped = fname.strip('.{}'.format(ext)).split('/')
    exp_name, akind, qty, md_label = [stripped[x] for x in [6, 7, -2, -1]]
    one, group, two, mouse, day = md_label.split('_')
    return group, mouse, int(day[1:])

# # kinda old
# # # EVENTS
# def create_df_from_series_data(experiment, dfs, days, ignore=False):
#     df = pd.DataFrame(dfs).T
#     df = df.reset_index().rename(index=str, columns={'level_0': 'group', 'level_1': 'mouse', 'level_2': 'day'})
#     df = df[df['day'].isin(days)]
#     if ignore:
#         # ignored = list() or experiment.ignored
#         df = remove_ignored_from_dataframe(experiment, df)
#     df = set_df_indices(df, experiment, index=['group', 'mouse', 'day'])
#     return df
#

# def load_ingestion_events_dataframe(experiment, days=(), ev_type='F', ignore=False):
#     path_to_npy1 = path_to_binary(experiment, subdir=os.path.join('preprocessing', '{}_timeset'.format(ev_type)))
#     print "loading {} event data from npy:\n{}".format(ev_type, path_to_npy1)
#     tot = len(list(find_files(path_to_npy1, ext='npy')))
#     dfs1, dfs2 = dict(), dict()
#     # durations
#     for cnt, fname in enumerate(find_files(path_to_npy1, ext='npy')):
#         vals = np.load(fname)
#         if len(vals):
#             dur, = np.diff(vals).T
#             interdur = vals[1:, 0] - vals[:-1, 1]
#         else:
#             dur, interdur = list(), list()
#
#         index = mouseday_label_from_filename(fname)
#         dfs1[index] = pd.Series(dur)
#         dfs2[index] = pd.Series(interdur)
#         progress(cnt, tot)
#
#     df1 = create_df_from_series_data(experiment, dfs1, days, ignore)
#     df2 = create_df_from_series_data(experiment, dfs2, days, ignore)
#
#     # Feeding and Licking Coeff
#     coeff_name = 'FC' if ev_type == 'F' else "LC"
#     path_to_npy2 = path_to_binary(experiment, subdir=os.path.join('preprocessing', coeff_name))
#     print "loading {} data from npy:\n{}".format(coeff_name, path_to_npy2)
#     tot = len(list(find_files(path_to_npy2, ext='npy')))
#     dfs3 = dict()
#     for cnt, fname in enumerate(find_files(path_to_npy2, ext='npy')):
#         vals = np.load(fname).tolist()
#         index = mouseday_label_from_filename(fname)
#         dfs3[index] = pd.Series(vals)
#         progress(cnt, tot)
#
#     df3 = create_df_from_series_data(experiment, dfs3, days, ignore) * 1000  # to mg/s
#     return df1, df2, df3
#
#
# @utils.timing
# def load_locomotion_events_dataframe(experiment, days=(), ignore=False):
#     keys = ['delta_t', 'idx_timestamps_out_hb', 'idx_timestamps_at_hb', 'velocity', 'distance']
#     print "loading timestamp data from npy:\n{}".format(keys)
#     dfs_list = [dict() for _ in range(9)]
#     tot = len(list(experiment.mousedays))
#     for cnt, md in enumerate(experiment.mousedays):
#         if md.day in days:
#             delta_t, idx_out, idx_at, vel, dist = [load_preprocessing_data(md, keys=[key])['preprocessing'][key]
#                                                    for key in keys]
#             dfs_list[0][md.label] = pd.Series(delta_t)
#             dfs_list[1][md.label] = pd.Series(vel)
#             dfs_list[2][md.label] = pd.Series(dist)
#             dfs_list[3][md.label] = pd.Series(delta_t[idx_out])
#             dfs_list[4][md.label] = pd.Series(vel[idx_out])
#             dfs_list[5][md.label] = pd.Series(dist[idx_out])  # todo: REWRITE
#             dfs_list[6][md.label] = pd.Series(delta_t[idx_at])
#             dfs_list[7][md.label] = pd.Series(vel[idx_at])
#             dfs_list[8][md.label] = pd.Series(dist[idx_at])
#             progress(cnt, tot)
#             cnt += 1
#
#     return [create_df_from_series_data(experiment, dfs, days, ignore) for dfs in dfs_list]
