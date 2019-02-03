# hcm/visualization/viz/features_vs_ct_csv_files.py
import os

from util import df_utils
from features_vs_ct_as_groups_avg import load_tidy_data


def write_to_csv(experiment, obs_period, bin_type, ignore):
    res_subdir = os.path.join('features', 'csv_files')
    days_label = obs_period.replace('-', '_').replace(' ', '_')
    # load
    _, df_all = load_tidy_data(experiment, obs_period, bin_type, ignore=ignore)

    # write
    fname = '{}_{}_features_vs_CT_{}_mousedays_{}_days'\
        .format(experiment.name, bin_type, '' if ignore else 'ALL', days_label)
    df_utils.save_dataframe_to_csv(experiment, df_all, res_subdir, fname)