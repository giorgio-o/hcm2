# hcm/visualization/viz/quality_control.py
import numpy as np
import os

from util import utils, file_utils, df_utils
from core.helpers.qc import raw_errors
import raster_distance_mouseday, position_density



def rasters_raw_errors(experiment, days, CSV=True):
    """ plot rasters for mousedays flagged in raw_errors"""
    df_raw, mds_list = raw_errors.raw_errors(experiment, days)
    print "plotting {} mouseday raw error instances..\nmousedays: {}".format(len(mds_list), mds_list)
    for mouse_name, day, code, window in mds_list:
        res_subdir = os.path.join('quality_control', 'raw_errors', 'code{}'.format(code))
        if code in [1, 3, 5]:
            # entire day
            kwargs = dict(df=df_raw, mouse_name=mouse_name, day=day, subtype=0, res_subdir=res_subdir, f_suffix='_qc')
            raster_distance_mouseday.draw_mouseday(experiment, **kwargs)

        # elif code in [2, 4]:  # go zoomin based on window
        #     # zoom in
        #     for tstart, dt in window:
        #         kwargs = dict(df=df, mouse_name=mouse_name, day=day, subtype=2, tstart=tstart, dt=dt, tstep=dt / 20,
        #                       res_subdir=res_subdir, f_suffix='test2')
        #         draw_mouseday(experiment, **kwargs)

    if CSV:  # save
        df, mds_list = raw_errors.raw_errors(experiment, days, remove_code_6=False)
        subdir = os.path.join('quality_control', 'raw_errors', 'csv_files')
        for name, dfg in df.groupby('error_code'):
            filename = '{}_raw_errors_code{}'.format(experiment.name, name)
            df_utils.save_dataframe_to_csv(experiment, dfg[dfg.columns[:-1]], subdir, filename)


# def rasters_selected_zoom_in(experiment):
#     """ rework """
#     res_subdir = os.path.join('quality_control', 'rasters', 'selected')
#     for mouse_name, day in mds_list:
#         kwargs = dict(mouse_name=mouse_name, day=day, subtype=0,  # 0: 'entire_day', 1: active states, 2:manual_zoomin
#                       as_num=1,  # None or 1, 2, 3....
#                       tstart=9 + 18 / 60. + 0 / 3600.,  # time window start: circadian hours and decimals
#                       dt=7 / 60. + 0 / 3600.,  # tstart+dt
#                       tstep=30 / 3600.,  # timestep for xticks
#                       res_subdir=res_subdir, f_suffix='test')
#         raster_distance_mouseday.draw_mouseday(experiment, **kwargs)


def all_rasters_mouseday(experiment, obs_period):
    from core.keys import obs_period_to_days
    days = obs_period_to_days[experiment.name][obs_period]
    res_subdir = os.path.join('quality_control', 'raster_all_mousedays')
    raster_distance_mouseday.plot_all_mds(experiment, days, res_subdir)


def position_platform_drift(experiment, days=(), key_cycle='24H'):
    """ plot 24H position density group multiples individual vs days """
    import pandas as pd
    days = days or experiment.days
    res_subdir = os.path.join('quality_control', 'platform_drift')
    for (xbins, ybins) in [(2, 4), (12, 24)]:
        df_all = file_utils.load_position(experiment, days, bin_type='7cycles', xbins=xbins, ybins=ybins)
        df24 = df_utils.select_cycle(df_all, key_cycle)
        ## todo :fix this
        stop
        tidy_df = pd.melt(df24.reset_index(), id_vars=['group', 'mouse', 'day', 'detected', 'observed'],
                          var_name='xy_bin', value_name='value')
        position_density.draw_group_multiples_indv_vs_days(tidy_df, experiment, days, xbins, ybins, bin_type='7cycles',
                                                           res_subdir=res_subdir, key_cycle=key_cycle)


def features_outliers(experiment, days=()):
    """ plot features for mousedays flagged in find_outliers todo """
    behaviors = ['F', 'D', 'L']
    mice_list = experiment.selected_mice_list
    # mice_list = experiment.mice_bout_event_errors_list(behav)
    features_across_CT.plot(experiment, days, bin_type, htype, err_type, behaviors, mice_list, ignore=False)


def find_feature_outliers(experiment, df_all, threshold=10):
    print "checking features outliers >{}sigma away ..".format(threshold)
    outs = list()
    dfs = list()
    for name, dfg in df_all.groupby(['group', 'mouse']):
        idx_out = np.abs(dfg - dfg.mean()) > threshold * dfg.std()
        df_out = dfg.iloc[np.where(idx_out)]
        df_out = df_out[~df_out.index.duplicated(keep='first')]
        if not df_out.empty:
            dfs.append(df_out)
            for index, row in df_out.iterrows():
                # behavs = set([utils.feature_to_behavior(f) for f in row[row.nonzero()[0]].index.tolist()])
                behavs = set(utils.feature_to_behavior(index[3]))
                outs.append(index + (list(behavs), )) # group mouse, day, bin_num, ['F', 'D', 'L']
    print "found {} outliers".format(len(outs))

    return dfs, outs


def feature_outliers(experiment, days=(), threshold=10, CSV=True):
    """ threshold [sigma] todo """
    from core.keys import features_short # all_features
    import pandas as pd
    days = days or experiment.days
    df_all = file_utils.load_features(experiment, days, bin_type='12bins', ignore=False)
    df_all = df_all.loc[pd.IndexSlice[:, :, :, features_short], :]

    df_list, outs = find_feature_outliers(experiment, df_all, threshold)

    if CSV:  # save
        subdir = os.path.join('quality_control', 'features', 'outliers', 'csv_files')
        for df in df_list:
            group, mouse = df.index.tolist()[0][:2]
            filename = '{}_features_outliers_group_{}_{}'.format(experiment.name, group, mouse)
            df_utils.save_dataframe_to_csv(experiment, df, subdir, filename)

    return df_list, outs