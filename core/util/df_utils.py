# hcm/util/file_utils.py
"""Dataframe utility file. """

import numpy as np
import pandas as pd


def save_dataframe_to_csv(experiment, df, subdir, filename):
    import os
    filename = os.path.join(experiment.path_to_results(subdir), filename) + '.csv'
    df.to_csv(filename)
    print "saved dataframe to csv:\n{}".format(filename)


def add_labels_to_df_mouseday(dd, md_label, column_keys=(), column_vals=(), index_name=None):
    dd.index.name = index_name
    keys = ('group', 'mouse', 'day') + column_keys
    vals = md_label + column_vals
    for key, vals in zip(keys, vals):
        dd[key] = vals
    return dd


def set_df_indices(df, ordered_tups, sub_index1=None, sub_index2=None):
    index_name = ['group', 'mouse', 'day']
    if sub_index1 is not None:
        if sub_index2 is None:
            index_name += [sub_index1]
        else:
            index_name += [sub_index1, sub_index2]
    df.set_index(index_name, inplace=True)
    return df.reindex(ordered_tups)


def merge_position_homebase_dataframes(df_all, df_hb):
    return pd.merge(df_all.reset_index(), df_hb[['detected', 'observed']].reset_index(), on=['group', 'mouse', 'day'],
                    how='inner').set_index(['group', 'mouse', 'day', 'timebin'])


def select_cycle(df_all, key_cycle):
    df_cycle = df_all.loc[pd.IndexSlice[:, :, :, key_cycle], :]
    df_cycle.index = df_cycle.index.droplevel('timebin')
    return df_cycle


def exclude_group_features(df, behav, groups=()):
    from core.keys import act_to_actlabel
    for group in groups:
        df.loc[group, feature_keys['L']] = np.nan
        print "WARNING: set {} {} data to nan".format(group, act_to_actlabel[behav])
    return df

# # # unused
# def average_position_over_mice(df_all):
#     """ not used """
#     dfs = list()
#     for (group_name, day), dfr in df_all.groupby(['group', 'day']):
#         index = pd.MultiIndex.from_product([[group_name], [day]], names=['group', 'day'])
#         data = [[dfr[col].mean()] for col in dfr.columns]
#         dfs.append(pd.DataFrame(data=data).T.set_index(index).rename(
#             columns={k: v for k, v in zip(range(len(dfr.columns)), dfr.columns)}))
#     return pd.concat(dfs)

# def drop_df_2d_nan_arrays(experiment, days, df):
#     """ drops nan values in a dataframe - used in position density.."""
#     idx = pd.IndexSlice
#     is_nan = np.array([np.array([np.isnan(arr).all() for arr in row]).all() for row in df[cycle_keys].values])
#     # check
#     ignored_labels = df.loc[idx[is_nan]].index.get_values().tolist()
#     ignored_mds, _ = experiment.ignored_lists(days)
#     assert set(ignored_labels) == set(ignored_mds), 'nan f-u'
#     return df.drop(ignored_labels)
#
#
#
# def collapse_df_columns_into_xbins_ybins_2d_arrays(dd, xbins, ybins):
#     """ pos dens """
#     cols = [col for col in dd.columns if col.isupper()]
#     values = [np.array(dd[col]).reshape(ybins, xbins) for col in cols]
#     data = {k:[v] for k,v in zip(cols, values)}
#     base_cols = [col for col in dd.columns if col.islower()]
#     base_values = [dd[col].unique()[0] for col in base_cols]
#     base_dict = {k: v for k, v in zip(base_cols, base_values)}
#     data.update(base_dict)
#     return pd.DataFrame(data)

# def flip_breakfast_df_columns_to_rows(dd):
#     """ no needed no more"""
#     # flip (index, columns) for (tbin_number, active_state_number)
#     as_cols = [col for col in dd.columns if len(col) < 3]  # active state numbers as string
#     vals = dd[as_cols].T.values
#     df1 = pd.DataFrame(vals, index=as_cols, columns=dd.index)
#     # convert index str to int and sort
#     df1.index = df1.index.map(int)
#     df1 = df1.reindex(range(1, len(df1) + 1))
#     # create df with labels
#     other_cols = [col for col in dd.columns if col not in as_cols]
#     labels = dd[other_cols].loc[1 : len(df1.index)].values
#     df2 = pd.DataFrame(labels, index=df1.index, columns=other_cols)
#     df = pd.concat([df1, df2], axis=1)
#     return df
