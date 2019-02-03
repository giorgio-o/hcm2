# hcm/callers/viz.py
""" HCM experiment data analysis and visualization """
from pandas import set_option
import argparse

from util import utils
from core.model.experiment import Experiment
from visualization.viz import position_density, rasters, raster_distance_mouseday
from visualization.viz import features_vs_ct_as_groups_avg, features_vs_ct_fdl_groups_avg
from visualization.viz import features_vs_ct_groups_day_breakdown
from visualization.viz import features_vs_ct_as_groups_avg_days_comparison
from visualization.viz import breakfast, breakfast_days_comparison
from visualization.viz import within_as_structure, within_as_structure_days_comparison
from visualization.viz import time_budget, time_budget_days_comparison


set_option('display.width', 300)


def get_parser():
    """ parse command line input """
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("akind")
    parser.add_argument("--obs_period", type=str, default="acclimated")
    parser.add_argument("--htype", type=str)
    parser.add_argument("--mouse_label", type=str)
    parser.add_argument("--bin_type", type=str)
    parser.add_argument("--xbins", type=int)
    parser.add_argument("--ybins", type=int)
    parser.add_argument("--timepoint", type=str)
    parser.add_argument("--err_type", type=str, default="sem")
    parser.add_argument("--ignore", type=bool, default=True)
    parser.add_argument("--csv_file", type=bool, default=False)
    parser.add_argument("--day_break", type=int, default=False)
    parser.add_argument("--write_days", type=int, default=True)
    parser.add_argument("--as_only", type=int, default=False)
    return parser


@utils.timing
def main():
    """ main analysis/visualization file.
        # python callers/viz.py <experiment_name> <analysis_kind>
        - akind = "raster":
            preprocess raw data
            saves to npy files
        - akind = "position":
            creates position data
            bin_type = ["7cycles", "4bins", "12bins", "24bins"]  # time bins
            (xbins, ybins) = [(2, 4), (12, 24)]  # cage discretization
            "7cycles": ["24H", "DC", "LC", "AS24H", "ASDC", "ASLC", "IS"]
            saves to json files
        - akind = "features":
            creates features data, bin_type=["3cycles", "4bins", "12bins", "24bins"]
            "3cycles": ["24H", "DC", "LC"]
            saves to json files
        - akind = "breakfast":
            creates breakfast hypothesis data
            timepoint=['onset', 'offset']  # from active state onset vs offset
            saves to json files
        - akind = "time_budget":
            creates time budget data
            bin_type = ["6cycles"]
            "6cycles": ["24H", "DC", "LC", "AS24H", "ASDC", "ASLC"]
            saves to json files
    """
    # analysis parameters
    parser = get_parser()
    args = parser.parse_args()
    name, akind = args.name, args.akind
    obs_period = args.obs_period
    err_type, ignore, csv_file = args.err_type, args.ignore, args.csv_files
    # initialize experiment
    experiment = Experiment(name)

    if akind == "raster":
        htype, write_days = args.htype, args.write_days
        if htype == "groups":
            rasters.group_rasters(experiment, obs_period, write_days=write_days, as_only=as_only)
        elif htype == "mice":
            if args.mouse_label is not None:
                rasters.mouse_raster(experiment, obs_period, mouse_label=args.mouse_label, write_days=write_days)
            else:
                rasters.all_mice_rasters(experiment, obs_period, write_days=write_days)
        elif htype == "mousedays":
            raster_distance_mouseday.plot_all_mds(experiment, obs_period)

    elif akind == "position":
        if csv_file:
            position_density.write_to_csv(experiment, obs_period, args.bin_type, ignore=ignore)
        else:
            position_density.position_density(experiment, obs_period, args.htype, args.bin_type, args.xbins, args.ybins,
                                              ignore)

    elif akind == "features":
        htype, bin_type = args.htype, args.bin_type
        day_break = args.day_break if args.day_break else False
        if htype == "groups":
            if csv_file:
                features_vs_ct.write_to_csv(experiment, obs_period, bin_type, ignore)
            else:
                features_vs_ct_as_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
                features_vs_ct_fdl_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
                if day_break:
                    features_vs_ct_groups_day_breakdown.facets_groups_day_breakdown(experiment, obs_period, bin_type,
                                                                                    err_type, ignore)

                if experiment.name in ["HiFat2"]:  # days comparison, only HiFat2
                    obs_period = "comparison"
                    features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_day(experiment,
                                                                                                  obs_period, bin_type,
                                                                                                  err_type, ignore)
                    features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_group(experiment,
                                                                                                    obs_period,
                                                                                                    bin_type, err_type,
                                                                                                    ignore)

    elif akind == "breakfast":
        htype, bin_type, timepoint = args.htype, args.bin_type, args.timepoint
        breakfast.breakfast(experiment, obs_period, htype, timepoint, err_type, ignore)

        # days comparison - HiFat2 only
        if experiment.name in ["HiFat2"]:
            obs_period = "comparison"
            breakfast_days_comparison.facets_groups_comparison_hue_group(experiment, obs_period, htype, timepoint,
                                                                         err_type, ignore)
            breakfast_days_comparison.facets_groups_comparison_hue_day(experiment, obs_period, htype, timepoint,
                                                                       err_type, ignore)

    elif akind == "within_as_structure":
        within_as_structure.within_as_structure(experiment, obs_period)

        # # days comparison, only HiFat2:
        if experiment.name in ["HiFat2"]:
            num_mins = [6, 15]
            within_as_structure_days_comparison.within_as_structure_mice(experiment, num_mins)
            within_as_structure_days_comparison.within_as_structure_groups(experiment, num_mins)

    elif akind == "time_budget":
        htype, bin_type = args.htype, args.bin_type
        time_budget.time_budgets(experiment, obs_period, htype, bin_type, err_type, ignore)

        # # days comparison, only HiFat2
        if experiment.name in ["HiFat2"]:
            obs_period = "comparison"
            time_budget_days_comparison.facets_groups_days_comparison_hue_group(experiment, obs_period, htype, bin_type,
                                                                                err_type, ignore)
            time_budget_days_comparison.facets_groups_days_comparison_hue_day(experiment, obs_period, htype, bin_type,
                                                                              err_type, ignore)



if __name__ == '__main__':
    main()


# old
# def run(exp_name, obs_period, akind, htype=None, bin_type=None, xbins=None, ybins=None, timepoint=None, err_type=None,
#         ignore=None):
#
#     experiment = Experiment(exp_name)
#
#     if akind == 'raster':  # plot entire group / mousedays
#         if htype == 'groups':
#             rasters.group_rasters(experiment, obs_period, write_days=True, AS_only=False)
#         elif htype == 'mousedays':  # plot single mousedays
#             raster_distance_mouseday.plot_all_mds(experiment, obs_period)
#         elif htype =='mice':  # plot given mouse (hardcoded)
#             # rasters.mouse_raster(experiment, obs_period, mouse_label=('C57BL6J', 'M2101'), write_days=True)
#             rasters.all_mice_rasters(experiment, obs_period, write_days=True)
#
#     elif akind == 'position':
#         position_density.position_density(experiment, obs_period, htype, bin_type, xbins, ybins, ignore)
#         # # csv files
#         position_density.write_to_csv(experiment, obs_period, bin_type, ignore=ignore)
#
#     elif akind == 'features':
#         if htype == 'groups':
#             # features_vs_ct_as_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
#             # features_vs_ct_fdl_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
#             # features_vs_ct_groups_day_breakdown.facets_groups_day_breakdown(experiment, obs_period, bin_type, err_type,
#             #                                                                 ignore)
#             # # csv files
#             features_vs_ct.write_to_csv(experiment, obs_period, bin_type, ignore)
#
#             # # days comparison, only HiFat2
#             if experiment.name in ['HiFat2']:
#                 obs_period = 'comparison'
#                 features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_day(experiment, obs_period,
#                                                                                               bin_type, err_type,
#                                                                                               ignore)
#                 features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_group(experiment, obs_period,
#                                                                                                 bin_type, err_type,
#                                                                                                 ignore)
#
#     elif akind == 'breakfast':
#         # breakfast.breakfast(experiment, obs_period, htype, timepoint, err_type, ignore)
#
#         # # days comparison, only HiFat2
#         if experiment.name in ['HiFat2']:
#             obs_period = 'comparison'
#             breakfast_days_comparison.facets_groups_comparison_hue_group(experiment, obs_period, htype, timepoint,
#                                                                          err_type, ignore)
#             # breakfast_days_comparison.facets_groups_comparison_hue_day(experiment, obs_period, htype, timepoint,
#             #                                                            err_type, ignore)
#
#     elif akind == 'within_as_structure':
#         # within_as_structure.within_as_structure(experiment, obs_period)
#
#         # # days comparison, only HiFat2:
#         if experiment.name in ['HiFat2']:
#             num_mins = [6, 15]
#             # within_as_structure_days_comparison.within_as_structure_mice(experiment, num_mins)
#             within_as_structure_days_comparison.within_as_structure_groups(experiment, num_mins)
#
#     elif akind == 'time_budget':
#         time_budget.time_budgets(experiment, obs_period, htype, bin_type, err_type, ignore=ignore)
#
#         # # days comparison, only HiFat2
#         if experiment.name in ['HiFat2']:
#             obs_period = 'comparison'
#             # time_budget_days_comparison.facets_groups_days_comparison_hue_group(experiment, obs_period, htype, bin_type, err_type, ignore)
#             # time_budget_days_comparison.facets_groups_days_comparison_hue_day(experiment, obs_period, htype, bin_type, err_type, ignore)
#
#

#     exp_name = sys.argv[1]  # experiment name
#     obs_period = sys.argv[2]  # see keys.obs_period_to_days
#     akind = sys.argv[3]
#     err_type = 'sem'  # seaborn: sem, sd, ci95
#     ignore = True
#     if akind == 'raster':
#         htype = sys.argv[4]  # None (groups), mouse_label,  mousedays
#         run(exp_name, obs_period, akind, htype=htype, ignore=ignore)
#
#     elif akind == 'position':
#         bin_type = sys.argv[4]  # 7cycles, 12bins, 4bins,
#         htype = sys.argv[5]
#         xbins, ybins = int(sys.argv[6]), int(sys.argv[7])  # (2, 4), (12, 24)
#         run(exp_name, obs_period, akind, bin_type=bin_type, htype=htype, xbins=xbins, ybins=ybins, ignore=ignore)
#
#     elif akind == 'features':
#         bin_type = sys.argv[4]  # 3cycles, 12bins, 4bins, 24bins
#         htype = sys.argv[5]  # groups. mice, mousedays
#         run(exp_name, obs_period, akind, bin_type=bin_type, htype=htype, err_type=err_type, ignore=ignore)
#
#     elif akind == 'breakfast':
#         timepoint = sys.argv[4]  # onset, offset
#         htype = sys.argv[5]
#         run(exp_name, obs_period, akind, timepoint=timepoint, htype=htype, err_type=err_type, ignore=ignore)
#
#     elif akind == 'within_as_structure':
#         run(exp_name, obs_period, akind, ignore=ignore)
#
#     elif akind == 'time_budget':
#         bin_type = sys.argv[4]  # 6cycles
#         htype = sys.argv[5]  # hierarchy: groups, mouse, mousedays
#         run(exp_name, obs_period, akind, htype=htype, bin_type=bin_type, err_type=err_type, ignore=ignore)
#     else:
#         raise ValueError("Received unknown code: {}".format(akind))

    # elif akind == 'days':
    #     SS_features_across_days.plot(experiment, days, bin_type, htype, err_type, ignore)
    #
    # elif akind == 'distributions':
    #     bt_type = 'F'
    #     features_distribution.plot(experiment, days, bt_type, htype, err_type, ignore)
    #
    # elif akind == 'breakfast':
    #     breakfast_viz.plot(experiment, days, ignore)
    #
    # elif akind == 'Log2(x_ux)':
    #     pass
    #     # argv[1],[2],[3]
    #     # SS_mdrl.plot_mdrl(experiment, days, htype, bin_type='3cycles', err_type=err_type, ignore=ignore)
    #     # # argv[1],[2]
    #     # SS_mdrl.write_to_csv(experiment, days, bin_type='3cycles', ignore=ignore)


# # # older
    # for days in days_act_to_actlabel[name].keys():
    #     run(name, days, akind, bin_type, htype, err_type, ignore)

    # elif name == 'HiFat2':
    #     for days in [tuple(range(5, 12+1)), tuple(range(10, 12+1)), tuple(range(13, 17+1)), tuple(range(18, 28+1)), tuple(range(5, 28+1))]:
    #         run(experiment, days, akind, bin_type, htype, err_type, ignore)
    #
    # elif name in ['HiFat1', 'StrainSurvey']:
    #     days = tuple(range(5, 16 + 1))
    #     run(experiment, days, akind, bin_type, htype, err_type, ignore)



#
# with open('old.csv', 'r') as t1, open('new.csv', 'r') as t2:
#     fileone = t1.readlines()
#     filetwo = t2.readlines()
#
# with open('update.csv', 'w') as outFile:
#     for line in filetwo:
#         if line not in fileone:
#             outFile.write(line)

    # # older
    # SS_features.plot_groups(experiment, days, bin_type, err_type, ignore)
    # SS_features.plot_mice(experiment, days, bin_type, err_type='sem', ignore=ignore)
    # SS_features_corrs.plot_groups(experiment, days, bin_type, ignore=True)
    # write_to_csv.features_tocsv(experiment, days, bintype)

        # HiFat1 only
        # features.plot_compare_all_groups(experiment, days, bin_type, err_type='sem')
        # per active state
        # features_active_state.plot_compare_all_groups(experiment, days)
        # features_active_state_by_category.groups_boxplot(experiment, days, CAT_CODE='1')
        # features_active_state_by_category.mouse_boxplot(experiment, days, CAT_CODE='1')

        # distributions
        # features_distribution.plot_groups(exp, days, bin_type)
        # features_distribution.plot_active_states_group_distribution(exp, days, bin_type='ASbins')
        # features.to_csv(experiment, days, bin_type)

        # feature compensation:
        # feature_keys = ['FBS', 'FBD', 'FBI'], #['LBS', 'LBD', 'LBI'], #['ASP', 'ASD', 'ASFI'], #['ASR', 'ASD', 'ASFI']
        # features_compensation.plot_feeding(exp, feature_keys, days, bin_type, err_type='sem')

        # feature correlations: [('ASD', 'ASR'), ('ASP', 'ASFI'), ('ASD', 'ASFI'), ('FBS', 'ASFBR'), ('FBD', 'FBI')
