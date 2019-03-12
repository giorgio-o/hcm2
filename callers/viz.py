# hcm2/callers/viz.py
""" G. Onnis, 2017
revised 11.2018
HCM experiment data module for analysis results and visualizations.
# Copyright (c) 2018. All rights reserved.
"""
from argparse import ArgumentParser

from pandas import set_option
import argparse

from util import utils
from core.model.experiment import Experiment
from visualization import position_density, rasters
from visualization import features_vs_ct_as_groups_avg, features_vs_ct_fdl_groups_avg, features_vs_ct_csvfiles
from visualization import features_vs_ct_groups_day_breakdown
from visualization import features_vs_ct_as_groups_avg_days_comparison
from visualization import breakfast, breakfast_days_comparison
from visualization import within_as_structure, within_as_structure_days_comparison
from visualization import time_budget, time_budget_days_comparison

set_option('display.width', 300)


def get_parser():
    """Parse command line input """

    parser = argparse.ArgumentParser()  # type: ArgumentParser
    parser.add_argument("name")
    parser.add_argument("akind")
    parser.add_argument("--obs_period", type=str, default="acclimated")
    parser.add_argument("--htype", type=str, default="groups")
    parser.add_argument("--mouse_label", type=str, default=None)
    parser.add_argument("--bin_type", type=str)
    parser.add_argument("--xbins", type=int, default=2)
    parser.add_argument("--ybins", type=int, default=4)
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
    """Main visualization file for analysis results and visualizations.

    Parameters
    ----------
    name : str
        Experiment name.
    akind : {"raster", "position", "features", "breakfast", "time_budget"}
        Type of analysis to be performed.
    bin_type : {"3cycles", "6cycles", "7cycles", "4bins", "12bins", "24bins"}
        Time binning.
    xbins, ybins: int
        horizontal (vs. vertical) cage discretization.
    htype : {"group", "mouse", "day"}
        Hierarchical tier for analysis.
    timepoint : {"onset", "offset"}
        startpoint for beakfast hypothesis.

    Examples
    --------
    Run analysis form command line:
    python callers/viz.py HiFat1 rasters --htype mice
    python callers/viz.py HiFat1 position --xbins 2 --ybins 4 --bin_type 7cycles
    python callers/viz.py HiFat1 features --bin_type 12bins

    """
    # analysis parameters
    parser = get_parser()
    args = parser.parse_args()
    name, akind = args.name, args.akind
    obs_period = args.obs_period
    err_type, ignore, csv_file = args.err_type, args.ignore, args.csv_file
    # initialize experiment
    experiment = Experiment(name)

    if akind == "raster":
        rasters.raster(experiment, obs_period, htype=args.htype, mouse_label=args.mouse_label,
                       write_days=args.write_days, as_only=args.as_only)

    elif akind == "position":
        position_density.position_density(experiment, obs_period, htype=args.htype, bin_type=args.bin_type,
                                          xbins=args.xbins, ybins=args.ybins, ignore=args.ignore,
                                          csv_file=args.csv_file)

    elif akind == "features":
        htype, bin_type, day_break = args.htype, args.bin_type, args.day_break
        if htype == "groups":
            if csv_file:
                features_vs_ct_csvfiles.write_to_csv(experiment, obs_period, bin_type, ignore)
            else:
                features_vs_ct_as_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
                features_vs_ct_fdl_groups_avg.facets_group_avgs(experiment, obs_period, bin_type, err_type, ignore)
                if day_break:
                    features_vs_ct_groups_day_breakdown.facets_groups_day_breakdown(
                        experiment, obs_period, bin_type, err_type, ignore)

                if experiment.name in ["HiFat2"]:  # days comparison, only HiFat2
                    obs_period = "comparison"
                    features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_day(
                        experiment, obs_period, bin_type, err_type, ignore)
                    features_vs_ct_as_groups_avg_days_comparison.facets_groups_comparison_hue_group(
                        experiment, obs_period, bin_type, err_type, ignore)

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

    elif akind == "within_as_structure":
        within_as_structure.within_as_structure(experiment, obs_period)

        # # days comparison, only HiFat2:
        if experiment.name in ["HiFat2"]:
            num_mins = [6, 15]
            within_as_structure_days_comparison.within_as_structure_mice(experiment, num_mins)
            within_as_structure_days_comparison.within_as_structure_groups(experiment, num_mins)

    else:
        raise ValueError("Unknown code {}".format(akind))

if __name__ == '__main__':
    main()