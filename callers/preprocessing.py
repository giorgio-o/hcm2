# hcm2/callers/preprocessing.py
"""G. Onnis, 2017
revised 11.2018

HCM experiment data preprocessing module.

# Copyright (c) 2018. All rights reserved.

"""

import numpy as np
import os
import argparse

from core.model.experiment import Experiment
from util import utils, file_utils
from core.keys import obs_period_to_days

np.set_printoptions(linewidth=250,
                    precision=5,
                    threshold=8000,
                    suppress=True)


def summarize_log_file(logfilename):
    """Summarizes errors in log file. """
    from util import file_utils
    logpath = os.path.join(file_utils.hcm_dir(), "logs")
    filename = os.path.join(logpath, logfilename)
    print "reading logfile:\n", filename
    with open(filename) as f:
        lines = [l for l in f if "group" in l or "ERROR" in l]

    outfile = os.path.join(logpath, logfilename[:-4] + "_summarized.log")
    with open(outfile, 'w') as f:
        for l in lines:
            f.write(l)
    print "wrote summarized .log file to:\n", outfile


def create_logger(args, days, logconfigfilename, exp_name):
    """Creates preprocessing log file. """
    import datetime
    import logging.config
    log_dir = os.path.join(file_utils.datadir(exp_name), "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    text = args.bin_type or args.timepoint or ""
    now = datetime.datetime.now()
    today_now = "{}{:02d}{:02d}_h{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    logfilename = "{}_preprocessing_{}{}_{}.log".format(args.name, args.akind, text, today_now)
    logfilename = os.path.join(log_dir, logfilename)
    logconfigfilename = os.path.join(file_utils.repo_dir(), logconfigfilename)
    logging.config.fileConfig(logconfigfilename, defaults=dict(logfilename=logfilename))
    log = logging.getLogger(__name__)
    log.info("Experiment {}, days: {}".format(args.name, days))
    return log, logfilename


def get_parser():
    """Parse command line input. """
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("akind")
    parser.add_argument("--bin_type", type=str)
    parser.add_argument("--xbins", type=int)
    parser.add_argument("--ybins", type=int)
    parser.add_argument("--timepoint", type=str)
    parser.add_argument("--obs_period", type=str, default="acclimated")
    return parser


@utils.timing
def main():
    """Main data preprocessing file. 
        Generates npy files from raw HCM data and saves it in: <exp_name>/preprocessing/ 
        Generates json files from npy preprocessed data and saves it in: <exp_name>/json_files/

    Parameters
    ----------
    name : str
        Experiment name.
    akind : {"raw", "position", "features", "breakfast", "time_budget"}
        Type of analysis to be performed:
            "raw": preprocess HCM raw data and saves a bunch of computed variables as .npy files in: 
                binary/<exp_name>/preprocessing/
            "position": computes occupancy times from preprocessed npy files and saves as json it in: 
                binary/<exp+name>/json_files/
                        
    bin_type : {"3cycles", "6cycles", "7cycles", "4bins", "12bins", "24bins"}
        Time binning:
            "3cycles": 24H, DC, LC.
            "6cycles": 24H, DC, LC, AS24H, ASDC, ASLC.
            "7cycles": 24H, DC, LC, AS24H, ASDC, ASLC, IS.
            "4bins": CT06-12 (last 6 hours of LC), CT12-18 (first 6 hours of DC), CT18-24 (last 6 hours of DC), 
                    CT24-06 (first 6 hours of LC).
            "12bins": 2-hours time bins, starting from CT06-08 to CT04-06.
            "24bins": 1-hour time bins, starting from CT06-07 to CT05-06.
            CT: Circadian Time.
            24H: 24hrs, DC: Dark Cycle, LC: Light Cycle, AS24H: Active States over the 24H, ASDC: Active states, DC, 
                ASLC: Active States, LC, IS: Inactive State.
    xbins, ybins: int
        Horizontal (vs. vertical) cage discretization, i.e. how many cells in the x (vs. y) direction. 
        Typical values: (2, 4) and (12, 24).
    htype : {"group", "mouse", "day"}
        Hierarchical tier for analysis.
    timepoint : {"onset", "offset"}
        startpoint for beakfast hypothesis.
    
    Examples
    --------
    Run analysis form command line:
    python callers/preprocessing.py HiFat1 raw
    python callers/preprocessing.py HiFat1 position --xbins 2 --ybins 4 --bin_type 7cycles
    python callers/preprocessing.py HiFat1 features --bin_type 12bins
    """
    # analysis parameters
    parser = get_parser()
    args = parser.parse_args()
    name, akind, obs_period = args.name, args.akind, args.obs_period
    # initialize experiment
    experiment = Experiment(name=name)
    days = obs_period_to_days[experiment.name][obs_period]
    # create logger
    logconfigfilename = "log_preprocessing.conf"
    logger, logfilename = create_logger(args, days, logconfigfilename, experiment.name)

    if akind == "raw":
        experiment.process_raw_data(days)
        summarize_log_file(logfilename)

    elif akind == "position":
        experiment.create_position(days, args.bin_type, args.xbins, args.ybins)

    elif akind == "features":
        experiment.create_features(days, bin_type=args.bin_type)

    elif akind == "breakfast":
        experiment.create_breakfast(days, timepoint=args.timepoint)

    elif akind == 'time_budget':  # todo: work in progress
        experiment.create_time_budget(days, args.bin_type)

    elif akind == "alles":
        experiment.process_raw_data(days)
        for xbins, ybins in [(2, 4), (12, 24)]:
            experiment.create_position(days, bin_type="7cycles", xbins=xbins, ybins=ybins)
        for bin_type in ["3cycles", "12bins"]:
            experiment.create_features(days, bin_type=bin_type)
        summarize_log_file(logfilename)
        
        
if __name__ == '__main__':
    main()