# hcm/callers/preprocessing.py
""" HCM experiment data preprocessing """
import numpy as np
import os
import argparse

from core.model.experiment import Experiment
from util import utils
from core.keys import obs_period_to_days

np.set_printoptions(linewidth=250,
                    precision=5,
                    threshold=8000,
                    suppress=True)


def summarize_log_file(logfilename):
    """ summarizes errors in log file """
    from util import file_utils
    logpath = os.path.join(file_utils.repo_dir(), "hcm/logs")
    filename = os.path.join(logpath, logfilename)
    print "reading logfile:\n", filename
    with open(filename) as f:
        lines = [l for l in f if "group" in l or "ERROR" in l]

    outfile = os.path.join(logpath, logfilename[:-4] + "_summarized.log")
    with open(outfile, 'w') as f:
        for l in lines:
            f.write(l)
    print "wrote summarized .log file to:\n", outfile


def create_logger(args, days):
    """ creates preprocessing log file"""
    import datetime
    import logging.config

    logconfigfilename = "log_preprocessing.conf"
    text = args.bin_type or args.timepoint or ""
    text = "_{}".format(text)
    now = datetime.datetime.now()
    today_now = "{}{:02d}{:02d}_h{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    logfilename = "{}_preprocessing_{}{}_{}.log".format(args.name, args.akind, text, today_now)
    logging.config.fileConfig(os.path.join("logs", logconfigfilename),
                              defaults=dict(logfilename=os.path.join("logs", logfilename)))
    log = logging.getLogger(__name__)
    log.info("Experiment {}, days: {}".format(args.name, days))
    return log, logfilename


def get_parser():
    """ parse command line input """
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
    """ main data preprocessing file.
        # python callers/preprocessing.py <experiment_name> <analysis_kind>
        - akind = "raw":
            preprocess raw data
            saves to npy files
        - akind = "position":
            creates position data
            bin_type = ["7cycles", "4bins", "12bins", "24bins"],
            (xbins, ybins) = [(2, 4), (12, 24)]
            "7cycles": ["24H", "DC", "LC", "AS24H", "ASDC", "ASLC", "IS"]
            saves to json files
        - akind = "features":
            creates features data, bin_type=["3cycles", "4bins", "12bins", "24bins"]
            "3cycles": ["24H", "DC", "LC"]
            saves to json files
        - akind = "breakfast":
            creates breakfast hypothesis data, timepoint=['onset', 'offset']
            saves to json files
        - akind = "time_budget":
            creates time budget data, bin_type=["6cycles"]
            "6cycles": ["24H", "DC", "LC", "AS24H", "ASDC", "ASLC"]
            saves to json files
    """
    # analysis parameters
    parser = get_parser()
    args = parser.parse_args()
    name, akind, obs_period = args.name, args.akind, args.obs_period
    # initialize experiment
    experiment = Experiment(name=name)
    days = obs_period_to_days[experiment.name][obs_period]
    # create logger
    logger, logfilename = create_logger(args, days)

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

    # obs_period =  'acclimated'  # see keys.obs_period_to_days
    # name = sys.argv[1]  # StrainSurvey, HiFat1, HiFat2, CORTTREAT, Stress_HCMe1r1, Stress1A, 2CFast
    # akind = sys.argv[2]
    # if akind == 'raw':
    #     # run callers/preprocessing.py <exp_name> raw
    #     E = run(name, obs_period, akind)
    # elif akind == 'position':
    #     # run callers/preprocessing.py <exp_name> position <bin_type> <xbins> <ybins>
    #     bin_type = sys.argv[3]  # 7cycles, 12bins, 4bins
    #     xbins, ybins = int(sys.argv[4]), int(sys.argv[5])  # (2, 4), (12, 24)
    #     E = run(name, obs_period, akind, bin_type=bin_type, xbins=xbins, ybins=ybins)
    # elif akind == 'features':
    #     # run callers/preprocessing.py <exp_name> features <bin_type>
    #     bin_type = sys.argv[3]  # 3cycles, 12bins, 4bins, 24bins
    #     E = run(name, obs_period, akind, bin_type=bin_type)
    # elif akind == 'breakfast':
    #     # run callers/preprocessing.py <exp_name> breakfast <timepoint>
    #     timepoint = sys.argv[3]  # onset, offset
    #     E = run(name, obs_period, akind, timepoint=timepoint)
    # elif akind == 'time_budget':
    #     # run callers/preprocessing.py <exp_name> time_budget
    #     bin_type = '6cycles'
    #     E = run(name, obs_period, akind, bin_type)
    # elif akind == 'alles':
    #     # run callers/preprocessing.py <exp_name> alles
    #     E = run(name, obs_period, akind)
    # else:
    #     raise ValueError("Received unknown code: {}".format(akind))

    # def summarize_log_file(logfilename):
    #     """ summarizes errors in log file """
    #     from util import file_utils
    #     logpath = os.path.join(file_utils.repo_dir(), 'hcm/logs')
    #     filename = os.path.join(logpath, logfilename)
    #     print "reading logfile:\n", filename
    #     with open(filename) as f:
    #         lines = [l for l in f if "group" in l or "ERROR" in l]
    # 
    #     outfile = os.path.join(logpath, logfilename[:-4] + '_summarized.log')
    #     with open(outfile, 'w') as f:
    #         for l in lines:
    #             f.write(l)
    #     print "wrote summarized .log file to:\n", outfile
    # 
    # 
    # def create_logger(name, days, akind, bin_type=None, timepoint=None):
    #     """ creates preprocessing log file"""
    #     import datetime
    #     import logging.config
    # 
    #     logconfigfilename = "log_preprocessing.conf"
    #     text = bin_type or timepoint or ""
    #     text = "_{}".format(text)
    #     now = datetime.datetime.now()
    #     today_now = '{}{:02d}{:02d}_h{:02d}{:02d}'.format(now.year, now.month, now.day, now.hour, now.minute)
    #     logfilename = '{}_preprocessing_{}{}_{}.log'.format(name, akind, text, today_now)
    #     logging.config.fileConfig(os.path.join('logs', logconfigfilename),
    #                               defaults=dict(logfilename=os.path.join('logs', logfilename)))
    #     log = logging.getLogger(__name__)
    #     log.info("Experiment {}, days: {}".format(name, days))
    #     return log, logfilename
    # 
    # 
    # @utils.timing
    # def run(name, obs_period, akind, bin_type=None, timepoint=None, xbins=None, ybins=None):
    #     """ main HCM data preprocessing file
    #         - raw: process HCM raw data, creates bouts and active states (npy files)
    #         - position: creates position data for 24H, DC, LC, AS24H, ASDC, ASLC (json files)
    #         - features: creates behavioral model features for feeding, drinking and locomotion (json files)
    #         - breakfast: creates breakfast hypothesis data (json files)
    #         - time budget: creates time budget data (json files)
    #         - alles: creates npy files from raw data (bouts and active states), position and feature data
    #     """
    #     from core.model.experiment import Experiment
    #     from core.keys import obs_period_to_days
    # 
    #     experiment = Experiment(name=name)
    #     days = obs_period_to_days[experiment.name][obs_period]
    #     # create logger
    #     logger, logfilename = create_logger(name, days, akind, bin_type, timepoint)
    # 
    #     if akind == "raw":  # first, create raw
    #         experiment.process_raw_data(days)
    #         summarize_log_file(logfilename)
    # 
    #     elif akind == 'position':  # second, create position
    #         experiment.create_position(days, bin_type, xbins, ybins)
    # 
    #     elif akind == "features":  # third, create features1
    #         experiment.create_features(days, bin_type=bin_type)
    # 
    #     elif akind == "breakfast":
    #         experiment.create_breakfast(days, timepoint=timepoint)
    # 
    #     elif akind == 'time_budget':
    #         experiment.create_time_budget(days, bin_type)
    # 
    #     elif akind == 'alles':
    #         experiment.process_raw_data(days)
    #         for xbins, ybins in [(2, 4), (12, 24)]:
    #             bin_type = '7cycles'
    #             experiment.create_position(days, bin_type, xbins, ybins)
    #         for bin_type in ['3cycles', '12bins']:
    #             experiment.create_features(days, bin_type=bin_type)
    #         summarize_log_file(logfilename)
    #     return experiment