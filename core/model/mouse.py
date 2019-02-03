# hcm/core/model/mouse.py
import logging
import os

from ..helpers import load_raw

logger = logging.getLogger(__name__)


class Mouse(object):
    """HCM mouse class. """

    def __init__(self, group=None, number=1, name=None):
        self.experiment = None or group.experiment
        self.group = group
        self.number = number  # ordinal
        self.name = "M{}".format(name)  # actual number e.g. "M2101"
        self.days = self.experiment.days
        self.data = dict(group=self.group, mouse=self.name, active_state_structure=dict())

    def __str__(self):
        return "group{}: {}, indv{}: {}".format(self.group.number, self.group.name, self.number, self.name)

    @property
    def label(self):
        """Returns tuple (group_name, mouse_number). """
        return self.group.name, self.name

    @property
    def is_ignored(self):
        """Returns True if mouse is ignored. """
        return (self.group.name, self.name) in self.experiment.ignored_mice

    @property
    def label_long(self):
        """Returns tuple (group_number, group_name, individual_number, mouse_number). """
        return self.group.number, self.group.name, self.number, self.name

    @property
    def filename_long(self):
        """Returns string for filename. """
        return "group{}_{}_indv{}_{}".format(*self.label_long)

    def all_mousedays(self, days=()):
        """Returns generator of all Mouseday objects for selected days. """
        from mouseday import MouseDay
        days = days or self.days
        return (MouseDay(mouse=self, day=day) for day in days)

    def mousedays(self, days=()):
        """Returns generator of valid (not ignored) Mouseday objects for selected days. """
        days = days or self.days
        path = os.path.join("preprocessing", "AS_timeset")
        labels = list(self.experiment.mouseday_labels_from_binary_path(self.experiment.path_to_binary(subdir=path)))
        return (md for md in self.all_mousedays(days) if md.label in labels)

    # data preprocessing
    def process_raw_data(self, days=(), fixers=()):
        """Raw data preprocessing, creates bouts and active states from event raw data.
            Stores computed variables as (keys, values) in mouseday data dictionary.
        """
        fixers = [f(self.experiment) for f in fixers]
        for md in self.all_mousedays(days):
            logger.info(str(md))
            md.load_raw_data()
            try:
                load_raw.check_consistency(md.data)
            except ValueError, err:
                logger.error("Skipping {} due to: {}.".format(md, err))
                continue
            except IndexError, err:
                logger.error("Skipping {} due to: {}. "
                             "This might be empty _photo, _lick, _or _move raw data.\n".format(md, err))
                continue
            # fixers
            for fix in fixers:
                fix(md)
            # process
            md.process_raw_data()
            md.create_ingestion_bouts()
            md.create_locomotion_bouts()
            md.create_active_states()
            # save
            md.save_npy_data()

    def create_position(self, days, bin_type, xbins, ybins):
        """Creates position data. """
        for md in self.mousedays(days):
            logger.info("{}: {}".format(bin_type, str(md)))
            md.create_position(bin_type, xbins, ybins, binary=False)

    def create_features(self, days=(), bin_type="12bins"):
        """Creates features data. """
        for md in self.mousedays(days):
            logger.info("{}: {}".format(bin_type, str(md)))
            md.create_features(bin_type)

    def create_breakfast(self, days, timepoint, tbin_size, num_secs):
        """Creates breakfast data. """
        days = days or self.days
        for md in self.mousedays(days):
            logger.info("{}".format(str(md)))
            md.create_breakfast(timepoint, tbin_size, num_secs)

    def create_within_as_structure(self, days, num_mins=15, for_mice=True):
        """Creates within active states structure data. """
        from operator import itemgetter
        days = days or self.days
        all_tuples = [x for md in self.mousedays(days) for x in md.create_within_as_structure(num_mins)]
        # sort by active state duration
        sorted_tups = sorted(all_tuples, key=itemgetter(1))
        return [x[0] for x in sorted_tups] if for_mice else all_tuples

    def create_time_budget(self, days, bin_type):
        """Creates time budget data. """
        days = days or self.days
        for md in self.mousedays(days):
            logger.info("{}".format(str(md)))
            md.create_time_budget(bin_type)