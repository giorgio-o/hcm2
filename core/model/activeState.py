import numpy as np

from util.utils import hcm_time_to_ct_string,seconds_to_mins_and_secs_tuple
import logging

logger = logging.getLogger(__name__)


class ActiveState(object):

    def __init__(self, mouseday=None, starttime=0, endtime=1):
        self.experiment = mouseday.experiment
        self.group = mouseday.group
        self.mouse = mouseday.mouse
        self.mouseday = mouseday
        self.day = mouseday.day
        self.starttime = starttime
        self.endtime = endtime
        self.array = np.array([starttime, endtime])

    def __str__(self):
        return "group{}: {}, indv{}: {}, day: {}, [{} -> {}] Active for {}m {}s".format(
            self.group.number, self.group.name, self.mouse.number, self.mouse.name, self.day,
            hcm_time_to_ct_string(self.starttime), hcm_time_to_ct_string(self.endtime),
            *seconds_to_mins_and_secs_tuple(self.duration))

    @property
    def duration(self):
        return self.endtime - self.starttime

