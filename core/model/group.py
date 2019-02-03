# hcm/core/model/group.py


class Group(object):
    """ HCM group class """

    def __init__(self, experiment, number=1, name=None):
        self.experiment = experiment
        self.number = number  # ordinal
        self.name = name
        self.data = dict(group=self.name)

    def __str__(self):
        return "group{}: {}".format(self.number, self.name)

    @property
    def label(self):
        """ returns group label tuple (group_number, group_name)"""
        return self.number, self.name

    @property
    def mice(self):
        """ returns generator of Mouse objects """
        from mouse import Mouse
        return (Mouse(group=self, number=number + 1, name=name) for number, name in
                enumerate(self.experiment.group_mice_dict[self.name]))

    @property
    def tot_mice(self):
        """ returns number of mice in group """
        return len(list(self.mice))

    @property
    def valid_mice(self):
        """ returns list of tuples (group_name, mouse_number) for valid (not ignored) mice """
        return [m.label for m in self.mice if not m.is_ignored]

    @property
    def ignored_mice(self):
        """ returns list of tuples (group_name, mouse_number) """
        return [m.label[1] for m in self.mice if m.is_ignored]

    def valid_mousedays(self, days=()):
        """ returns list of tuples (mouse_number, day) for selected days for valid (not ignored) mice"""
        return [md.label[1:] for md in self.experiment.all_mousedays(days) if
                md.group.label == self.label and not md.is_ignored]

    def ignored_mousedays(self, days=()):
        """ returns list of tuples (mouse_number, day) for selected days for ignored mice"""
        return [md.label[1:] for md in self.experiment.all_mousedays(days) if
                md.group.label == self.label and md.is_ignored]

    @property
    def mouse_numbers(self):
        """ returns list of mouse_numbers """
        return self.experiment.group_mice_dict[self.name]  # int

    def process_raw_data(self, days=(), fixers=()):
        """ raw data preprocessing """
        for mouse in self.mice:
            # for mouse in list(self.mice)[:2]:
            mouse.process_raw_data(days, fixers)

    def create_position(self, days, bin_type, xbins, ybins):
        """ creates position data """
        for mouse in self.mice:
            # for mouse in list(self.mice)[:2]:
            mouse.create_position(days, bin_type, xbins, ybins)

    def create_features(self, days, bin_type):
        """ creates features data """
        for mouse in self.mice:
            # for mouse in list(self.mice)[:2]:
            mouse.create_features(days, bin_type)

    def create_breakfast(self, days, timepoint, tbin_size, num_secs):
        """ creates breakfast data """
        for mouse in self.mice:
            # for mouse in list(self.mice)[:3]:
            mouse.create_breakfast(days, timepoint, tbin_size, num_secs)

    def create_within_as_structure(self, days):
        """ creates within active states structure data """
        from operator import itemgetter
        from util.utils import flat_out_list
        all_tuples = list()
        for mouse in self.mice:
            # for mouse in list(self.mice)[:4]:
            if not mouse.is_ignored:
                all_tuples.append(mouse.create_within_as_structure(days, for_mice=False))
        # sort by active state duration
        sorted_tups = sorted(flat_out_list(all_tuples), key=itemgetter(1))
        return [x[0] for x in sorted_tups]

    def create_time_budget(self, days, bin_type):
        """ creates time budget data """
        for mouse in self.mice:
            # for mouse in list(self.mice)[:3]:
            mouse.create_time_budget(days, bin_type)