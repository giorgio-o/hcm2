# Darren Rhea, 2012
# Chris Hillar revision, May 2013
# Ram Mehta revision, June 2013
#
# Fix data stream.
#
# Copyright (c) 2013. All rights reserved.
from __future__ import division

import collections
import numpy as np
import logging

from base import Fixer
from util.cage import Cage

logger = logging.getLogger()


class DeviceFix(Fixer):
    CODE_POSITION = 1

    CODE_PHOTO_START = 2

    CODE_PHOTO_STOP = 4

    CODE_LICK_START = 3

    CODE_LICK_STOP = 6

    IndexData = collections.namedtuple("IndexData", (
        "pos_start", "pos_stop", "first_non_pos", "last_non_pos", "initial_seq_exists", "final_seq_exists"))

    C = Cage()
    photo_pos = C.photo_pos
    lick_pos = C.lick_pos

    def __call__(self, md):
        data_ = md.data['preprocessing']
        # Get the device data
        photo_start_events, photo_stop_events = self.extract_startstop_events(data_['_photobeam_data'][1:, :],
                                                                              self.CODE_PHOTO_START,
                                                                              self.CODE_PHOTO_STOP, self.photo_pos)

        lick_start_events, lick_stop_events = self.extract_startstop_events(data_['_lickometer_data'][1:, :],
                                                                            self.CODE_LICK_START, self.CODE_LICK_STOP,
                                                                            self.lick_pos)

        # Label the movement data with position codes
        shape = (data_['_movement_data'].shape[0], 1)
        position_events = np.hstack((data_['_movement_data'], self.CODE_POSITION * np.ones(shape)))
        # Make the time-sorted list of events
        events = np.vstack(
            (position_events, lick_start_events, lick_stop_events, photo_start_events, photo_stop_events))

        # sort the events
        events = events[events[:, 0].argsort()]
        events = self.fix_events(events)
        data_['t'], data_['x'], data_['y'] = \
            events[:, 0], events[:, 1], events[:, 2]

    @staticmethod
    def extract_startstop_events(data, start_code, stop_code, pos, delta=1.e-8):
        """ Returns a numpy array with 4 columns (time, x, y, event_type). """
        if data.shape[0] == 0:
            logger.warn("Device fix just experienced no events in the whole day!")
            return np.ones((0, 4)), np.ones((0, 4))

        shape = (data.shape[0], 1)
        positions = np.kron(np.ones(shape), pos)
        start = np.hstack((data[:, 0].reshape(shape), positions, start_code * np.ones(shape)))

        stop = np.hstack(((data[:, 0] + data[:, 1] + delta).reshape(shape), positions, stop_code * np.ones(shape)))

        return start, stop

    def find_next_non_pos_event(self, event_types, offset=0):
        nevents = event_types.shape[0]
        while offset < nevents and event_types[offset] == self.CODE_POSITION:
            offset += 1
        return offset

    def find_next_pos_event(self, event_types, offset=0):
        nevents = event_types.shape[0]
        while offset < nevents and event_types[offset] != self.CODE_POSITION:
            offset += 1
        return offset

    def extract_index_data(self, event_types):
        nevents = event_types.shape[0]
        final_seq_exists = True
        pos_start_idx, pos_stop_idx = [], []
        n = first_non_pos_idx = self.find_next_non_pos_event(event_types)
        if n >= nevents:
            raise ValueError("No non position events found during device fix.")

        while True:
            n = self.find_next_pos_event(event_types, n)
            last_non_pos_idx = n - 1
            if n >= nevents:
                final_seq_exists = False
                break

            pos_start_idx.append(n)
            n = self.find_next_non_pos_event(event_types, n)
            if n >= nevents:
                break

            pos_stop_idx.append(n)

        return self.IndexData(pos_start=pos_start_idx, pos_stop=pos_stop_idx, first_non_pos=first_non_pos_idx,
                              last_non_pos=last_non_pos_idx, initial_seq_exists=(event_types[0] == self.CODE_POSITION),
                              final_seq_exists=final_seq_exists)

    def get_known_pos(self, event_type):
        if event_type in (self.CODE_LICK_STOP, self.CODE_LICK_START):
            return self.lick_pos
        if event_type in (self.CODE_PHOTO_STOP, self.CODE_PHOTO_START):
            return self.photo_pos
        if event_type == self.CODE_POSITION:
            raise ValueError("Should not receive a position code.")

        raise ValueError("Received unknown code %s" % event_type)

    def fix_events(self, events):
        """ Fix the runs so they start and stop at their known positions. """
        event_types = np.array(events[:, 3], "int")
        index_data = self.extract_index_data(event_types)
        fixed = events.copy()

        # number of fully anchored runs := len(index_data.pos_stop)
        for i in xrange(len(index_data.pos_stop)):
            self.fix_run(fixed, events, event_types, index_data.pos_start[i], index_data.pos_stop[i])

        # Fix positions prior to first lick/eat event
        if index_data.initial_seq_exists:
            trans = events[index_data.first_non_pos, 1:3] - events[index_data.first_non_pos - 1, 1:3]
            fixed[:index_data.first_non_pos, 1:3] = events[:index_data.first_non_pos, 1:3] + np.kron(
                np.ones((index_data.first_non_pos, 1)), trans)

        # Fix positions after the last lick/eat event
        if index_data.final_seq_exists:
            trans = events[index_data.last_non_pos, 1:3] - events[index_data.last_non_pos + 1, 1:3]
            length = events.shape[0] - index_data.last_non_pos - 1
            fixed[index_data.last_non_pos + 1:, 1:3] = events[index_data.last_non_pos + 1:, 1:3] + np.kron(
                np.ones((length, 1)), trans)

        events = fixed
        move_idx_depr = np.nonzero(events[:, 3] == 1)[0]  # move_idx_depr = mlab.find(events[:, 3] == 1)  #deprecated
        move_idx = np.where(events[:, 3] == 1)[0]
        np.testing.assert_array_equal(move_idx, move_idx_depr, 'wrong')
        events = events[move_idx, :]
        return events

    def fix_run(self, fixed, events, event_types, start_idx, stop_idx):
        start_event_type = event_types[start_idx - 1]
        known_start_pos = self.get_known_pos(start_event_type)
        stop_event_type = event_types[stop_idx]
        known_stop_pos = self.get_known_pos(stop_event_type)

        supposed_start_pos = events[start_idx, 1:3]
        supposed_stop_pos = events[stop_idx - 1, 1:3]
        if stop_idx >= start_idx + 2:
            trans = known_start_pos - supposed_start_pos
            correct = known_stop_pos - (supposed_stop_pos + trans)
            time_0_to_1 = np.zeros((stop_idx - start_idx, 1))
            time_0_to_1[:, 0] = (events[start_idx:stop_idx, 0] - events[start_idx - 1, 0]) * 1.0 / (
                    events[stop_idx, 0] - events[start_idx - 1, 0])

            fixed[start_idx:stop_idx, 1:3] = events[start_idx:stop_idx, 1:3] + np.kron(
                np.ones((stop_idx - start_idx, 1)), trans) + np.kron(time_0_to_1, correct)

        if stop_idx == start_idx + 1:
            fixed[start_idx, 1:3] = known_start_pos
