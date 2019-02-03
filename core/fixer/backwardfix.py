# hcm2/core/fixer/backwardfix.py
"""
# Darren Rhea, 2012
# Chris Hillar revision, May 2013
#
# Main generic wrapper for Home Enclosure Monitoring Experiment
#
# Copyright (c) 2013. All rights reserved.
"""

from __future__ import division

import numpy as np

from base import Fixer


class BackwardFix(Fixer):

    def __call__(self, md):
        data_ = md.data['preprocessing']
        # Create empty arrays
        backward_x = np.nan * data_['uncorrected_x']
        backward_y = np.nan * data_['uncorrected_y']

        # Loop backward through uncorrected X positions
        for i in xrange(len(data_['uncorrected_x']) + 1, 0, -1):
            # Position placeholders, a=x, b=y
            if i == len(data_['uncorrected_x']) + 1:
                # presume they are sleeping at the end.
                a = -13.25  # mouseCMXSleeping
                b = 38.0  # mouseCMYSleeping
            else:
                # Otherwise, use the last corrected position with delta x and y
                a = backward_x[i - 1] - data_['_delta_x'][i - 2]
                b = backward_y[i - 1] - data_['_delta_y'][i - 2]

            # Use the cage boundaries if the x position is in violation
            a = max(-16.25, a)  # cmx_lower
            a = min(3.75, a)  # cmx_upper

            # Use the cage boundaries if the y position is in violation
            b = max(0.4, b)  # mouseCMYAtPhoto
            b = min(43.0, b)  # cmy_upper

            # Apply the nest boundaries
            if a < -6.25:  # nestRightX
                b = min(b, 39.0)  # nestTopY
            if b > 33.0:  # nestBottomY
                a = max(a, -13.25)  # nestLeftX
            if a > -10.5:  # enclosurePenaltyLowerBoundXMin
                b = max(b, 1)

            # Update the x and y positions after all comparisons
            backward_x[i - 2] = a
            backward_y[i - 2] = b

        # a bad fix of a larger problem that the last entries of backward are super crazy
        backward_x[-1] = backward_x[-2]
        backward_y[-1] = backward_y[-2]

        # We assume you calculated it because you want to use it / go with it
        self.x = backward_x
        self.y = backward_y
        # md.data['preprocessing']['x'] = backward_x
        # md.data['preprocessing']['y'] = backward_y
