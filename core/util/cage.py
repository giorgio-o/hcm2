# hcm/util/cage.py
import numpy as np


class Cage(object):
    """HCM cage object. """

    # Define the positions of the devices
    # Maybe Evan was wrong, this is the Alan measurement
    lick_pos = np.array([0.0, 0.0 + 3.0])
    # according to Evan, this is akind the center of mass of an eating mouse should be
    photo_pos = np.array([-12.7500 + 0.0, -2.6 + 3.000])

    def __init__(self, PRINTOUT=False):
        self.xNovelObject = 3.75
        self.yNovelObject = 22.5
        self.nearnessRadius = 7.0
        # locations
        # sleeping is used in backwardfix
        self.mouseCMXSleeping = -13.25  # old alan/darren value
        self.mouseCMYSleeping = 38.0  # old
        # self.mouseCMXSleeping= -10. # corrected by xy_plots 
        # self.mouseCMYSleeping=36.5 
        # this is some point close to the center of available space in the niche
        self.mouseCMXAtNest = -13.25 + 4.25  # larry
        self.mouseCMYAtNest = 38.0 - 1.15
        self.mouseCMXAtPhoto = -12.75  # the Alan opinion
        self.mouseCMYAtPhoto = 0.4  # old alan/darren
        # self.mouseCMXAtPhoto=-13.75 # corrected by xy_plots 
        # self.mouseCMYAtPhoto=3.0 
        self.mouseCMXAtLick = 0.0
        self.mouseCMYAtLick = 3.0
        self.mouseCMXDoorway = -8.25 - 1.2  # after Larry's
        self.mouseCMYDoorway = 35.0 + 1.2  #
        # # # #CHEKC THIS!!!!!
        self.enclosureMidX = -6.25
        self.enclosureMidY = 22.0
        # boundaries
        self.cmy_lower = 1.0
        self.cmx_upper = 3.75
        self.cmx_lower = -16.25
        # nest xs
        self.nestLeftX = self.cmx_lower + 3.0
        self.nestRightX = -6.25 + 1.2  # observational. + 1.2cm after Larry's observation (May 2015)
        self.nestBottomXMax = -8.75  # observational
        self.cmy_upper = 43.0
        # nest ys
        self.nestTopY = self.cmy_upper - 3.0  # -4.0 observational, -3.0 after Larry's
        self.nestRightYMin = 35.5  # observational
        self.nestBottomY = 33 - 1.2  # observational - 1.2cm after Larry's
        # any figure of the cage you want to plot
        self.pictureXMin = self.cmx_lower
        self.pictureXMax = self.cmx_upper
        self.pictureYMin = 0  # cage bottom at feeder (cmy_lower-1)
        self.pictureYMax = self.cmy_upper
        self.enclosurePenaltyLowerBoundXMin = -10.5  # cage bottom at lickometer, at Y = self.cmy_lower
        self.enclosurePenaltyLowerBoundXMax = 3.75
        # # center_cage rectangle: following Larry's drawing, niche is about 12x9
        self.xr_lower = self.cmx_lower + 7
        self.xr_upper = self.cmx_upper - 7
        self.yr_lower = self.cmy_lower + 12
        self.yr_upper = self.cmy_lower + 31
        # mouse detection coordinates xy at lick, feed, nest
        self.xy_l = [self.mouseCMXAtLick, self.mouseCMYAtLick]
        self.xy_p = [self.mouseCMXAtPhoto, self.mouseCMYAtPhoto + 0.6]  # Larry
        self.xy_n = [self.mouseCMXAtNest, self.mouseCMYAtNest]
        # origin at cage bottom at lickometer
        self.xy_o = [self.mouseCMXAtLick, self.cmy_lower]
        # distances from origin
        self.to_l = np.sqrt((self.xy_l[0] - self.xy_o[0]) ** 2 + (self.xy_l[1] - self.xy_o[1]) ** 2)
        self.to_p = np.sqrt((self.xy_p[0] - self.xy_o[0]) ** 2 + (self.xy_p[1] - self.xy_o[1]) ** 2)
        self.to_n = np.sqrt((self.xy_n[0] - self.xy_o[0]) ** 2 + (self.xy_n[1] - self.xy_o[1]) ** 2)
        # four corners coordinates
        self.xy_topleft = [self.cmx_lower, self.cmy_upper]
        self.xy_botleft = [self.cmx_lower, 0]  # cage bottom at feeder
        self.xy_botright = [self.cmx_upper, 0]
        self.xy_topright = [self.cmx_upper, self.cmy_upper]
        # cage size
        self.dx = self.cmx_upper - self.cmx_lower
        self.dy = self.cmy_upper - self.cmy_lower

        # add dictionary location to rectangle in (2, 4) cage discretization
        self.activity_to_rectangle = {'N': [(0, 0)], 'F': [(3, 0)], 'D': [(3, 1)]}

        if PRINTOUT:
            print "mouse xy coordinates at locations and distances from origin (at lickometer device)", self.xy_o
            print "lick %s\ndistance_to_origin=%1.2f" % (self.xy_l, self.to_l)
            print "photo %s\ndist=%1.2f" % (self.xy_p, self.to_p)
            print "nest %s\ndist=%1.2f" % (self.xy_n, self.to_n)

            print "(outer) cage corners:"
            print "xy_topleft ", self.xy_topleft
            print "xy_botleft ", self.xy_botleft
            print "xy_botright ", self.xy_botright
            print "xy_topright ", self.xy_topright

            print "cage size: %1.2f x %1.2f cm2" % (self.dx, self.dy)

    @property
    def cage_boundaries(self):
        """returns HCM cage boundary coordinates. """
        return (self.cmx_lower, self.cmx_upper), (self.cmy_lower, self.cmy_upper)

    def map_rectangle_to_cage_coordinates(self, rect, xbins=2, ybins=4):
        """Converts a rectangle in xbins x ybins to corresponding rectangle in Cage coordinates.
            Format is [[p1, p2], [p3, p4]] where pi = (cage_height_location, cage_length_location).
            ### THIS GIVES WRONG CAGE LOCATIONS for top bottom left right
            # # # xbins ybins do NOT reflect cage geometry perfectly
        """
        ll = self.cmx_upper - self.cmx_lower
        hh = self.cmy_upper - self.cmy_lower
        delta_x = ll / xbins
        delta_y = hh / ybins
        r, c = rect
        coord_y = self.cmy_upper - delta_y * r
        coord_x = self.cmx_lower + delta_x * c
        return ((coord_x, coord_y), (coord_x + delta_x, coord_y), (coord_x, coord_y - delta_y),
                (coord_x + delta_x, coord_y - delta_y))

    def print_rect_cage_coordinates(self, rect):
        """Prints HCM cage coordinates. """
        tl, tr, bl, br = self.map_rectangle_to_cage_coordinates(rect)
        print "Mouse at rect {} location in original coordinates top_left, top_right, bot_left, bot_right: ".format(
            rect)
        print tl, tr, bl, br  # nest: (0, 0)(-16.25, 43.0) (-6.25, 43.0) (-16.25, 32.5) (-6.25, 32.5)

    @staticmethod
    def map_ethel_obs_to_cage2x4_grid(obs_nest_3x7):
        """Return ethel coordinates in (2,4) grid discretization. """
        a, b = obs_nest_3x7
        consistent = None
        if a >= 2 and b == 7:
            consistent = [(0, 0)]

        # out of niche
        elif a == 3 and b == 6:
            consistent = [(1, 0)]
        elif a == 2 and b == 6:
            consistent = [(0, 1)]

        elif a == 1 and b >= 6:
            consistent = [(0, 1)]
        elif a == 1 and b == 5:
            consistent = [(1, 1)]
        elif a == 1 and b == 4:
            consistent = [(1, 1), (2, 1)]
        elif a == 1 and b == 3:
            consistent = [(2, 1)]
        elif a == 1 and b <= 2:
            consistent = [(3, 1)]

        elif a == 2 and b == 5:
            consistent = [(1, 0), (1, 1)]
        elif a == 2 and b == 4:
            print "!! Whoa mouse slept in the middle of the cage !!"
            consistent = [(1, 0), (1, 1), (2, 0), (2, 1)]
        elif a == 2 and b == 3:
            consistent = [(2, 0), (2, 1)]
        elif a == 2 and b <= 2:
            consistent = [(3, 0), (3, 1)]

        elif a == 3 and b == 5:
            consistent = [(1, 0)]
        elif a == 3 and b == 4:
            consistent = [(1, 0), (2, 0)]
        elif a == 3 and b == 3:
            consistent = [(2, 0)]
        elif a == 3 and b <= 2:
            consistent = [(3, 0)]

        # elif a == 0 and b == 0:         # check with Ethel
        #     consistent = [[3, 1]]

        return consistent