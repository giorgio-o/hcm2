# hcm/visualization/viz/within_as_structure_days_comparison.py
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from core.keys import obs_period_to_days
from visualization.plot_util.plot_utils import add_figtitle, save_figure, set_subplots_labels
from within_as_structure import draw_structure, set_layout


def within_as_structure_mice(experiment, num_mins):
    if experiment.name in ['HiFat2']:
        for nmins in num_mins:
            res_subdir = os.path.join('within_as_structure')
            days_dict = OrderedDict((l, v) for l, v in obs_period_to_days[experiment.name].items()
                                    if l not in ['acclimated', 'comparison'])
            xmax = nmins if num_mins is not None else 200
            nrows = len(days_dict)

            # figure
            for group in list(experiment.groups):
                ncols = len(list(group.valid_mice))
                fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5), sharex=True, sharey=True)

                for num, (frame, days) in enumerate(obs_period_to_days[experiment.name].iteritems()):
                    if frame not in ['acclimated', 'comparison']:
                        m = 0
                        for mouse in list(group.mice):
                            if not mouse.is_ignored:
                                print "drawing within_as_structure for: {}, {} days".format(mouse, frame)
                                ax = axes[num][m]
                                data = mouse.create_within_as_structure(days, nmins)
                                draw_structure(ax, data)
                                set_layout(ax, xmax, num_as=len(data))
                                m += 1

                row_labels = [k for k in days_dict.keys() if k != 'acclimated']
                col_labels = [m[1] for m in group.valid_mice]
                xlabel = "AS time from onset [min]"
                ylabel = "active state counts"
                set_subplots_labels(fig, ncols, row_labels, col_labels, xlabel, ylabel, rxpad=-0.5)
                days = obs_period_to_days[experiment.name]['acclimated']
                title = "{}\nwithin active state structure\ngroup{}: {}\ndays comparison:\n{}" \
                    .format(str(experiment), group.number, group.name, days)
                add_figtitle(fig, title, ypad=-0.03)
                # plt.subplots_adjust(hspace=0.5, wspace=0.4)
                # save
                fname = '{}_within_as_structure_group{}_{}_days_comparison_{}mins'\
                    .format(experiment.name, group.number, group.name, nmins)
                save_figure(experiment, fig, res_subdir, fname)


def within_as_structure_groups(experiment, num_mins):
    if experiment.name in ['HiFat2']:
        for nmins in num_mins:
            res_subdir = os.path.join('within_as_structure')
            days_dict = OrderedDict((l, v) for l, v in obs_period_to_days[experiment.name].items()
                                    if l not in ['acclimated', 'comparison'])

            xmax = nmins if num_mins is not None else 200

            # figure
            nrows = len(days_dict)
            ncols = len(list(experiment.groups))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 16), sharex=True, sharey=True)
            for g, group in enumerate(list(experiment.groups)):
                for num, (frame, days) in enumerate(obs_period_to_days[experiment.name].iteritems()):
                    if frame not in ['acclimated', 'comparison']:
                        print "drawing within_as_structure for: {}, {} days".format(group, frame)
                        ax = axes[num][g]
                        data = group.create_within_as_structure(days)
                        draw_structure(ax, data)
                        set_layout(ax, xmax, num_as=len(data))

            row_labels = ['{}:\n{}'.format(k, v) for k, v in days_dict.items() if k != 'acclimated']
            col_labels = [g.name for g in experiment.groups]
            xlabel = "AS time from onset [min]"
            ylabel = "active state counts"
            set_subplots_labels(fig, ncols, row_labels, col_labels, xlabel, ylabel, ylpad=-0.05)
            title = "{}\nwithin active state structure\ngroups, acclimated days comparison:\n{}"\
                .format(str(experiment), group.number, group.name, obs_period_to_days[experiment.name]['acclimated'])
            add_figtitle(fig, title, ypad=-0.03)
            # plt.subplots_adjust(wspace=0.3)
            # save
            fname = '{}_within_as_structure_groups_days_comparison_{}mins' \
                .format(experiment.name, nmins)
            save_figure(experiment, fig, res_subdir, fname)