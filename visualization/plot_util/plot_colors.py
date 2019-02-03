# hcm/visualization/plot_util/plot_colors.py
""" plotting colors """
import seaborn as sns

colors = {'active state': 'BuPu_d', 'inactive state': 'Greys_d', 'feeding': 'Oranges_d', 'drinking': 'Blues_d',
          'locomotion': 'BuGn_d', 'other': 'YlOrRd_d'}

fcolors = {
    'active state': {
        0: '#735775',  # pantone 525U
        1: '#9361B0',  # 527U
        2: '#CEA5E1',  # 529U
        3: '#E4C7EB'},  # 531U
    'inactive state': {
        0: '.6',
        1: '0.4',
        2: '0.2',
        3: '0'},
    'feeding': {
        0: '#A6813D',  # 125U
        1: '#D79133',  # 124U
        2: '#FFAC2A',  # 123U
        3: '#FFCC52'},  # 121U

    'drinking': {
        0: '#3F5666',  # 303U
        1: '#28628E',  # 301U
        2: '#1295D8',  # 299U
        3: '#7ECCEE'},  # 297U

    'locomotion': {
        0: '#5B794E',  # 364U
        1: '#56944F',  # 362U
        2: '#5DB860',  # 360U
        3: '#97D88A',  # 358U
        4: '#A8AA31',  # 397U
        5: '#DBE200'},  # 396U

    'other': {
        0: '#DD5061',  # 199U, dark red
        1: '#700C13',  # dark red
        2: '.5',  # mid-grey
        3: '#B02B0A'}}  # light dark red


def get_features_palette(ftype, num_items):
    return sns.color_palette(colors[ftype], num_items)