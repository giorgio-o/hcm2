# hcm/core/keys.py
"""Various dictionaries:
    - groups ordered
    - Strain Survey experiment full group names
    - observation period to days numbers and viceversa
    - keys for raw data preprocessing
    - activity types
    - feature keys
    - activity to activity label
    - seaborn confidence intervals
    - cycles keys and timebins
    - active state code lookup
    - error codes map
    - homebase position map
    - errors to be removed from bouts
    
"""

from collections import OrderedDict

# experiments
groups_ordered = {'WR1': ['LFD', 'HFD', 'HFD-R'], 'WR2': ['LFD', 'HFD', 'HFD-R'],
                  '2cD1A2aCRE': ['2cD1Cre', '2cA2aCre', '2cD1-WT'],  # , '2cA2a-WT'],
                  '2cD1A2aCRE2': ['2cD1Cre', '2cA2aCre', '2cD1-WT', '2cA2a-WT'], '2CFast': ['2CWT', '2CKO'],
                  '1ASTRESS': ['WT', 'KO'], 'Stress_HCMe1r1': ['SalineIP', 'Restraint'],  # 'Crawlball',
                  'CORTTREAT': ['CO', 'CORT'], 'HiFat2': ['WT', '2CKO'], 'HiFat1': ['WTLF', 'WTHF', '2CLF', '2CHF'],
                  'StrainSurvey': ['C57BL6J', 'BALB', 'A', '129S1', 'DBA', 'C3H', 'AKR', 'SWR', 'SJL', 'FVB', 'WSB',
                                   'CZECH', 'CAST', 'JF1', 'MOLF', 'SPRET']}

SS_full_group_name = {'0': '', '1': 'C57BL/6J', '2': 'BALB/cByJ', '3': 'A/J', '4': '129S1/SvImJ', '5': 'DBA/2J',
                      '6': 'C3H/HeJ', '7': 'AKR/J', '8': 'SWR/J', '9': 'SJL/J', '10': 'FVB/NJ', '11': 'WSB/Ei',
                      '12': 'CZECHII/Ei', '13': 'CAST/Ei', '14': 'JF1/Ms', '15': 'MOLF/Ei', '16': 'SPRET/Ei'}

obs_period_to_days = dict(WR1=OrderedDict([('acclimated', range(10, 29 + 1))]),
                          WR2=OrderedDict([('acclimated', range(5, 19 + 1))]), HiFat2=OrderedDict(
        [('chow', range(5, 12 + 1)), ('transition', [13, 14, 15, 16, 17]), ('baseline', range(21, 28 + 1)),
         ('acclimated', range(5, 28 + 1)), ('comparison', range(5, 13) + range(13, 18) + range(21, 29))]),
                          HiFat1=OrderedDict([('acclimated', range(5, 16 + 1))]),
                          StrainSurvey=OrderedDict([('acclimated', range(5, 16 + 1))]))

# binaries
preprocessing_keys = ['_light_data', '_photobeam_data', '_lickometer_data', 'recording_start_stop_time',
                      '_movement_data', '_delta_x', '_delta_y', 'uncorrected_t', 'uncorrected_x', 'uncorrected_y', 't',
                      'x', 'y', '_nest_loc_x', 'nest_loc_y', 'start_bodyweight_grams', 'end_bodyweight_grams',
                      'bw_gain_grams', 'food_consumed_grams', 'liquid_consumed_grams']

# features
acts = ['AS', 'F', 'D', 'L']
all_feature_keys = dict(AS=['ASP', 'ASN', 'ASR', 'ASD'],
                        F=['TF', 'ASFI', 'FBN', 'FBR', 'ASFBR', 'FBT', 'ASFBP', 'FBS', 'FBD', 'FBI'],
                        D=['TD', 'ASDI', 'DBN', 'DBR', 'ASDBR', 'DBT', 'ASDBP', 'DBS', 'DBD', 'DBI'],
                        L=['TL', 'ASLI', 'LBN', 'LBR', 'ASLBR', 'LBT', 'ASLBP', 'LBS', 'LBD', 'LBI'])

all_features = [f for act in acts for f in all_feature_keys[act]]

feature_keys_short = dict(AS=['ASP', 'ASR', 'ASD'], F=['TF', 'ASFI', 'ASFBR', 'FBS', 'FBD', 'FBI'],
                          D=['TD', 'ASDI', 'ASDBR', 'DBS', 'DBD', 'DBI', ],
                          L=['TL', 'ASLI', 'ASLBR', 'LBS', 'LBD', 'LBI'])

features_short = [f for act in acts for f in feature_keys_short[act]]

feature_units = dict(ASP='[-]', ASR='[1/hr]', ASD='[min]', TF='[g]', ASFI='[mg/s]', FBN='[-]', FBR='[-]',
                     ASFBR='[1/AS.hr]', FBT='[s]', ASFBP='[-]', FBS='[mg]', FBD='[s]', FBI='[mg/s]', TD='[g]',
                     ASDI='[mg/s]', DBN='[-]', DBR='[-]', ASDBR='[1/AS.hr]', DBT='[s]', ASDBP='[-]', DBS='[mg]',
                     DBD='[s]', DBI='[mg/s]', TL='[m]', ASLI='[cm/s]', LBN='[-]', LBR='[-]', ASLBR='[1/AS.hr]',
                     LBT='[s]', ASLBP='[-]', LBS='[cm]', LBD='[s]', LBI='[cm/s]')

feature_keys_bt_distr = ['bout_counts', 'bout_duration', 'bout_interpause', 'bout_size', 'bout_intensity',
                         'bout_probability']
feature_keys_as_distr = ['active_state_counts', 'active_state_duration', 'active_state_interpause',
                         'active_state_probability']

# text
act_to_actlabel = dict(AS='active state', IS='inactive state', F='feeding', D='drinking', L='locomotion', O='other')

# plotting
to_seaborn_ci = dict(sd='sd', sem=68, ci95=95)

# binning
cycle_keys = ['24H', 'DC', 'LC', 'AS24H', 'ASDC', 'ASLC', 'IS']
#
# bin_num_to_cycle = {1: '24H', 2: 'DC', 3: 'LC', 4: 'AS24H', 5: 'ASDC', 6: 'ASLC', 7: 'IS'}

cycle_timebins = {'3cycles': cycle_keys[:3], '6cycles': cycle_keys[:-1], '7cycles': cycle_keys,
                  '4bins': ['bin{}'.format(x) for x in range(1, 4 + 1)],
                  '12bins': ['bin{}'.format(x) for x in range(1, 12 + 1)],
                  '24bins': ['bin{}'.format(x) for x in range(1, 24 + 1)], 'DC2': ['bin1', 'bin2']}

# bin12_keys = ['bin{}'.format(x) for x in range(1, 12+1)]
# binDC2_keys = ['bin1', 'bin2']
# bin4_keys = ['bin{}'.format(x) for x in range(1, 4+1)]
# bin24_keys = ['bin{}'.format(x) for x in range(1, 24+1)]

AS_code_lookup = {'1': {'11': 'LC', '22': 'DC', '12': 'LC -> DC', '21': 'DC -> LC', '13': 'LC -> LC'},
                  '2': {'11': 'LC 2', '12': 'LC 2 -> DC 1', '13': 'LC 2 -> DC 2', '14': 'LC 2 -> LC 1', '22': 'DC 1',
                        '23': 'DC 1 -> DC 2', '24': 'DC 1 -> LC 1', '33': 'DC 2', '34': 'DC 2 -> LC 1', '44': 'LC 1'}}

# error codes
hcm_raw_error_code = {1: 'platform drift', 2: 'high velocity', 3: 'no events (all day)', 4: 'long event',
                      5: 'no events (long time)', 6: 'firing while not at device'}

# homebase position mapping
homebase_2dto1d = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3, (2, 0): 4, (2, 1): 5, (3, 0): 6, (3, 1): 7}
homebase_1dto2d = {val: key for key, val in homebase_2dto1d.iteritems()}
homebase_1dto_string = {(0,): 'niche', (1,): 'top-right', (2,): 'below niche', (3,): 'below top-right',
                        (4,): 'above feeder', (5,): 'above lickometer', (6,): 'feeder', (7,): 'lickometer',
                        (0, 1): 'niche and top-left', (0, 2): "niche and below", (1, 3): "top-left and below",
                        (2, 3): "mid-top quarter", (2, 4): "mid-left cage", (3, 5): "mid-right cage",
                        (4, 5): "mid-bottom quarter", (4, 6): "feeder and above", (5, 7): "lickometer and above",
                        (6, 7): "feeder and lickometer"}

# # kinda old
# check outliers from raw data
md_bout_events_remove_errors_dict = {'1ASTRESS': dict(feeding=[], drinking=[], locomotion=[]),
                                     'Stress_HCMe1r1': dict(feeding=[], drinking=[], locomotion=[]),
                                     'CORTTREAT': dict(feeding=[], drinking=[('CO', 'M1', 7),
                                                                             # homebase at lick. short (legit) AS to
                                                                             # begin the day
                                                                             ('CO', 'M1', 17),  # same
                                                                             ('CO', 'M26', 16),  # same
                                                                             ('CORT', 'M13', 17),
                                                                             # homebase a t lick, first and last (
                                                                             # legit) short AS
                                                                             ], locomotion=[('CO', 'M27', 11)]),
                                     # slice of bin. spurious. one move event before CT6
                                     'HiFat2': dict(feeding=[], drinking=[], locomotion=[('WT', 'M6693', 7)]),
                                     # last bout
                                     'HiFat1': dict(feeding=[], drinking=[],
                                                    locomotion=[('WTLF', 'M5732', 11),  # remove first loco bout
                                                                ('WTHF', 'M5926', 16),  # same
                                                                ('2CLF', 'M5678', 12)]),  # one loco bout in CT02-04
                                     'StrainSurvey': dict(feeding=[],
                                                          drinking=[('A', 'M7203', 16),  # homebase at lick, short AS
                                                                    ('AKR', 'M8407', 8),  #
                                                                    ('AKR', 'M8407', 10),
                                                                    # starts last day outside hb (niche), with short
                                                                    # drink
                                                                    ('FVB', 'M2110', 16),
                                                                    # ends last day outside hb (niche), with short drink
                                                                    ('FVB', 'M2210', 16),  # same
                                                                    ('WSB', 'M2111', 16),  # same
                                                                    ('CZECH', 'M2112', 16),  # same
                                                                    ('CZECH', 'M2212', 16),  # same
                                                                    ('MOLF', 'M5215', 7)],
                                                          locomotion=[('129S1', 'M8204', 10),  # remove first loco bout
                                                                      ('DBA', 'M2105', 5),  # 5 loco bouts at the end
                                                                      ('DBA', 'M5205', 13),  # # first loco bout
                                                                      ('AKR', 'M2107', 5),  # 4 loco bouts in CT04-06
                                                                      ('SJL', 'M4209', 7),  # first loco bout
                                                                      ('SJL', 'M4209', 8),  # same
                                                                      ('FVB', 'M2110', 16),  # last loco bout
                                                                      ('FVB', 'M2210', 16),  # same
                                                                      ('WSB', 'M2111', 16),  # same
                                                                      ('MOLF', 'M5215', 7)]),  # first loco bout
                                     }

# # kinda old
# days_act_to_actlabel = {
#     '2cD1A2aCRE': {tuple(range(5, 31 + 1)): 'acclimated', (6, 7, 8, 9, 10): 'baseline',
#                    tuple(range(6, 10 + 1)): 'test1', },
#     '2cD1A2aCRE2': {tuple(range(5, 31 + 1)): 'acclimated', (13, 14, 15, 16, 17): 'baseline',
#                     tuple(range(6, 17 + 1)): 'test', tuple(range(13, 17 + 1)): 'test2', },
#     '2CFast': {tuple(range(5, 11 + 1)): 'chow', (12): 'fast', tuple(range(13, 18 + 1)): 'refeed',
#                tuple(range(5, 18 + 1)): 'acclimated', },
#     '1ASTRESS': {tuple(range(6, 13 + 1)): 'baseline1', tuple(range(21, 24 + 1)): 'baseline-post-EZM',
#                  tuple(range(25, 31 + 1)): 'baseline2', (23, 24, 25, 26): 'stressor',
#                  tuple(range(6, 31 + 1)): 'acclimated', tuple(range(6, 13 + 1) + range(21, 31 + 1)): 'test'},
#     'Stress_HCMe1r1': {(23, 25): 'two', (25,): 'stressor', tuple(range(4, 31 + 1)): 'acclimated',
#                        tuple(range(16, 25 + 1)): 'test'},
#     'CORTTREAT': {(4, 5, 6, 7): 'baseline_pre-cort', (8, 9, 10, 11, 12, 13, 14): 'transition_post-cort',
#                   (15, 16, 17, 18, 19, 20): 'baseline_post-cort', tuple(range(8, 20 + 1)): 'cort_treatment',
#                   (21,): 'novel_object', (23,): 'zero-maze_run', tuple(range(4, 20 + 1)): 'acclimated', },
#     'HiFat2': {tuple(range(5, 12 + 1)): 'chow', (13, 14, 15, 16, 17): 'post-diet',
#                tuple(range(18, 28 + 1)): 'post-diet-baseline', tuple(range(5, 28 + 1)): 'acclimated'},
#     'HiFat1': {tuple(range(5, 16 + 1)): 'acclimated'},  # , tuple(range(5, 9)): 'test1'},
#     'StrainSurvey': {tuple(range(5, 16 + 1)): 'acclimated', (5, 6, 7): 'test1'}}

# obs_period_to_days = dict((key, dict((v, k) for k, v in val.iteritems()))
#                           for key, val in days_act_to_actlabel.iteritems())
