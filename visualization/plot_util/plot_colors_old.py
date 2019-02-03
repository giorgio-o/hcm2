fcols = {
    'active_state': {
        0: '#735775',  # pantone 525U
        1: '#9361B0',  # 527U
        2: '#CEA5E1',  # 529U
        3: '#E4C7EB'},  # 531U
    'inactive_state': {
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


# def get_n_colors(npos):
#     new_d = dict()
#     for key, d in fcols.iteritems():
#         new_d[key] = {k: d[x] for k, x in enumerate(npos)}
#     return new_d
#
#
# def get_feature_colors(experiment, bin_type):
#     if experiment.name == "StrainSurvey":
#         # if bin_type in ['3cycles', '12bins']:
#         ascol, fcol, dcol, lcol = [fcols[x] for x in ['active', 'food', 'drink', 'loco']]
#
#         colors = dict(ASP=ascol[1], ASN=ascol[1], ASR=ascol[1], ASD=ascol[1], TF=fcol[2], ASFI=fcol[1], FBN=fcol[3], FBR=fcol[3],
#                       ASFBR=fcol[3], FBT=fcol[3], ASFBP=fcol[3], FBS=fcol[3], FBD=fcol[3], FBI=fcol[3], TD=dcol[2],
#                       ASDI=dcol[1], DBN=dcol[3], DBR=dcol[3], ASDBR=dcol[3], DBT=dcol[3], ASDBP=dcol[3],
#                       DBS=dcol[3], DBD=dcol[3], DBI=dcol[3], TL=lcol[2], ASLI=lcol[1], LBN=lcol[3], LBR=lcol[3],
#                       ASLBR=lcol[3], LBT=lcol[3], ASLBP=lcol[3], LBS=lcol[3], LBD=lcol[3], LBI=lcol[3])
#
#     return colors, (ascol, fcol, dcol, lcol)



# def get_activity_colors(act=None):
#     if act is None:
#         return [fcols[x] for x in ['active', 'food', 'drink', 'loco']]
#     return fcols[act]

# def get_feature_colors_by_group():
#     ascol, fcol, dcol, lcol = get_activity_colors()

#     elif experiment == 'HiFat1':
#         colors = dict(ASP=ascol[0], ASR=ascol[0], ASD=ascol[0], TF=fcol[0], ASFI=fcol[1], FBN=fcol[2], FBR=fcol[2],
#                       ASFBR=fcol[2], FBT=fcol[2], ASFBP=fcol[2], FBS=fcol[2], FBD=fcol[2], FBI=fcol[2],
#                       TD=dcol[0], ASDI=dcol[1], DBN=dcol[2], DBR=dcol[2], ASDBR=dcol[2], DBT=dcol[2], ASDBP=dcol[2],
#                       DBS=dcol[2], DBD=dcol[2], DBI=dcol[2], TL=lcol[0], ASLI=lcol[1], LBN=lcol[2], LBR=lcol[2],
#                       ASLBR=lcol[2], LBT=lcol[2], ASLBP=lcol[2], LBS=lcol[2], LBD=lcol[2], LBI=lcol[2])
#
#         colors2 = {'WTHF': {'ASP': '0.3', 'ASN': '0.3', 'ASR': '0.3', 'ASD': '0.3', 'TF': '0.3', 'ASFI': '0.3',
#                              'FBN': '0.3', 'FBR': '0.3', 'ASFBR': '0.3', 'FBT': '0.3', 'ASFBP': '0.3', 'FBS': '0.3',
#                              'FBD': '0.3', 'FBI': '0.3', 'TD': '0.3', 'ASDI': '0.3', 'DBN': '0.3', 'DBR': '0.3',
#                              'ASDBR': '0.3', 'DBT': '0.3', 'ASDBP': '0.3', 'DBS': '0.3', 'DBD': '0.3',
#                              'DBI': '0.3', 'TL': '0.3', 'ASLI': '0.3', 'LBN': '0.3', 'LBR': '0.3', 'ASLBR': '0.3',
#                              'LBT': '0.3', 'ASLBP': '0.3', 'LBS': '0.3', 'LBD': '0.3', 'LBI': '0.3'},
#                    'WTLF': {'ASP': '0.7', 'ASN': '0.7', 'ASR': '0.7', 'ASD': '0.7', 'TF': '0.7', 'ASFI': '0.7',
#                              'FBN': '0.7', 'FBR': '0.7', 'ASFBR': '0.7', 'FBT': '0.7', 'ASFBP': '0.7', 'FBS': '0.7',
#                              'FBD': '0.7', 'FBI': '0.7', 'TD': '0.7', 'ASDI': '0.7', 'LBN': '0.7', 'DBN': '0.7', 'DBR': '0.7',
#                              'ASDBR': '0.7', 'DBT': '0.7', 'ASDBP': '0.7', 'DBS': '0.7', 'DBD': '0.7',
#                              'DBI': '0.7', 'TL': '0.7', 'ASLI': '0.7', 'LBR': '0.7', 'ASLBR': '0.7',
#                              'LBT': '0.7', 'ASLBP': '0.7', 'LBS': '0.7', 'LBD': '0.7', 'LBI': '0.7'},
#                    '2CHF': {'ASP': ascol[1], 'ASN': ascol[1], 'ASR': ascol[1], 'ASD': ascol[1], 'TF': fcol[1], 'ASFI': fcol[1],
#                             'FBN': fcol[1], 'FBR': fcol[1], 'ASFBR': fcol[1], 'FBT': fcol[1], 'ASFBP': fcol[1], 'FBS': fcol[1],
#                             'FBD': fcol[1], 'FBI': fcol[1], 'TD': dcol[1], 'ASDI': dcol[1], 'DBN': dcol[1], 'DBR': dcol[1],
#                             'ASDBR': dcol[1], 'DBT': dcol[1], 'ASDBP': dcol[1], 'DBS': dcol[1], 'DBD': dcol[1],
#                             'DBI': dcol[1], 'TL': lcol[1], 'ASLI': lcol[1], 'LBN': lcol[1], 'LBR': lcol[1], 'ASLBR': lcol[1],
#                             'LBT': lcol[1], 'ASLBP': lcol[1], 'LBS': lcol[1], 'LBD': lcol[1], 'LBI': lcol[1]},
#                    '2CLF': {'ASP': ascol[2], 'ASN': ascol[2], 'ASR': ascol[2], 'ASD': ascol[2], 'TF': fcol[3], 'ASFI': fcol[3],
#                              'FBN': fcol[3], 'FBR': fcol[3], 'ASFBR': fcol[3], 'FBT': fcol[3], 'ASFBP': fcol[3], 'FBS': fcol[3],
#                              'FBD': fcol[3], 'FBI': fcol[3], 'TD': dcol[3], 'ASDI': dcol[3], 'DBN': dcol[3], 'DBR': dcol[3],
#                              'ASDBR': dcol[3], 'DBT': dcol[3], 'ASDBP': dcol[3], 'DBS': dcol[3], 'DBD': dcol[3],
#                              'DBI': dcol[3], 'TL': lcol[3], 'ASLI': lcol[3], 'LBN': lcol[3], 'LBR': lcol[3], 'ASLBR': lcol[3],
#                              'LBT': lcol[3], 'ASLBP': lcol[3], 'LBS': lcol[3], 'LBD': lcol[3], 'LBI': lcol[3]}}
#
#     return colors, colors2

# colors, colors2 = get_feature_colors_by_group()
