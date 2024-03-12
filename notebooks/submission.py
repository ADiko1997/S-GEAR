from utils import *


CFG_FILES = [
    #TSN + AVT-h (train and train+val models)
    # ('expts/01_ek100_avt_TS_384_tsn.txt', 0),
    # ('expts/01_ek100_avt_TS_384_tsn_trainval.txt', 0),
    # ('expts/01_ek100_avt_TS_384_tsn_test_trainval.txt', 0),
    #irCSN152/IG65M + AVT-h
    # ('expts/01_ek100_avt_TS_384_igm.txt', 0),
    # ('expts/01_ek100_avt_TS_384_igm_trainval.txt', 0),
    # ('expts/01_ek100_avt_TS_384_igm_test_trainval.txt', 0),
    #AVT
    ('expts/01_ek100_SGEAR.txt', 0),
    ('expts/01_ek100_SGEAR_test_trainval.txt', 0),
    #obj AVT
    # ('expts/01_ek100_avt_TS_384_obj.txt', 0),
    # ('expts/01_ek100_avt_TS_384_obj_trainval.txt', 0),
    # ('expts/01_ek100_avt_TS_384_obj_test_trainval.txt', 0),
    # Longer AVT
    # ('expts/01_ek100_avt_TS_384.txt', 0),
    ('expts/01_ek100_avt_TS_384_test_trainval.txt', 0),
    ('expts/01_ek100_avt_TS_384_submission_2_.txt', 0),
    ('expts/01_ek100_avt_TS_384_submission_2.txt', 0),
    ('expts/01_ek100_avt_TS_384_submission.txt', 0),


]
WTS = [
       # TSN feats
    #    0.5, 1.0,
       # irCSN152/IG65M feats from AVT
    #    0.5, 1.0,
       # vit-224 S-GEAR
       1.0, 1.5,
       #obj tsn feats from RULSTM
    #    0.5, 1.0,
       # vit-384 S-GEAR
       2.0, 1.5, 1.5, 1.5]
SLS = [1, 3, 4]

package_results_for_submission_ek100(CFG_FILES, WTS, SLS)


# # Path: notebooks/submission.py

#Best RGB

# CFG_FILES = [
#     # TSN + AVT-h (train and train+val models)
#     ('expts/01_ek100_avt_TS_384_tsn.txt', 0),
#     ('expts/01_ek100_avt_TS_384_tsn_trainval.txt', 0),
#     ('expts/01_ek100_avt_TS_384_tsn_test_trainval.txt', 0),
#     # irCSN152/IG65M + AVT-h
#     ('expts/01_ek100_avt_TS_384_igm.txt', 0),
#     ('expts/01_ek100_avt_TS_384_igm_trainval.txt', 0),
#     ('expts/01_ek100_avt_TS_384_igm_test_trainval.txt', 0),
#     # AVT
#     ('expts/01_ek100_SGEAR.txt', 0),
#     ('expts/01_ek100_SGEAR_test_trainval.txt', 0),
#     # # Flow, obj AVT
#     # ('expts/01_ek100_avt_TS_384_obj.txt', 0),
#     # ('expts/01_ek100_avt_TS_384_flow.txt', 0),
#     # ('expts/01_ek100_avt_TS_384_obj_trainval.txt', 0),
#     # ('expts/01_ek100_avt_TS_384_flow_trainval.txt', 0),
#     # Longer AVT
#     ('expts/01_ek100_avt_TS_384.txt', 0),
#     ('expts/01_ek100_avt_TS_384_test_trainval.txt', 0),
#     ('expts/01_ek100_avt_TS_384_submission_2_.txt', 0),
#     ('expts/01_ek100_avt_TS_384_submission_2.txt', 0),
#     ('expts/01_ek100_avt_TS_384_submission.txt', 0),


# ]
# WTS = [
#        # TSN feats
#        0.5, 0.5, 1.0,
#        # irCSN152/IG65M feats from AVT
#        0.5, 0.5, 1.0,
#        # vit-224 S-GEAR
#        0.5, 1.0,
#        # Flow, obj tsn feats from RULSTM
#     #    0.5, 0.5, 0.5, 0.5,
#        # vit-384 S-GEAR
#        1.5, 2.0, 1.5, 1.5, 1.5]
# SLS = [1, 3, 4]

# package_results_for_submission_ek100(CFG_FILES, WTS, SLS)