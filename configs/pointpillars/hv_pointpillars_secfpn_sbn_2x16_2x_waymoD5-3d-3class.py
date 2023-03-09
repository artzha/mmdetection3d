#WAYMO TRAIN/EVAL
# _base_ = [
#     '../_base_/models/hv_pointpillars_secfpn_waymo.py',
#     '../_base_/datasets/waymoD5-3d-3class.py', #Uncomment this to train/eval with waymo
#     '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py',
# ]

#KITTI EVAL
_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

#NUSCENES
# _base_ = [
#     '../_base_/models/hv_pointpillars_secfpn_waymo.py',
#     '../_base_/datasets/nus-3d.py',
#     '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py',
# ]