# _base_ = [
#     '../_base_/models/hv_pointpillars_fpn_nus.py',
#     '../_base_/datasets/nus-3d-3class.py', '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py'
# ]

#3Class on nuscenes
# _base_ = [
#     '../_base_/models/hv_pointpillars_fpn_nus_3class.py',
#     '../_base_/datasets/nus-3d-3class.py', 
#     '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py'
# ]

#3class Eval on KITTI
# _base_ = [
#     '../_base_/models/hv_pointpillars_fpn_nus_3class.py',
#     '../_base_/datasets/kitti-3d-3class.py', 
#     '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py'
# ]

# 3 Class Eval on Waymo
_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus_3class.py',
    '../_base_/datasets/waymoD5-3d-3class.py', 
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
