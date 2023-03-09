# _base_ = [
#     '../_base_/models/hv_pointpillars_fpn_nus.py',
#     '../_base_/datasets/nus-3d.py', '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py'
# ]

#3Class
_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus_3class.py',
    '../_base_/datasets/nus-3d-3class.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]