_base_ = [
    '../_base_/models/centerpoint_0075voxel_second_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

file_client_args = dict(backend='disk')

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.85, -74.85, -2, 74.85, 74.85, 4]

model = dict(
    pts_bbox_head=dict(
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4),
            init_bias=-2.19,
            final_kernel=3),
        # fpr bbox range
        bbox_coder=dict(pc_range=point_cloud_range[:2])
        ),
    test_cfg=dict(pts=dict(
        nms_type='circle',
        pc_range=point_cloud_range[:2])),
    # Model Point Ranges
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    )

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        # Add double-flip augmentation
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict( train=dict(dataset=dict(load_interval=1)),
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
