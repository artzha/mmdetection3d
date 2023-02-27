# Copyright (c) AMRL. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

from glob import glob
from os.path import join

import mmcv
import numpy as np
import tensorflow as tf


class CODa2KITTI(object):
    """CODa to KITTI converter.
    This class serves as the converter to change the coda raw data to KITTI
    format.
    Args:
        load_dir (str): Directory to load CODa raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_coda_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            'CENTER'
        ]
        self.type_list = [
            'BIKE',
            'VEHICLE', 
            'PERSON', 
            'TREE', 
            'POLE',  
            'CHAIR', 
            'TABLE', 
            'OTHER' 
        ]
        self.coda_to_kitti_class_map = {
            'Other': 'DontCare',
            'PERSON': 'Pedestrian',
            'VEHICLE': 'Car',
            'BIKE': 'Cyclist',
            'TREE': 'Tree',
            'POLE': 'Pole',
            'CHAIR': 'Chair',
            'TABLE': 'Table'
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.
        Args:
            file_idx (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            self.save_image(frame, file_idx, frame_idx)
            self.save_calib(frame, file_idx, frame_idx)
            self.save_lidar(frame, file_idx, frame_idx)
            self.save_pose(frame, file_idx, frame_idx)
            self.save_timestamp(frame, file_idx, frame_idx)

            if not self.test_mode:
                self.save_label(frame, file_idx, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format. Jpg is the original format
        used by Waymo Open dataset. Saving in png format will cause huge (~3x)
        unnesssary storage waste.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img.image)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        # First return
        points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, mask_indices))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def save_timestamp(self, frame, file_idx, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(
                join(f'{self.timestamp_save_dir}/{self.prefix}' +
                     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
                'w') as f:
            f.write(str(frame.timestamp_micros))

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir,
                self.timestamp_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.timestamp_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.
        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.
        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.
        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.
        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return 

# # Copyright (c) OpenMMLab. All rights reserved.
# import os
# from collections import OrderedDict
# from os import path as osp
# from typing import List, Tuple, Union

# import mmcv
# import numpy as np
# from pyquaternion import Quaternion
# from shapely.geometry import MultiPoint, box

# from mmdet3d.core.bbox import points_cam2img
# from mmdet3d.datasets import CODataset

# nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
#                   'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
#                   'barrier')

# nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
#                   'pedestrian.moving', 'pedestrian.standing',
#                   'pedestrian.sitting_lying_down', 'vehicle.moving',
#                   'vehicle.parked', 'vehicle.stopped', 'None')


# def create_coda_infos(    root_path,
#                           info_prefix,
#                           version='train',
#                           max_sweeps=10):
#     """Create info file of nuscene dataset.

#     Given the raw data, generate its related info file in pkl format.

#     Args:
#         root_path (str): Path of the data root.
#         info_prefix (str): Prefix of the info file to be generated.
#         version (str, optional): Version of the data.
#             Default: 'v1.0-trainval'.
#         max_sweeps (int, optional): Max number of sweeps.
#             Default: 10.
#     """
#     available_vers = ['train', 'val', 'test']
#     assert version in available_vers

#     metadata_path = osp.join(root_path, 'coda_format/calibrations')
#     metadata_files = [json_file for json_file in os.listdir(metadata_path) if osp.isfile(osp.join(metadata_path, json_file))]
#     import json

#     train_scenes = []
#     val_scenes = []
#     test_scenes = []
#     for mfilename in metadata_files:
#         mpath = osp.join(metadata_path, mfilename)
#         mfile = open(mfilename, 'r')
#         mfile_json = json.load(mfile)

#         if version in mfile_json['ObjectTracking'].keys():
#             if version=='train':
#                 train_scenes.extend(mfile_json['ObjectTracking'][version])
#             elif version=='test':
#                 test_scenes.extend(mfile_json['ObjectTracking'][version])
#             elif version=='val':
#                 val_scenes.extend(mfile_json['ObjectTracking'][version])
#             else:
#                 raise ValueError('unknown')

#     test = 'test' in version
#     if test:
#         print('test scene: {}'.format(len(train_scenes)))
#     else:
#         print('train scene: {}, val scene: {}'.format(
#             len(train_scenes), len(val_scenes)))

#     train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
#         train_scenes, val_scenes, test, max_sweeps=max_sweeps)

#     metadata = dict(version=version)
#     if test:
#         print('test sample: {}'.format(len(train_nusc_infos)))
#         data = dict(infos=train_nusc_infos, metadata=metadata)
#         info_path = osp.join(root_path,
#                              '{}_infos_test.pkl'.format(info_prefix))
#         mmcv.dump(data, info_path)
#     else:
#         print('train sample: {}, val sample: {}'.format(
#             len(train_nusc_infos), len(val_nusc_infos)))
#         data = dict(infos=train_nusc_infos, metadata=metadata)
#         info_path = osp.join(root_path,
#                              '{}_infos_train.pkl'.format(info_prefix))
#         mmcv.dump(data, info_path)
#         data['infos'] = val_nusc_infos
#         info_val_path = osp.join(root_path,
#                                  '{}_infos_val.pkl'.format(info_prefix))
#         mmcv.dump(data, info_val_path)


# def get_available_scenes(nusc):
#     """Get available scenes from the input nuscenes class.

#     Given the raw data, get the information of available scenes for
#     further info generation.

#     Args:
#         nusc (class): Dataset class in the nuScenes dataset.

#     Returns:
#         available_scenes (list[dict]): List of basic information for the
#             available scenes.
#     """
#     available_scenes = []
#     print('total scene num: {}'.format(len(nusc.scene)))
#     for scene in nusc.scene:
#         scene_token = scene['token']
#         scene_rec = nusc.get('scene', scene_token)
#         sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
#         sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
#         has_more_frames = True
#         scene_not_exist = False
#         while has_more_frames:
#             lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
#             lidar_path = str(lidar_path)
#             if os.getcwd() in lidar_path:
#                 # path from lyftdataset is absolute path
#                 lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
#                 # relative path
#             if not mmcv.is_filepath(lidar_path):
#                 scene_not_exist = True
#                 break
#             else:
#                 break
#         if scene_not_exist:
#             continue
#         available_scenes.append(scene)
#     print('exist scene num: {}'.format(len(available_scenes)))
#     return available_scenes


# def _fill_trainval_infos(
#                          train_scenes,
#                          val_scenes,
#                          test=False,
#                          max_sweeps=10):
#     """Generate the train/val infos from the raw data.

#     Args:
#         coda (:obj:`coda`): Dataset class in the CODa dataset.
#         train_scenes (list[str]): Basic information of training scenes.
#         val_scenes (list[str]): Basic information of validation scenes.
#         test (bool, optional): Whether use the test mode. In test mode, no
#             annotations can be accessed. Default: False.
#         max_sweeps (int, optional): Max number of sweeps. Default: 10.

#     Returns:
#         tuple[list[dict]]: Information of training set and validation set
#             that will be saved to the info file.
#     """
#     train_nusc_infos = []
#     val_nusc_infos = []

#     for sample in mmcv.track_iter_progress(nusc.sample):
#         lidar_token = sample['data']['LIDAR_TOP']
#         sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
#         cs_record = nusc.get('calibrated_sensor',
#                              sd_rec['calibrated_sensor_token'])
#         pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#         lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

#         mmcv.check_file_exist(lidar_path)

#         info = {
#             'lidar_path': lidar_path,
#             'token': sample['token'],
#             'sweeps': [],
#             'cams': dict(),
#             'lidar2ego_translation': cs_record['translation'],
#             'lidar2ego_rotation': cs_record['rotation'],
#             'ego2global_translation': pose_record['translation'],
#             'ego2global_rotation': pose_record['rotation'],
#             'timestamp': sample['timestamp'],
#         }

#         l2e_r = info['lidar2ego_rotation']
#         l2e_t = info['lidar2ego_translation']
#         e2g_r = info['ego2global_rotation']
#         e2g_t = info['ego2global_translation']
#         l2e_r_mat = Quaternion(l2e_r).rotation_matrix
#         e2g_r_mat = Quaternion(e2g_r).rotation_matrix

#         # obtain 6 image's information per frame
#         camera_types = [
#             'CAM_FRONT',
#             'CAM_FRONT_RIGHT',
#             'CAM_FRONT_LEFT',
#             'CAM_BACK',
#             'CAM_BACK_LEFT',
#             'CAM_BACK_RIGHT',
#         ]
#         for cam in camera_types:
#             cam_token = sample['data'][cam]
#             cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
#             cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
#                                          e2g_t, e2g_r_mat, cam)
#             cam_info.update(cam_intrinsic=cam_intrinsic)
#             info['cams'].update({cam: cam_info})

#         # obtain sweeps for a single key-frame
#         sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
#         sweeps = []
#         while len(sweeps) < max_sweeps:
#             if not sd_rec['prev'] == '':
#                 sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
#                                           l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
#                 sweeps.append(sweep)
#                 sd_rec = nusc.get('sample_data', sd_rec['prev'])
#             else:
#                 break
#         info['sweeps'] = sweeps
#         # obtain annotation
#         if not test:
#             annotations = [
#                 nusc.get('sample_annotation', token)
#                 for token in sample['anns']
#             ]
#             locs = np.array([b.center for b in boxes]).reshape(-1, 3)
#             dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
#             rots = np.array([b.orientation.yaw_pitch_roll[0]
#                              for b in boxes]).reshape(-1, 1)
#             velocity = np.array(
#                 [nusc.box_velocity(token)[:2] for token in sample['anns']])
#             valid_flag = np.array(
#                 [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
#                  for anno in annotations],
#                 dtype=bool).reshape(-1)
#             # convert velo from global to lidar
#             for i in range(len(boxes)):
#                 velo = np.array([*velocity[i], 0.0])
#                 velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
#                     l2e_r_mat).T
#                 velocity[i] = velo[:2]

#             names = [b.name for b in boxes]
#             for i in range(len(names)):
#                 if names[i] in NuScenesDataset.NameMapping:
#                     names[i] = NuScenesDataset.NameMapping[names[i]]
#             names = np.array(names)
#             # we need to convert box size to
#             # the format of our lidar coordinate system
#             # which is x_size, y_size, z_size (corresponding to l, w, h)
#             gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
#             assert len(gt_boxes) == len(
#                 annotations), f'{len(gt_boxes)}, {len(annotations)}'
#             info['gt_boxes'] = gt_boxes
#             info['gt_names'] = names
#             info['gt_velocity'] = velocity.reshape(-1, 2)
#             info['num_lidar_pts'] = np.array(
#                 [a['num_lidar_pts'] for a in annotations])
#             info['num_radar_pts'] = np.array(
#                 [a['num_radar_pts'] for a in annotations])
#             info['valid_flag'] = valid_flag

#         if sample['scene_token'] in train_scenes:
#             train_nusc_infos.append(info)
#         else:
#             val_nusc_infos.append(info)

#     return train_nusc_infos, val_nusc_infos


# def obtain_sensor2top(nusc,
#                       sensor_token,
#                       l2e_t,
#                       l2e_r_mat,
#                       e2g_t,
#                       e2g_r_mat,
#                       sensor_type='lidar'):
#     """Obtain the info with RT matric from general sensor to Top LiDAR.

#     Args:
#         nusc (class): Dataset class in the nuScenes dataset.
#         sensor_token (str): Sample data token corresponding to the
#             specific sensor type.
#         l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
#         l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
#             in shape (3, 3).
#         e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
#         e2g_r_mat (np.ndarray): Rotation matrix from ego to global
#             in shape (3, 3).
#         sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

#     Returns:
#         sweep (dict): Sweep information after transformation.
#     """
#     sd_rec = nusc.get('sample_data', sensor_token)
#     cs_record = nusc.get('calibrated_sensor',
#                          sd_rec['calibrated_sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#     data_path = str(nusc.get_sample_data_path(sd_rec['token']))
#     if os.getcwd() in data_path:  # path from lyftdataset is absolute path
#         data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
#     sweep = {
#         'data_path': data_path,
#         'type': sensor_type,
#         'sample_data_token': sd_rec['token'],
#         'sensor2ego_translation': cs_record['translation'],
#         'sensor2ego_rotation': cs_record['rotation'],
#         'ego2global_translation': pose_record['translation'],
#         'ego2global_rotation': pose_record['rotation'],
#         'timestamp': sd_rec['timestamp']
#     }
#     l2e_r_s = sweep['sensor2ego_rotation']
#     l2e_t_s = sweep['sensor2ego_translation']
#     e2g_r_s = sweep['ego2global_rotation']
#     e2g_t_s = sweep['ego2global_translation']

#     # obtain the RT from sensor to Top LiDAR
#     # sweep->ego->global->ego'->lidar
#     l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
#     e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
#     R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
#         np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#     T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
#         np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
#     T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
#                   ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
#     sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
#     sweep['sensor2lidar_translation'] = T
#     return sweep


# def export_2d_annotation(root_path, info_path, version, mono3d=True):
#     """Export 2d annotation from the info file and raw data.

#     Args:
#         root_path (str): Root path of the raw data.
#         info_path (str): Path of the info file.
#         version (str): Dataset version.
#         mono3d (bool, optional): Whether to export mono3d annotation.
#             Default: True.
#     """
#     # get bbox annotations for camera
#     camera_types = [
#         'CAM_FRONT',
#         'CAM_FRONT_RIGHT',
#         'CAM_FRONT_LEFT',
#         'CAM_BACK',
#         'CAM_BACK_LEFT',
#         'CAM_BACK_RIGHT',
#     ]
#     nusc_infos = mmcv.load(info_path)['infos']
#     nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
#     # info_2d_list = []
#     cat2Ids = [
#         dict(id=nus_categories.index(cat_name), name=cat_name)
#         for cat_name in nus_categories
#     ]
#     coco_ann_id = 0
#     coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
#     for info in mmcv.track_iter_progress(nusc_infos):
#         for cam in camera_types:
#             cam_info = info['cams'][cam]
#             coco_infos = get_2d_boxes(
#                 nusc,
#                 cam_info['sample_data_token'],
#                 visibilities=['', '1', '2', '3', '4'],
#                 mono3d=mono3d)
#             (height, width, _) = mmcv.imread(cam_info['data_path']).shape
#             coco_2d_dict['images'].append(
#                 dict(
#                     file_name=cam_info['data_path'].split('data/nuscenes/')
#                     [-1],
#                     id=cam_info['sample_data_token'],
#                     token=info['token'],
#                     cam2ego_rotation=cam_info['sensor2ego_rotation'],
#                     cam2ego_translation=cam_info['sensor2ego_translation'],
#                     ego2global_rotation=info['ego2global_rotation'],
#                     ego2global_translation=info['ego2global_translation'],
#                     cam_intrinsic=cam_info['cam_intrinsic'],
#                     width=width,
#                     height=height))
#             for coco_info in coco_infos:
#                 if coco_info is None:
#                     continue
#                 # add an empty key for coco format
#                 coco_info['segmentation'] = []
#                 coco_info['id'] = coco_ann_id
#                 coco_2d_dict['annotations'].append(coco_info)
#                 coco_ann_id += 1
#     if mono3d:
#         json_prefix = f'{info_path[:-4]}_mono3d'
#     else:
#         json_prefix = f'{info_path[:-4]}'
#     mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


# def get_2d_boxes(nusc,
#                  sample_data_token: str,
#                  visibilities: List[str],
#                  mono3d=True):
#     """Get the 2D annotation records for a given `sample_data_token`.

#     Args:
#         sample_data_token (str): Sample data token belonging to a camera
#             keyframe.
#         visibilities (list[str]): Visibility filter.
#         mono3d (bool): Whether to get boxes with mono3d annotation.

#     Return:
#         list[dict]: List of 2D annotation record that belongs to the input
#             `sample_data_token`.
#     """

#     # Get the sample data and the sample corresponding to that sample data.
#     sd_rec = nusc.get('sample_data', sample_data_token)

#     assert sd_rec[
#         'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
#         ' for camera sample_data!'
#     if not sd_rec['is_key_frame']:
#         raise ValueError(
#             'The 2D re-projections are available only for keyframes.')

#     s_rec = nusc.get('sample', sd_rec['sample_token'])

#     # Get the calibrated sensor and ego pose
#     # record to get the transformation matrices.
#     cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
#     pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
#     camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

#     # Get all the annotation with the specified visibilties.
#     ann_recs = [
#         nusc.get('sample_annotation', token) for token in s_rec['anns']
#     ]
#     ann_recs = [
#         ann_rec for ann_rec in ann_recs
#         if (ann_rec['visibility_token'] in visibilities)
#     ]

#     repro_recs = []

#     for ann_rec in ann_recs:
#         # Augment sample_annotation with token information.
#         ann_rec['sample_annotation_token'] = ann_rec['token']
#         ann_rec['sample_data_token'] = sample_data_token

#         # Get the box in global coordinates.
#         box = nusc.get_box(ann_rec['token'])

#         # Move them to the ego-pose frame.
#         box.translate(-np.array(pose_rec['translation']))
#         box.rotate(Quaternion(pose_rec['rotation']).inverse)

#         # Move them to the calibrated sensor frame.
#         box.translate(-np.array(cs_rec['translation']))
#         box.rotate(Quaternion(cs_rec['rotation']).inverse)

#         # Filter out the corners that are not in front of the calibrated
#         # sensor.
#         corners_3d = box.corners()
#         in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
#         corners_3d = corners_3d[:, in_front]

#         # Project 3d box to 2d.
#         corner_coords = view_points(corners_3d, camera_intrinsic,
#                                     True).T[:, :2].tolist()

#         # Keep only corners that fall within the image.
#         final_coords = post_process_coords(corner_coords)

#         # Skip if the convex hull of the re-projected corners
#         # does not intersect the image canvas.
#         if final_coords is None:
#             continue
#         else:
#             min_x, min_y, max_x, max_y = final_coords

#         # Generate dictionary record to be included in the .json file.
#         repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
#                                     sample_data_token, sd_rec['filename'])

#         # If mono3d=True, add 3D annotations in camera coordinates
#         if mono3d and (repro_rec is not None):
#             loc = box.center.tolist()

#             dim = box.wlh
#             dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
#             dim = dim.tolist()

#             rot = box.orientation.yaw_pitch_roll[0]
#             rot = [-rot]  # convert the rot to our cam coordinate

#             global_velo2d = nusc.box_velocity(box.token)[:2]
#             global_velo3d = np.array([*global_velo2d, 0.0])
#             e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
#             c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
#             cam_velo3d = global_velo3d @ np.linalg.inv(
#                 e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
#             velo = cam_velo3d[0::2].tolist()

#             repro_rec['bbox_cam3d'] = loc + dim + rot
#             repro_rec['velo_cam3d'] = velo

#             center3d = np.array(loc).reshape([1, 3])
#             center2d = points_cam2img(
#                 center3d, camera_intrinsic, with_depth=True)
#             repro_rec['center2d'] = center2d.squeeze().tolist()
#             # normalized center2D + depth
#             # if samples with depth < 0 will be removed
#             if repro_rec['center2d'][2] <= 0:
#                 continue

#             ann_token = nusc.get('sample_annotation',
#                                  box.token)['attribute_tokens']
#             if len(ann_token) == 0:
#                 attr_name = 'None'
#             else:
#                 attr_name = nusc.get('attribute', ann_token[0])['name']
#             attr_id = nus_attributes.index(attr_name)
#             repro_rec['attribute_name'] = attr_name
#             repro_rec['attribute_id'] = attr_id

#         repro_recs.append(repro_rec)

#     return repro_recs


# def post_process_coords(
#     corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
# ) -> Union[Tuple[float, float, float, float], None]:
#     """Get the intersection of the convex hull of the reprojected bbox corners
#     and the image canvas, return None if no intersection.

#     Args:
#         corner_coords (list[int]): Corner coordinates of reprojected
#             bounding box.
#         imsize (tuple[int]): Size of the image canvas.

#     Return:
#         tuple [float]: Intersection of the convex hull of the 2D box
#             corners and the image canvas.
#     """
#     polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
#     img_canvas = box(0, 0, imsize[0], imsize[1])

#     if polygon_from_2d_box.intersects(img_canvas):
#         img_intersection = polygon_from_2d_box.intersection(img_canvas)
#         intersection_coords = np.array(
#             [coord for coord in img_intersection.exterior.coords])

#         min_x = min(intersection_coords[:, 0])
#         min_y = min(intersection_coords[:, 1])
#         max_x = max(intersection_coords[:, 0])
#         max_y = max(intersection_coords[:, 1])

#         return min_x, min_y, max_x, max_y
#     else:
#         return None


# def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
#                     sample_data_token: str, filename: str) -> OrderedDict:
#     """Generate one 2D annotation record given various information on top of
#     the 2D bounding box coordinates.

#     Args:
#         ann_rec (dict): Original 3d annotation record.
#         x1 (float): Minimum value of the x coordinate.
#         y1 (float): Minimum value of the y coordinate.
#         x2 (float): Maximum value of the x coordinate.
#         y2 (float): Maximum value of the y coordinate.
#         sample_data_token (str): Sample data token.
#         filename (str):The corresponding image file where the annotation
#             is present.

#     Returns:
#         dict: A sample 2D annotation record.
#             - file_name (str): file name
#             - image_id (str): sample data token
#             - area (float): 2d box area
#             - category_name (str): category name
#             - category_id (int): category id
#             - bbox (list[float]): left x, top y, dx, dy of 2d box
#             - iscrowd (int): whether the area is crowd
#     """
#     repro_rec = OrderedDict()
#     repro_rec['sample_data_token'] = sample_data_token
#     coco_rec = dict()

#     relevant_keys = [
#         'attribute_tokens',
#         'category_name',
#         'instance_token',
#         'next',
#         'num_lidar_pts',
#         'num_radar_pts',
#         'prev',
#         'sample_annotation_token',
#         'sample_data_token',
#         'visibility_token',
#     ]

#     for key, value in ann_rec.items():
#         if key in relevant_keys:
#             repro_rec[key] = value

#     repro_rec['bbox_corners'] = [x1, y1, x2, y2]
#     repro_rec['filename'] = filename

#     coco_rec['file_name'] = filename
#     coco_rec['image_id'] = sample_data_token
#     coco_rec['area'] = (y2 - y1) * (x2 - x1)

#     if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
#         return None
#     cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
#     coco_rec['category_name'] = cat_name
#     coco_rec['category_id'] = nus_categories.index(cat_name)
#     coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
#     coco_rec['iscrowd'] = 0

#     return coco_rec
