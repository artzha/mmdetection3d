# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
import pyquaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmcv.utils import print_log

from ..core.bbox import Box3DMode, points_cam2img
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose

CLASS_INDEX_MAP = {
    'car': 0,
    "bicycle": 1,
    'pedestrian': 2
}

@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.pcd_limit_range=[-50, -50, -5, 50, 50, 3]

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_kitti_results(results, jsonfile_prefix)
            from mmdet3d.core.evaluation import kitti_eval

            #Format nuscenes annotation format to kitti style
            kitti_data_infos = self.format_kitti_data_infos_annos(self.data_infos)
            gt_annos = kitti_data_infos
            import pdb; pdb.set_trace()
            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)

            return ap_dict
        else:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files)

            if tmp_dir is not None:
                tmp_dir.cleanup()

            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)

            return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
            
    """
    ADDED TO FORMAT NUSCENES TO KITTI
    """
    def format_kitti_data_infos_annos(self, data_infos, data_format='kitti'):
        annos_dict = []
        # Contents of each entry in annos_dict
        # {
        #     'name': [],         #name
        #     'truncated': [],    #truneated  (0 is assummed)
        #     'occluded': [],     #occluded (not available assummed to be clear)
        #     'alpha': [],
        #     'bbox': [],         # 2D bbox
        #     'dimensions': [],
        #     'location': [], 
        #     'rotation_y': [],
        #     'score': [],
        #     'index': [],
        #     'difficulty': [],
        #     'num_points_in_gt': []
        # }

        """
        Refer to the code snippet below for converting bboxes from nusc to kitti format
        https://github.com/cxy1997/3D_adapt_auto_driving/blob/8508e0bb98142c339898a14a7adef557a921b7ba/convert/nusc2kitti.py#L328
        
        The relation between nuscenes and kitti coordinates is note worthy:
            1. The coordinate conventino for nuscenes is  
        """
        imsize = (1600, 900)
        for sample_idx, infos in enumerate(data_infos):
            num_samples = len(infos['gt_names'])
            tmp_anno_dict = {}
            tmp_anno_dict['name']             = np.array([])
            tmp_anno_dict['truncated']        = np.array([])
            tmp_anno_dict['occluded']         = np.array([])
            tmp_anno_dict['score']            = np.array([])
            tmp_anno_dict['difficulty']       = np.array([])
            tmp_anno_dict['num_points_in_gt'] = np.array([])

            tmp_anno_dict['bbox']       = np.empty((0, 4))
            tmp_anno_dict['location']   = np.empty((0, 3))
            tmp_anno_dict['dimensions'] = np.empty((0, 3))
            tmp_anno_dict['alpha']      = np.array([])
            tmp_anno_dict['rotation_y'] = np.array([])
            tmp_anno_dict['index']      = np.array([])

            #Convert boxes from BOX format to LiDARInstance3DBoxes
            gt_boxes_lidar =  LiDARInstance3DBoxes(
                    infos['gt_boxes'],
                    box_dim=infos['gt_boxes'].shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

            # """Reference: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py"""
            rect = pyquaternion.Quaternion(axis=[1, 0, 0], angle=0).rotation_matrix.astype(np.float32) # Camera is already rectified
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = infos['cams']['CAM_FRONT']['cam_intrinsic']  # Cameras are always rectified.
            P0 = p_left_kitti.astype(np.float32)
            P0 = gt_boxes_lidar.tensor.new_tensor(P0)

            velo_to_cam_rot = infos['cams']['CAM_FRONT']['sensor2lidar_rotation']
            velo_to_cam_trans = infos['cams']['CAM_FRONT']['sensor2lidar_translation']
            Trv2c = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            """"""

            gt_boxes_camera = gt_boxes_lidar.convert_to(Box3DMode.CAM, rect @ Trv2c)

            box_corners = gt_boxes_camera.corners
            box_corners_in_image = points_cam2img(box_corners, P0)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # Post-processing
            # check box_preds
            limit_range = gt_boxes_lidar.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((gt_boxes_lidar.center > limit_range[:3]) &
                            (gt_boxes_lidar.center < limit_range[3:]))
            valid_inds = valid_pcd_inds.all(-1)

            for box_idx in range(len(valid_inds)):
                if valid_inds[box_idx]==0:
                    continue
                gt_box_center   = gt_boxes_camera[box_idx].center.numpy().reshape(-1,)
                gt_box_dim      = gt_boxes_camera[box_idx].dims.numpy().reshape(-1,)
                gt_box_rot_y    = -gt_boxes_camera[box_idx].yaw.numpy().reshape(-1,) #invert sign to for flipped z from lidar to camera frames
                #TODO: add in 2d bbox projection
                box_x, box_y, rot_y = gt_box_center[0], gt_box_center[2], gt_box_rot_y
                alpha = -np.arctan2(box_x,box_y) + rot_y

                tmp_anno_dict['name']       = np.append(tmp_anno_dict['name'], infos['gt_names'][box_idx])
                tmp_anno_dict['truncated']  = np.append(tmp_anno_dict['truncated'], 0.0)
                tmp_anno_dict['occluded']   = np.append(tmp_anno_dict['occluded'], 0)
                tmp_anno_dict['score']      = np.append(tmp_anno_dict['score'], 0)
                tmp_anno_dict['difficulty'] = np.append(tmp_anno_dict['difficulty'], -1)
                tmp_anno_dict['num_points_in_gt']   = np.append(tmp_anno_dict['num_points_in_gt'], infos['num_lidar_pts'][box_idx])

                tmp_anno_dict['bbox']           = np.vstack( (tmp_anno_dict['bbox'], box_2d_preds[box_idx, :].numpy() ))
                tmp_anno_dict['alpha']          = np.append(tmp_anno_dict['alpha'], alpha)

                tmp_anno_dict['location']       = np.vstack((tmp_anno_dict['location'],  gt_box_center))
                tmp_anno_dict['dimensions']     = np.vstack((tmp_anno_dict['dimensions'], gt_box_dim))
                tmp_anno_dict['rotation_y']     = np.append(tmp_anno_dict['rotation_y'], gt_box_rot_y)

                class_idx = -1
                if infos['gt_names'][box_idx] in CLASS_INDEX_MAP:
                    class_idx = CLASS_INDEX_MAP[infos['gt_names'][box_idx]]
                tmp_anno_dict['index'] = np.append(tmp_anno_dict['index'], class_idx)
            annos_dict.append(tmp_anno_dict)
        return annos_dict

    def format_kitti_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       data_format='waymo'):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            data_format (str, optional): Output data format.
                Default: 'waymo'. Another supported choice is 'kitti'.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        assert ('waymo' in data_format or 'kitti' in data_format), \
            f'invalid data_format {data_format}'

        if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
                                                       pklfile_prefix_,
                                                       submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir
            
    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]

            sample_idx = idx
            image_shape = (1600, 900)

            box_dict = self.convert_valid_bboxes(pred_dicts, info, sample_idx)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name':         [],
                    'truncated':    [],
                    'occluded':     [],
                    'alpha':        [],
                    'bbox':         [],
                    'dimensions':   [],
                    'location':     [],
                    'rotation_y':   [],
                    'score':        []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info, sample_idx=-1):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)
        
        # """Reference: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py"""
        rect = pyquaternion.Quaternion(axis=[1, 0, 0], angle=0).rotation_matrix.astype(np.float32) # Camera is already rectified
        p_left_kitti = np.zeros((3, 4))
        p_left_kitti[:3, :3] = self.data_infos[sample_idx]['cams']['CAM_FRONT']['cam_intrinsic']  # Cameras are always rectified.
        P0 = p_left_kitti.astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)

        velo_to_cam_rot = self.data_infos[sample_idx]['cams']['CAM_FRONT']['sensor2lidar_rotation']
        velo_to_cam_trans = self.data_infos[sample_idx]['cams']['CAM_FRONT']['sensor2lidar_translation']
        Trv2c = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
        """"""

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
