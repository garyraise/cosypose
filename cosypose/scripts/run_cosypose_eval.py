from cosypose.utils.tqdm import patch_tqdm; patch_tqdm()  # noqa
import torch.multiprocessing
import os
import time
import json

from collections import OrderedDict
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import pickle as pkl
import logging

from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR

from cosypose.utils.distributed import init_distributed_mode, get_world_size

from cosypose.lib3d import Transform

from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

from cosypose.evaluation.meters.pose_meters import PoseErrorMeter
from cosypose.evaluation.pred_runner.multiview_predictions import MultiviewPredictionRunner
from cosypose.evaluation.eval_runner.pose_eval import PoseEvaluation

import cosypose.utils.tensor_collection as tc
from cosypose.evaluation.runner_utils import format_results, gather_predictions
from cosypose.utils.distributed import get_rank


from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.datasets.bop import remap_bop_targets
from cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper

from cosypose.datasets.samplers import ListSampler
from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@MEMORY.cache
def load_flownet_results():
    results_path = LOCAL_DATA_DIR /'results' / 'bop---356706' / 'dataset=bracket_assembly'
    results = pkl.loads(results_path.read_bytes())
    infos, poses, bboxes = [], [], []

    l_offsets = (LOCAL_DATA_DIR / 'bop_datasets/ycbv' / 'offsets.txt').read_text().strip().split('\n')
    ycb_offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset)) * 0.001
        ycb_offsets[obj_id] = offset

    def mat_from_qt(qt):
        wxyz = qt[:4].copy().tolist()
        xyzw = [*wxyz[1:], wxyz[0]]
        t = qt[4:].copy()
        return Transform(xyzw, t)

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            label = f'obj_{obj_id:06d}'
            infos.append(dict(
                scene_id=scene_id,
                view_id=view_id,
                score=result['rois'][n, 1],
                label=label,
            ))
            bboxes.append(result['rois'][n, 2:6])
            pose = mat_from_qt(result['poses'][n])
            offset = ycb_offsets[obj_id]
            pose = pose * Transform((0, 0, 0, 1), offset).inverse()
            poses.append(pose.toHomogeneousMatrix())

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data

@MEMORY.cache
def load_posecnn_results():
    results_path = LOCAL_DATA_DIR / 'saved_detections' / 'ycbv_posecnn.pkl'
    results = pkl.loads(results_path.read_bytes())
    infos, poses, bboxes = [], [], []

    l_offsets = (LOCAL_DATA_DIR / 'bop_datasets/ycbv' / 'offsets.txt').read_text().strip().split('\n')
    ycb_offsets = dict()
    for l_n in l_offsets:
        obj_id, offset = l_n[:2], l_n[3:]
        obj_id = int(obj_id)
        offset = np.array(json.loads(offset)) * 0.001
        ycb_offsets[obj_id] = offset

    def mat_from_qt(qt):
        wxyz = qt[:4].copy().tolist()
        xyzw = [*wxyz[1:], wxyz[0]]
        t = qt[4:].copy()
        return Transform(xyzw, t)

    for scene_view_str, result in results.items():
        scene_id, view_id = scene_view_str.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        n_dets = result['rois'].shape[0]
        for n in range(n_dets):
            obj_id = result['rois'][:, 1].astype(np.int)[n]
            label = f'obj_{obj_id:06d}'
            infos.append(dict(
                scene_id=scene_id,
                view_id=view_id,
                score=result['rois'][n, 1],
                label=label,
            ))
            bboxes.append(result['rois'][n, 2:6])
            pose = mat_from_qt(result['poses'][n])
            offset = ycb_offsets[obj_id]
            pose = pose * Transform((0, 0, 0, 1), offset).inverse()
            poses.append(pose.toHomogeneousMatrix())

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data

@MEMORY.cache
def load_custom_detection_from_gt(ds_name='bracket_assembly'):
    categroy_to_model = {
            'bolt': '1',
            'nut': '5',
            }
    # train_classes = ['5'] if 'nut' in ds_name else None
    debug = 'debug' in ds_name
    dataset_name = 'bracket_assembly'
    if 'debug' in ds_name:
        dataset_name = 'bracket_assembly_debug'
    if '04_22' in ds_name:
        dataset_name = 'bracket_assembly_04_22'
    if '05_04' in ds_name:
        categroy_to_model = {
            'bolt': '1',
            'nut': '5',
            }
        dataset_name = 'syn_fos_j_assembly_left_centered_05_04_2023_15_15'
        # train_classes = ['4'] if 'nut' in ds_name else None
    train_classes = [v for k, v in categroy_to_model.items() if k in ds_name]
    # print(dataset_name, train_classes)
    path_data_dir = LOCAL_DATA_DIR / 'bop_datasets' / dataset_name
    # print(path_data_dir)
    path_scene_dir = os.path.join(path_data_dir, "train_pbr")
    scene_names = os.listdir(path_scene_dir)
    infos, poses, bboxes = [], [], []
    for scene_id, scene_name in enumerate(scene_names):
        path_scene_gt_info = os.path.join(path_scene_dir, scene_name, "scene_gt_info.json")
        path_scene_gt = os.path.join(path_scene_dir, scene_name, "scene_gt.json")
        path_scene_gt_camera = os.path.join(path_scene_dir, scene_name, "scene_camera.json")
        with open(path_scene_gt_info, "r") as f:
            json_data_gt_info = json.load(f)
        with open(path_scene_gt, "r") as f:
            json_data_gt = json.load(f)
        with open(path_scene_gt_camera, "r") as f:
            json_gt_camera = json.load(f)
        img_names_rgb = os.listdir(os.path.join(path_scene_dir, scene_name, "rgb"))
        for img_id, img_name in enumerate(img_names_rgb[:-1]):
            if not f"{img_id}" in json_data_gt_info:
                continue
            if not f"{img_id}" in json_data_gt:
                continue
            if not f"{img_id}" in json_gt_camera:
                continue
            cam_R_w2c = json_gt_camera[f"{img_id}"]["cam_R_w2c"]
            cam_t_w2c = json_gt_camera[f"{img_id}"]["cam_t_w2c"]
            row0 = [cam_R_w2c[0], cam_R_w2c[1], cam_R_w2c[2], cam_t_w2c[0]] 
            row1 = [cam_R_w2c[3], cam_R_w2c[4], cam_R_w2c[5], cam_t_w2c[1]]
            row2 = [cam_R_w2c[6], cam_R_w2c[7], cam_R_w2c[8], cam_t_w2c[2]]           
            row3 = [0, 0, 0, 1]
            cam_rot_loc_mat = np.asarray([row0, row1, row2, row3])
            # TODO: ADD no_sym / single_cat option for inference
            for label_idx, label in enumerate(json_data_gt[f"{img_id}"]):
                obj_id = label["obj_id"] # int
                if train_classes and str(obj_id) not in train_classes:
                    continue
                list_bbox = json_data_gt_info[f"{img_id}"][label_idx]["bbox_visib"]
                xmin = list_bbox[0]
                ymin = list_bbox[1]
                xmax = list_bbox[0] +  list_bbox[2]
                ymax = list_bbox[1] +  list_bbox[3]
                list_bbox = [xmin, ymin, xmax, ymax]
                list_rot  = json_data_gt[f"{img_id}"][label_idx]["cam_R_m2c"]
                list_loc  = json_data_gt[f"{img_id}"][label_idx]["cam_t_m2c"]
                
                row0 = [list_rot[0], list_rot[1], list_rot[2], list_loc[0]] 
                row1 = [list_rot[3], list_rot[4], list_rot[5], list_loc[1]]
                row2 = [list_rot[6], list_rot[7], list_rot[8], list_loc[2]]
                row3 = [0, 0, 0, 1]
                rot_loc_mat = np.asarray([row0, row1, row2, row3])
                rot_loc_mat = np.matmul(np.linalg.inv(cam_rot_loc_mat), rot_loc_mat)
                infos.append(dict(
                        scene_id=scene_id,
                        view_id=img_id,
                        score=1,
                        label=f"obj_{obj_id:06d}",
                    ))
                poses.append(rot_loc_mat)
                bboxes.append(list_bbox)
    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)).float(),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data

@MEMORY.cache
def load_pix2pose_results(all_detections=True, remove_incorrect_poses=False):
    if all_detections:
        results_path = LOCAL_DATA_DIR / 'saved_detections' / 'tless_pix2pose_retinanet_vivo_all.pkl'
    else:
        results_path = LOCAL_DATA_DIR / 'saved_detections' / 'tless_pix2pose_retinanet_siso_top1.pkl'
    pix2pose_results = pkl.loads(results_path.read_bytes())
    
    infos, poses, bboxes = [], [], []
    for key, result in pix2pose_results.items():
        scene_id, view_id = key.split('/')
        scene_id, view_id = int(scene_id), int(view_id)
        boxes = result['rois']
        scores = result['scores']
        poses_ = result['poses']

        labels = result['labels_txt']
        new_boxes = boxes.copy()
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,3] = boxes[:,2]
        for o, label in enumerate(labels):
            t = poses_[o][:3, -1]
            if remove_incorrect_poses and (np.sum(t) == 0 or np.max(t) > 100):
                pass
            else:
                infos.append(dict(
                    scene_id=scene_id,
                    view_id=view_id,
                    score=scores[o],
                    label=label,
                ))
                bboxes.append(new_boxes[o])
                poses.append(poses_[o])

    data = tc.PandasTensorCollection(
        infos=pd.DataFrame(infos),
        poses=torch.as_tensor(np.stack(poses)),
        bboxes=torch.as_tensor(np.stack(bboxes)).float(),
    ).cpu()
    return data


def get_pose_meters(scene_ds):
    ds_name = scene_ds.name
    compute_add = False
    spheres_overlap_check = True
    large_match_threshold_diameter_ratio = 0.5
    print("ds_name", ds_name)
    if ds_name == 'tless.primesense.test.bop19':
        targets_filename = 'test_targets_bop19.json'
        visib_gt_min = -1
        n_top = -1  # Given by targets
    elif ds_name == 'tless.primesense.test':
        targets_filename = 'all_target_tless.json'
        n_top = 1
        visib_gt_min = 0.1
    elif 'ycbv' in ds_name:
        compute_add = True
        visib_gt_min = -1
        targets_filename = None
        n_top = 1
        spheres_overlap_check = False
    elif 'bracket_assembly' in ds_name:
        compute_add = True
        targets_filename = None
        visib_gt_min = -1
        n_top = 1  # Given by targets
        spheres_overlap_check = False
    else:
        raise ValueError

    if 'tless' in ds_name:
        object_ds_name = 'tless.eval'
    elif 'ycbv' in ds_name:
        object_ds_name = 'ycbv.bop-compat.eval'  # This is important for definition of symmetric objects
    elif 'bracket_assembly' in ds_name:
        object_ds_name = ds_name
    else:
        raise ValueError

    if targets_filename is not None:
        targets_path = scene_ds.ds_dir / targets_filename
        targets = pd.read_json(targets_path)
        targets = remap_bop_targets(targets)
    else:
        targets = None

    object_ds = make_object_dataset(object_ds_name)
    print("object_ds_name", object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)

    error_types = ['ADD-S'] + (['ADD(-S)'] if compute_add else [])

    # base_kwargs = dict(
    #     mesh_db=mesh_db,
    #     exact_meshes=True,
    #     sample_n_points=None,
    #     errors_bsz=1,

    #     # BOP-Like parameters
    #     n_top=n_top,
    #     visib_gt_min=visib_gt_min,
    #     targets=targets,
    #     spheres_overlap_check=spheres_overlap_check,
    # )
    # sample less points
    base_kwargs = dict(
        mesh_db=mesh_db,
        # exact_meshes=True,
        # sample_n_points=None,
        exact_meshes=False,
        sample_n_points=100,
        errors_bsz=1,
        # BOP-Like parameters
        n_top=n_top,
        visib_gt_min=visib_gt_min,
        targets=targets,
        spheres_overlap_check=spheres_overlap_check,
    )

    meters = dict()
    for error_type in error_types:
        # For measuring ADD-S AUC on T-LESS and average errors on ycbv/tless.
        meters[f'{error_type}_ntop=BOP_matching=OVERLAP'] = PoseErrorMeter(
            error_type=error_type, consider_all_predictions=False,
            match_threshold=large_match_threshold_diameter_ratio,
            report_error_stats=True, report_error_AUC=True, **base_kwargs)

        if 'ycbv' in ds_name:
            # For fair comparison with PoseCNN/DeepIM on YCB-Video ADD(-S) AUC
            meters[f'{error_type}_ntop=1_matching=CLASS'] = PoseErrorMeter(
                error_type=error_type, consider_all_predictions=False,
                match_threshold=np.inf,
                report_error_stats=False, report_error_AUC=True, **base_kwargs)

        if 'tless' in ds_name:
            meters.update({f'{error_type}_ntop=BOP_matching=BOP':  # For ADD-S<0.1d
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1, **base_kwargs),

                           f'{error_type}_ntop=ALL_matching=BOP':  # For mAP
                           PoseErrorMeter(error_type=error_type, match_threshold=0.1,
                                          consider_all_predictions=True,
                                          report_AP=True, **base_kwargs)})
    return meters


def load_models(coarse_run_id, refiner_run_id=None, n_workers=8, object_set='tless'):
    if object_set == 'tless':
        object_ds_name, urdf_ds_name = 'tless.bop', 'tless.cad'
    elif 'bracket_assembly' in object_set:
        object_ds_name, urdf_ds_name = object_set, 'bracket_assembly'
    else:
        object_ds_name, urdf_ds_name = 'ycbv.bop-compat.eval', 'ycbv'

    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=n_workers)
    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id):
        if run_id is None:
            return
        run_dir = EXP_DIR / run_id
        cfg = yaml.unsafe_load((run_dir / 'config.yaml').read_text())
        cfg = check_update_config(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
            ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        if DEBUG:
            model.enable_debug()
        return model

    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    model = CoarseRefinePosePredictor(coarse_model=coarse_model,
                                      refiner_model=refiner_model)
    # print("mesh_db",object_ds, mesh_db)
    return model, mesh_db


def main():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if 'cosypose' in logger.name:
            logger.setLevel(logging.INFO)

    logger.info("Starting ...")
    init_distributed_mode()

    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--config', default='tless-bop', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--job_dir', default='', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--nviews', dest='n_views', default=1, type=int)
    parser.add_argument('--coarse_run_id', dest='coarse_run_id', default=131619, type=int)
    args = parser.parse_args()

    coarse_run_id = None
    refiner_run_id = None
    n_workers = 8
    n_plotters = 8
    n_views = 1

    scene_id = None
    group_id = None
    n_groups = None
    frame_ids = None
    n_views = args.n_views
    skip_mv = args.n_views < 2
    skip_predictions = False
    if args.coarse_run_id:
        coarse_run_id = args.coarse_run_id
    object_set = 'tless'
    if 'tless' in args.config:
        object_set = 'tless'
        coarse_run_id = 'tless-coarse--10219'
        refiner_run_id = 'tless-refiner--585928'
        n_coarse_iterations = 1
        n_refiner_iterations = 4
    elif 'ycbv' in args.config:
        object_set = 'ycbv'
        refiner_run_id = 'ycbv-refiner-finetune--251020'
        n_coarse_iterations = 0
        n_refiner_iterations = 2
    elif 'bracket_assembly' in args.config:
        # make nut-only object dataset or all categories
        object_set = args.config
        
        # # all cat_sym (baseline)
        # coarse_run_id = f'bracket_assembly_coarse--626765'
        # refiner_run_id = f'bracket_assembly_refiner--990144'
        # # single_cat_no_sym
        # coarse_run_id = 'bracket_assembly_coarse--12034'
        # refiner_run_id = 'bracket_assembly_refiner--8403'
        # single_cat_sym
        # coarse_run_id = 'bracket_assembly_coarse--206480'
        # refiner_run_id = 'bracket_assembly_coarse--206480'
        # 05_04
        coarse_run_id = 'bracket_assembly_nut_05_04_nosym_noaug_coarse--246643'
        refiner_run_id = 'bracket_assembly_nut_05_04_nosym_noaug_refiner--806506'
        # single frame sym nut
        
        n_coarse_iterations = 1
        n_refiner_iterations = 2
    else:
        raise ValueError(args.config)

    if args.config == 'tless-siso':
        ds_name = 'tless.primesense.test'
        assert n_views == 1
    elif args.config == 'tless-vivo':
        ds_name = 'tless.primesense.test.bop19'
    elif args.config == 'ycbv':
        ds_name = 'ycbv.test.keyframes'
    elif 'bracket_assembly' in args.config:
        ds_name = args.config
    else:
        raise ValueError(args.config)

    global DEBUG
    DEBUG = False
    if args.debug:
        DEBUG = args.debug
        if 'tless' in args.config:
            scene_id = None
            group_id = 64
            n_groups = 2
        else:
            scene_id = 48
            n_groups = 2
            scene_id = 0
            frame_ids = [10,20,30,40]
        n_workers = 0
        n_plotters = 0
    n_rand = np.random.randint(1e10)
    save_dir = RESULTS_DIR / f'{args.config}-n_views={n_views}-{args.comment}-{n_rand}'
    logger.info(f"SAVE DIR: {save_dir}")
    logger.info(f"Coarse: {coarse_run_id}")
    logger.info(f"Refiner: {refiner_run_id}")

    # Load dataset
    scene_ds = make_scene_dataset(ds_name)

    if scene_id is not None:
        mask = scene_ds.frame_index['scene_id'] == scene_id
        scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)
    if frame_ids is not None:
        scene_ds.frame_index = scene_ds.frame_index.iloc[frame_ids,:]
    # Predictions
    print("object_set", object_set)
    predictor, mesh_db = load_models(coarse_run_id, refiner_run_id, n_workers=n_plotters, object_set=object_set)

    mv_predictor = MultiviewScenePredictor(mesh_db)
    base_pred_kwargs = dict(
        n_coarse_iterations=n_coarse_iterations,
        n_refiner_iterations=n_refiner_iterations,
        skip_mv=skip_mv,
        pose_predictor=predictor,
        mv_predictor=mv_predictor,
    )
    skip_predictions = False
    if skip_predictions:
        pred_kwargs = {}
    elif 'bracket_assembly' in ds_name:
        bracket_detections = load_custom_detection_from_gt(ds_name).cpu()
        pred_kwargs = {
            'pix2pose_detections': dict(
                detections=bracket_detections,
                **base_pred_kwargs
            )
        }
    elif 'tless' in ds_name:
        pix2pose_detections = load_pix2pose_results(all_detections='bop19' in ds_name).cpu()
        pred_kwargs = {
            'pix2pose_detections': dict(
                detections=pix2pose_detections,
                **base_pred_kwargs
            )
        }
    elif 'ycbv' in ds_name:
        posecnn_detections = load_posecnn_results()
        pred_kwargs = {
            'posecnn_init': dict(
                detections=posecnn_detections,
                use_detections_TCO=posecnn_detections,
                **base_pred_kwargs
            ),
        }
    else:
        raise ValueError(ds_name)

    scene_ds_pred = MultiViewWrapper(scene_ds, n_views=n_views)
    if group_id is not None:
        mask = scene_ds_pred.frame_index['group_id'] == group_id
        scene_ds_pred.frame_index = scene_ds_pred.frame_index[mask].reset_index(drop=True)
    elif n_groups is not None:
        scene_ds_pred.frame_index = scene_ds_pred.frame_index[:n_groups]

    pred_runner = MultiviewPredictionRunner(
        scene_ds_pred, batch_size=1, n_workers=n_workers,
        cache_data=len(pred_kwargs) > 1)

    all_predictions = dict()
    for pred_prefix, pred_kwargs_n in pred_kwargs.items():
        logger.info(f"Prediction: {pred_prefix}")
        preds = pred_runner.get_predictions(**pred_kwargs_n)
        for preds_name, preds_n in preds.items():
            all_predictions[f'{pred_prefix}/{preds_name}'] = preds_n

    logger.info("Done with predictions")
    torch.distributed.barrier()

    # Evaluation
    predictions_to_evaluate = set()
    if 'ycbv' in ds_name:
        det_key = 'posecnn_init'
        all_predictions['posecnn'] = posecnn_detections
        predictions_to_evaluate.add('posecnn')
        predictions_to_evaluate.add(f'{det_key}/refiner/iteration={n_refiner_iterations}')
    elif 'tless' in ds_name:
        det_key = 'pix2pose_detections'
        predictions_to_evaluate.add(f'{det_key}/refiner/iteration={n_refiner_iterations}')
    elif 'bracket_assembly' in ds_name: # BOP dataset
        det_key = 'pix2pose_detections'
        predictions_to_evaluate.add(f'{det_key}/coarse/iteration=1')
        predictions_to_evaluate.add(f'{det_key}/refiner/iteration={n_refiner_iterations}')
    else:
        raise ValueError(ds_name)

    if args.n_views > 1:
        for k in [
                # f'ba_input',
                # f'ba_output',
                f'ba_output+all_cand'
        ]:
            predictions_to_evaluate.add(f'{det_key}/{k}')

    all_predictions = OrderedDict({k: v for k, v in sorted(all_predictions.items(), key=lambda item: item[0])})
    # Evaluation.
    meters = get_pose_meters(scene_ds)
    mv_group_ids = list(iter(pred_runner.sampler))
    scene_ds_ids = np.concatenate(scene_ds_pred.frame_index.loc[mv_group_ids, 'scene_ds_ids'].values)
    sampler = ListSampler(scene_ds_ids)
    eval_runner = PoseEvaluation(scene_ds, meters, n_workers=n_workers,
                                 cache_data=True, batch_size=1, sampler=sampler)

    eval_metrics, eval_dfs = dict(), dict()
    for preds_k, preds in all_predictions.items():
        if preds_k in predictions_to_evaluate:
            logger.info(f"Evaluation : {preds_k} (N={len(preds)})")
            if len(preds) == 0:
                preds = eval_runner.make_empty_predictions()
            eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            preds.cpu()
        else:
            logger.info(f"Skipped: {preds_k} (N={len(preds)})")

    all_predictions = gather_predictions(all_predictions)

    metrics_to_print = dict()
    if 'ycbv' in ds_name:
        metrics_to_print.update({
            f'posecnn/ADD(-S)_ntop=1_matching=CLASS/AUC/objects/mean': f'PoseCNN/AUC of ADD(-S)',

            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD(-S)_ntop=1_matching=CLASS/AUC/objects/mean': f'Singleview/AUC of ADD(-S)',
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=1_matching=CLASS/AUC/objects/mean': f'Singleview/AUC of ADD-S',

            f'{det_key}/ba_output+all_cand/ADD(-S)_ntop=1_matching=CLASS/AUC/objects/mean': f'Multiview (n={args.n_views})/AUC of ADD(-S)',
            f'{det_key}/ba_output+all_cand/ADD-S_ntop=1_matching=CLASS/AUC/objects/mean': f'Multiview (n={args.n_views})/AUC of ADD-S',
        })
    elif 'bracket_assembly'  in ds_name or 'tless' in ds_name:
        metrics_to_print.update({
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Singleview/AUC of ADD-S',
            # f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Singleview/ADD-S<0.1d',
            f'{det_key}/refiner/iteration={n_refiner_iterations}/ADD-S_ntop=ALL_matching=BOP/mAP': f'Singleview/mAP@ADD-S<0.1d',


            f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=OVERLAP/AUC/objects/mean': f'Multiview (n={args.n_views})/AUC of ADD-S',
            # f'{det_key}/ba_output+all_cand/ADD-S_ntop=BOP_matching=BOP/0.1d': f'Multiview (n={args.n_views})/ADD-S<0.1d',
            f'{det_key}/ba_output+all_cand/ADD-S_ntop=ALL_matching=BOP/mAP': f'Multiview (n={args.n_views}/mAP@ADD-S<0.1d)',
        })
    else:
        raise ValueError

    metrics_to_print.update({
        f'{det_key}/ba_input/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview before BA/ADD-S (m)',
        f'{det_key}/ba_output/ADD-S_ntop=BOP_matching=OVERLAP/norm': f'Multiview after BA/ADD-S (m)',
    })
    if get_rank() == 0:
        save_dir.mkdir()
        results = format_results(all_predictions, eval_metrics, eval_dfs, print_metrics=True)
        (save_dir / 'full_summary.txt').write_text(results.get('summary_txt', ''))
        # print("results,all_predictions, all_predictions", results, all_predictions, eval_dfs)

        full_summary = results['summary']
        summary_txt = 'Results:'
        for k, v in metrics_to_print.items():
            if k in full_summary:
                summary_txt += f"\n{v}: {full_summary[k]}"
        logger.info(f"{'-'*80}")
        logger.info(summary_txt)
        logger.info(f"{'-'*80}")

        torch.save(results, save_dir / 'results.pth.tar')
        (save_dir / 'summary.txt').write_text(summary_txt)
        logger.info(f"Saved: {save_dir}")


if __name__ == '__main__':
    # patch_tqdm()
    main()
    # time.sleep(2)
    if get_world_size() > 1:
        torch.distributed.barrier()