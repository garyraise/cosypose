import numpy as np
import logging
import cosypose.utils.tensor_collection as tc
import pandas as pd
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
from random import randint
import cosypose.utils.tensor_collection as tc
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.scripts.run_detection_eval import load_detector
from cosypose.scripts.run_cosypose_eval import load_models
from cosypose.scripts.convert_data_to_bop import RelativePoseDataset, RotationType
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor


def get_prediction(data, detector_model, pose_model=None):
    # img, nut_to_camera, cam_R, cam_R = data
    img, relative_pose, camera_orientation, cam_R, cam_R, relative_transform, target_transform = data
    # TODO
    # img check shape
    img = torch.unsqueeze(img, 0)
    img = img.cuda().float() / 255
    print("img", img.shape)
    bracket_detections = detector_model.get_detections(
                    images=img,
                    one_instance_per_class=False,
                )
    bracket_detections = 
    # bracket_detections = tc.concatenate(bracket_detections)
    # pred_kwargs = {
    #         'pix2pose_detections': dict(
    #             detections=bracket_detections
    #         )
    # }
    # generate from random list_bbox
    # list_bbox, infos = [], []
    # for i in range(2):
    #     start_x, start_y = randint(0,img.shape[1]), randint(0, img.shape[2])
    #     end_x, end_y = randint(start_x, img.shape[2]), randint(start_y,img.shape[2])
    #     list_bbox.append([start_x, start_y, end_x, end_y])
    #     info = dict(frame_obj_id=i,
    #                 label='obj_000005',
    #                 visib_fract=1,
    #                 scene_id='000000',
    #                 view_id='0',
    #                 batch_im_id=0)
    #     infos.append(info)
    # bracket_detections = tc.PandasTensorCollection(
    #     infos=pd.DataFrame(infos),
    #     bboxes=torch.as_tensor(np.stack(list_bbox)).float().cuda()
    # )
    camera_json = {
                "cx": 377.614210,
                "cy": 245.553823,
                "depth_scale": 1.0,
                "fx": 1339.996108,
                "fy": 1339.743975,
                "height": 540,
                "width": 720
            }
    cam_K = np.asarray([camera_json['fx'],
            0,
            camera_json['cx'],
            0,
            camera_json['fy'],
            camera_json['cy'],
            0,
            0,
            1]).reshape([3,3])
    cam_K = torch.as_tensor(np.expand_dims(cam_K, 0)).cuda()
    candidates, sv_preds = pose_model.get_predictions(
                img, K=cam_K, detections=bracket_detections,
                n_coarse_iterations=1,
                data_TCO_init=None,
                n_refiner_iterations=0,
                )
    
    return sv_preds

def main(det_run_id, coarse_run_id, refiner_run_id=None, object_set='bracket_assembly_05_04'):
    n_coarse_iterations = 1
    n_refiner_iterations = 0
    skip_mv = True
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    pose_ds = RelativePoseDataset(
        h5_filename=data_path,
        rotation_type=RotationType.EULER,
        grayscale=False,
        # load_to_memory=True,
        # rotational_order=[1, 1, 1],
        # preprocess_fn=crop_numpy_image_to_torch,
    )
    detector_model = load_detector(det_run_id)
    pose_predictor, mesh_db = load_models(coarse_run_id, object_set=object_set)
    mv_predictor = MultiviewScenePredictor(mesh_db)
    base_pred_kwargs = dict(
        n_coarse_iterations=n_coarse_iterations,
        n_refiner_iterations=n_refiner_iterations,
        skip_mv=skip_mv,
        pose_predictor=pose_predictor,
        mv_predictor=mv_predictor,
    )
    for img_id, data in enumerate(pose_ds):
        res = get_prediction(data, detector_model, pose_predictor)
        print(res)
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser('Det Pose Infer')
    parser.add_argument('--config', default='bracket_assembly_05_04_nut_bolt', type=str)
    # parser.add_argument('--coarse_run_id', dest='coarse_run_id', default=246643, type=int)
    args = parser.parse_args()
    # coarse_run_id = args.coarse_run_id
    coarse_run_id = 'bracket_assembly_nut_05_04_nosym_noaug_coarse--246643'
    # det_run_id = 'detector-detector-bracket_assembly_05_04--237393'
    
    det_run_id = 'detector-detector-bracket_assembly_05_04--122444'
    pose_run_id = 'bracket_assembly_nut_bolt_05_04_nosym_noaug_coarse--302621'
    refiner_run_id = 'bracket_assembly_nut_05_04_nosym_noaug_refiner--806506'
    refiner_run_id = None
    object_set = args.config
    main(det_run_id, coarse_run_id, refiner_run_id, object_set=object_set)
    
    