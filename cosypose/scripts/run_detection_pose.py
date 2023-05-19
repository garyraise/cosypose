import numpy as np
import logging
import pandas as pd
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse

import cosypose.utils.tensor_collection as tc
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.scripts.run_detection_eval import load_detector
from cosypose.scripts.run_cosypose_eval import load_models
from cosypose.scripts.run_detection_eval import load_detector
from cosypose.scripts.convert_data_to_bop import RelativePoseDataset, RotationType


def get_prediction(data, detector_model, pose_model=None):
    img, nut_to_camera, cam_R, cam_t = data
    # TODO
    # img check shape
    img = data['images'].cuda().float().permute(0, 3, 1, 2) / 255
    bracket_detections = detector_model.get_detections(
                    images=img,
                    one_instance_per_class=False,
                )
    bracket_detections = tc.concatenate(bracket_detections)
    pred_kwargs = {
            'pix2pose_detections': dict(
                detections=bracket_detections
            )
    }
    
    detections_ = detections_.cuda().float()
    data_TCO_init = detections_ if use_detections_TCO else None
    detections__ = detections_ if not use_detections_TCO else None
    cam_K = np.asarray([self.camera_json['fx'],
            0,
            self.camera_json['cx'],
            0,
            self.camera_json['fy'],
            self.camera_json['cy'],
            0,
            0,
            1]).reshape([3,3])
    
    candidates, sv_preds = pose_model.get_predictions(
                img, cam_K, detections=detections__,
                n_coarse_iterations=1,
                data_TCO_init=data_TCO_init,
                n_refiner_iterations=2,
                )
    
    return sv_preds

def main():
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    pose_ds = RelativePoseDataset(
        h5_filename=data_path,
        rotation_type=RotationType.EULER,
        grayscale=False,
        # load_to_memory=True,
        # rotational_order=[1, 1, 1],
        # preprocess_fn=crop_numpy_image_to_torch,
    )
    detector_object_set = "bracket_assembly"
    detector_model = load_detector(det_run_id)

    for img_id, data in enumerate(pose_ds):
        object_set = args.config
        get_prediction(data, detector_model, pose_model)
        

    
    return

if __name__=="__main__":

    parser = argparse.ArgumentParser('Det Pose Infer')
    parser.add_argument('--config', default='bracket_assembly_nut_bolt', type=str)
    # parser.add_argument('--coarse_run_id', dest='coarse_run_id', default=246643, type=int)
    args = parser.parse_args()
    # coarse_run_id = args.coarse_run_id
    coarse_run_id = 'bracket_assembly_nut_05_04_nosym_noaug_coarse--246643'
    det_run_id = 'detector-bracket_assembly--528073'
    main(det_run_id, coarse_run_id)
    
    