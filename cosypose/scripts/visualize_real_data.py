import numpy as np
import logging
import cosypose.utils.tensor_collection as tc
import pandas as pd
import os

from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.scripts.convert_data_to_bop import RelativePoseDataset, RotationType
from cosypose.visualization.singleview import render_prediction_wrt_camera
from cosypose.evaluation.data_utils import parse_obs_data
# debug
import torch
from cosypose.visualization.singleview import filter_predictions, make_singleview_prediction_matplotlib
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.config import LOCAL_DATA_DIR
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    pose_ds = RelativePoseDataset(
        h5_filename=data_path,
        rotation_type=RotationType.EULER,
        grayscale=False
    )
    urdf_ds_name = 'bracket_assembly'
    scene_id = '000000'
    renderer = BulletSceneRenderer(urdf_ds_name)
    obj_id = 5 # nut only dataset
    save_dir = Path("/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/visualizations") / 'real_data_rend'
    os.makedirs(save_dir, exist_ok=True)
    for img_id, data in enumerate(pose_ds):
        img, relative_pose, camera_orientation, camera_position, camera_transform, relative_transform, target_transform = data
        image = np.transpose(img, axes=(1,2,0))
        plt.imsave(save_dir / f"gt_{img_id}.png", image.cpu().numpy())
        cam_K = pose_ds.cam_K
        camera_info = dict(K=cam_K,
            TWC=camera_transform, # 4x4
            resolution=(224,224))
        frame_info = dict(scene_id=scene_id,
                          view_id=img_id)
        camera = dict(TWC=camera_info["TWC"])
        objects = [dict(TWO=target_transform, #relative_transform
                       bbox=[],
                       name=f"obj_{obj_id:06d}")]
        obs = dict(frame_info=frame_info,
                   camera=camera,
                   objects=objects
                   )
        list_obs = parse_obs_data(obs)
        # camera: a list of camera_transformation matrix for multiview camera infos
        pred_rendered = render_prediction_wrt_camera(renderer, list_obs, camera=camera_info)
        plt.imsave(save_dir / f"pred_{img_id}.png", pred_rendered)
        print("pred_rendered", pred_rendered.shape, save_dir / f"pred_{img_id}.png")

if __name__=='__main__':
    main()
    # scene_id, view_id = 0, 10
    
    # result_id = 'bracket_assembly_05_04_nut_bolt_nosym_debug-n_views=1--1880799376'
    # ds_name, urdf_ds_name = 'bracket_assembly_05_04_nut_bolt', 'bracket_assembly'

    # results = LOCAL_DATA_DIR / 'results' / result_id / 'results.pth.tar'
    # scene_ds = make_scene_dataset(ds_name)
    # results = torch.load(results)['predictions']
    # pred_key = 'pix2pose_detections/coarse/iteration=1'
    # print(scene_id, view_id, pred_key)
    # this_preds = filter_predictions(results[pred_key], scene_id, view_id)
    # renderer = BulletSceneRenderer(urdf_ds_name)
    # rgb_input, pred_rendered = make_singleview_prediction_matplotlib(scene_ds, renderer, this_preds)
    # renderer.disconnect()
    # plt.figure()
    # f, ax = plt.subplots(2,1) 
    # ax[0].imshow(rgb_input)
    # ax[1].imshow(pred_rendered)