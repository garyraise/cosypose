# import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import logging
import cosypose.utils.tensor_collection as tc
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import argparse
import open3d as o3d
from open3d.visualization import rendering

from random import randint
import cosypose.utils.tensor_collection as tc
from cosypose.datasets.datasets_cfg import make_scene_dataset
from cosypose.scripts.run_detection_eval import load_detector
from cosypose.scripts.run_cosypose_eval import load_models
from cosypose.scripts.convert_data_to_bop import RelativePoseDataset, RotationType
from cosypose.integrated.multiview_predictor import MultiviewScenePredictor

camera_json = {
        "cx": 377.614210,
        "cy": 245.553823,
        "depth_scale": 1.0,
        "fx": 1339.996108,
        "fy": 1339.743975,
        "height": 540,
        "width": 720,
    }

def get_prediction(data, detector_model, pose_model=None):
    # img, nut_to_camera, cam_R, cam_R = data
    (
        img,
        relative_pose,
        camera_orientation,
        cam_R,
        cam_R,
        relative_transform,
        target_transform,
    ) = data
    # original_img = img

    img = torch.unsqueeze(img, 0)
    img = img.cuda().float() / 255
    bracket_detections = detector_model.get_detections(
        images=img,
        one_instance_per_class=False,
    )

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
    cam_K = np.asarray(
        [
            camera_json["fx"],
            0,
            camera_json["cx"],
            0,
            camera_json["fy"],
            camera_json["cy"],
            0,
            0,
            1,
        ]
    ).reshape([3, 3])
    cam_K = torch.as_tensor(np.expand_dims(cam_K, 0)).cuda()
    _, sv_preds = pose_model.get_predictions(
        img,
        K=cam_K,
        detections=bracket_detections,
        n_coarse_iterations=1,
        data_TCO_init=None,
        n_refiner_iterations=0,
    )
    return bracket_detections, sv_preds

# def render_scene():
#     render = rendering.OffscreenRenderer(img_width, img_height)

def main(
    det_run_id,
    coarse_run_id,
    refiner_run_id=None,
    output=None,
    object_set="bracket_assembly_05_04",
):
    n_coarse_iterations = 1
    n_refiner_iterations = 0
    skip_mv = True
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    if not output:
        output = Path('/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/visualizations/real_data_rend')
        os.makedirs(output, exist_ok=True)
    detections_folder = output / "detections"
    renders_folder = output / "renders"
    os.makedirs(detections_folder, exist_ok=True)
    os.makedirs(renders_folder, exist_ok=True)

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
    width, height = (720, 540) # np_img.shape[1], np_img.shape[0]
    render = rendering.OffscreenRenderer(width, height, headless=True)
    for img_id, data in enumerate(pose_ds):
        (
            img,
            relative_pose,
            camera_orientation,
            cam_R,
            cam_R,
            relative_transform,
            target_transform
        ) = data
        print('img_id', img_id)
        # detections, poses = get_prediction(data, detector_model, pose_predictor)
        # print(detections.infos['label'])

        # boxes = np.array(detections.bboxes.cpu()).astype(int)
        # np_img = (
        #     np.array(data[0].squeeze().permute(1, 2, 0).cpu()).astype(np.uint8).copy()
        # )
        # for i in range(boxes.shape[0]):
        #     top_left = tuple(boxes[i, :2].tolist())
        #     bottom_right = tuple(boxes[i, 2:].tolist())
        #     cv2.rectangle(
        #         np_img,
        #         top_left,
        #         bottom_right,
        #         color=(255, 0, 0),
        #         thickness=2,
        #     )
        #     cv2.putText(np_img, detections.infos['label'][i], (bottom_right[0] - 70, bottom_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        # cv2.imwrite(os.path.join(detections_folder, f"{img_id:06d}.png"), np_img)
        # setup camera intrinsic values
        pinhole = o3d.camera.PinholeCameraIntrinsic(width, height, camera_json['fx'], camera_json['fy'], camera_json['cx'], camera_json['cy'])
            
        # Pick a background colour of the rendered image, I set it as black (default is light gray)
        render.scene.set_background([163, 8, 193, 0.2])  # RGBA

        # # now create your mesh
        # mesh_db = mesh_db.batched().cuda().float()
        # meshes = mesh_db.select(labels)
        # render GT
        gt_label = 'obj_000004'
        gt_mesh_path = mesh_db.infos[gt_label]['mesh_path']
        mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        # define further mesh properties, shape, vertices etc  (omitted here)  
        # Define a simple unlit Material.
        # (The base color does not replace the mesh's own colors.)
        mtl = o3d.visualization.rendering.Material()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"
        # o3d.visualization.draw_geometries([mesh])
        # cv2.imwrite(str(renders_folder / f"mesh_img_{img_id}.png"), mesh_img)
        # add mesh to the scene
        T = np.eye(4)
        rotation = np.eye(3)
        translation = np.array([0, 0, 0.2])
        T[:3, :3] = rotation
        T[:3, 3] = translation
        print("T, ", T)
        print("relative transformation", relative_transform)
        mesh_t = deepcopy(mesh).transform(T)
        relative_transform[:3, 3] = [relative_transform[2, 3], relative_transform[1, 3], -relative_transform[0, 3]]
        print("relative transformation after", relative_transform)
        mesh_real = deepcopy(mesh).transform(relative_transform)
        # render.scene.add_geometry(f"mesh_{gt_label}", mesh_t, mtl)
        # render pred
        # for label in detections.infos['label']:
        #     mesh_path = mesh_db.infos[label]['mesh_path']# rewrite mesh_db from open3d
        #     mesh = o3d.io.read_triangle_mesh(mesh_path)
        #     # mesh = o3d.geometry.TriangleMesh()
        #     mesh.paint_uniform_color([1.0, 0.0, 0.0]) # set Red color for mesh 
            # define further mesh properties, shape, vertices etc  (omitted here)  

            # # Define a simple unlit Material.
            # # (The base color does not replace the mesh's own colors.)
            # mtl = o3d.visualization.rendering.Material()
            # mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
            # mtl.shader = "defaultUnlit"

            # # add mesh to the scene
            # render.scene.add_geometry(f"mesh_{label}", mesh, mtl)

        # if self.exact_meshes:
        #     assert len(labels) == 1
        #     n_points = self.mesh_db.infos[labels[0]]['n_points']
        #     points = meshes.points[:, :n_points]
        # else:
        #     if self.sample_n_points is not None:
        #         points = meshes.sample_points(self.sample_n_points, deterministic=True)
        #     else:
        #         points = meshes.points

        

        render.scene.camera.set_projection(pinhole.intrinsic_matrix, 0.01, 10.0, float(width), float(height))

        mtl = o3d.visualization.rendering.Material()  # or MaterialRecord(), for later versions of Open3D
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        # Add the arrow mesh to the scene.
        # (These are thicker than the main axis arrows, but the same length.)
        render.scene.add_geometry("rotated_model", mesh_t, mtl)
        render.scene.add_geometry("real_model", mesh_real, mtl)
        # Since the arrow material is unlit, it is not necessary to change the scene lighting.
        #render.scene.scene.enable_sun_light(False)
        #render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

        # Optionally set the camera field of view (to zoom in a bit)
        # vertical_field_of_view = 15.0  # between 5 and 90 degrees
        # aspect_ratio = width / height  # azimuth over elevation
        # near_plane = 0.1
        # far_plane = 50.0
        # fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        # render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 1]  # look_at target
        eye = [0, 0, 0]  # camera position
        up = [0, 1, 0]  # camera orientation
        render.scene.camera.look_at(center, eye, up)

        img_o3d = render.render_to_image()

        # we can now save the rendered image right at this point 
        o3d_render_path = str(renders_folder / f"output_{img_id}.png")
        o3d.io.write_image(o3d_render_path, img_o3d, 9)
        print("RENDERED: ", o3d_render_path)
        # img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
        # cv2.imwrite(str(renders_folder / f"orginal_{img_id}.png"), img_cv2)
        original_img_path = str(renders_folder / f'original_{img_id}.png')
        np_img = (
            np.array(img.squeeze().permute(1, 2, 0).cpu()).astype(np.uint8).copy()
        )
        plt.imsave(original_img_path, np_img)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Det Pose Infer")
    parser.add_argument("--config", default="bracket_assembly_05_04", type=str)
    parser.add_argument("--output", default=None)
    # parser.add_argument('--coarse_run_id', dest='coarse_run_id', default=246643, type=int)
    args = parser.parse_args()
    # coarse_run_id = args.coarse_run_id
    # coarse_run_id = "bracket_assembly_nut_05_04_nosym_noaug_coarse--246643"
    # det_run_id = 'detector-detector-bracket_assembly_05_04--237393'

    det_run_id = "detector-detector-bracket_assembly_05_04--122444"
    pose_run_id = "bracket_assembly_nut_bolt_05_04_nosym_noaug_coarse--302621"
    refiner_run_id = "bracket_assembly_nut_05_04_nosym_noaug_refiner--806506"
    refiner_run_id = None
    object_set = args.config

    main(det_run_id, pose_run_id, refiner_run_id, object_set=object_set, output=args.output)
