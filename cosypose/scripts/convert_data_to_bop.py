import h5py
import numpy as np
import torch
import json
import os
import shutil
import torchvision

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from pathlib import Path
# from visual_servo.util import *

from enum import Enum
from torchvision.transforms.functional import resized_crop

# def rot_x(angle):
#     cosa = np.cos(angle)
#     sina = np.sin(angle)
#     return np.array([[1,0,0], [0, cosa, -sina], [0, sina, cosa]])

# def rot_y(angle):
#     cosa = np.cos(angle)
#     sina = np.sin(angle)
#     return np.array([[cosa, 0, sina], [0,1,0], [-sina, 0, cosa]])

# def rot_z(angle):
#     cosa = np.cos(angle)
#     sina = np.sin(angle)
#     return np.array([[cosa, -sina, 0], [sina, cosa, 0], [0,0,1]])

# # R_n = R(yaw_n, pitch_n, roll_n) R_{n - 1} R_{n - 2} ... R_1 R_0
# def get_rotation(rx, ry, rz):
#     rotation = rx.dot(ry).dot(rz)
#     # matrices.append(rotation.dot(last_matrix))
#     return rotation

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
X_OFFSET = 180

class RotationType(Enum):
    QUATERNION = "quaternion"
    EULER = "euler"


def crop_numpy_image_to_torch(image):
    img_torch = torch.from_numpy(image).permute(2, 0, 1)
    output_img = resized_crop(
        img_torch,
        0,
        X_OFFSET,
        image.shape[0],
        image.shape[0],
        [INPUT_HEIGHT, INPUT_WIDTH],
    ).float()
    return output_img

class RelativePoseDataset(Dataset):
    def __init__(
        self,
        h5_filename,
        rotation_type,
        grayscale,
        load_to_memory=True,
        rotational_order=[1, 1, 1],
        preprocess_fn=crop_numpy_image_to_torch,
    ) -> None:
        super(RelativePoseDataset).__init__()
        self._rotation_type = rotation_type
        self._h5_file = h5py.File(h5_filename, libver="latest")

        self._grayscale = (
            torchvision.transforms.Grayscale(num_output_channels=3)
            if grayscale
            else None
        )

        if load_to_memory:
            self._images = self._h5_file["images"][:]
        else:
            self._images = self._h5_file["images"]

        self._poses = self._h5_file["poses"]
        self._target_poses = self._h5_file["target_poses"]
        self._camera_poses = self._h5_file["camera_poses"]
        self._rotational_order = np.array(rotational_order)

        assert self._images.shape[0] == self._poses.shape[0]
        self._num_instances = self._images.shape[0]
        
        self._preprocess_fn = preprocess_fn
        self.scene_id = '000000'
        self.create_camera_json()
        self.cam_K = np.asarray([self.camera_json['fx'],
                          0,
                          self.camera_json['cx'],
                          0,
                          self.camera_json['fy'],
                          self.camera_json['cy'],
                          0,
                          0,
                          1]).reshape((3,3))

    def __len__(self):
        return self._num_instances

    def __getitem__(self, idx):
        img = self._images[idx]
        
        if self._preprocess_fn is not None:
            output_img = self._preprocess_fn(img)
        else:
            output_img = img

        pose = self._poses[idx]
        position = pose[:3]
        orientation = Rotation.from_quat([pose[3:]])
        transform = np.eye(4)
        transform[:3, :3] = orientation.as_matrix()
        transform[:3, 3] = position
        
        target_pose = self._target_poses[idx]
        target_position = target_pose[:3]
        target_orientation = Rotation.from_quat([target_pose[3:]])
        target_transform = np.eye(4)
        target_transform[:3, :3] = target_orientation.as_matrix()
        target_transform[:3, 3] = target_position

        camera_pose = self._camera_poses[idx]
        camera_position = camera_pose[:3]
        camera_orientation = Rotation.from_quat([camera_pose[3:]])
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = camera_orientation.as_matrix()
        camera_transform[:3, 3] = camera_position
        
        relative_transform = np.linalg.inv(camera_transform).dot(target_transform)
        relative_orientation = Rotation.from_matrix(relative_transform[:3, :3])
        if np.any(self._rotational_order > 1):
            euler = relative_orientation.as_euler("xyz")
            symmetric = (euler + np.pi / self._rotational_order) % (2.0 * np.pi)
            symmetric %= 2.0 * np.pi / self._rotational_order
            symmetric -= np.pi / self._rotational_order
            relative_orientation = Rotation.from_euler("xyz", symmetric)

        if self._rotation_type == RotationType.EULER:
            relative_pose = np.zeros(6)
            relative_pose[3:] = relative_orientation.as_euler("xyz")

        elif self._rotation_type == RotationType.QUATERNION:
            relative_pose = np.zeros(7)
            relative_pose[3:] = relative_orientation.as_quat()
        elif self._rotation_type == RotationType.NONE:
            relative_orientation = np.zeros(3)
            relative_pose = np.zeros(3)

        relative_pose[:3] = relative_transform[:3, 3]
        if self._grayscale is not None:
            output_img = self._grayscale(output_img)
        return output_img, torch.tensor(relative_pose).float(),  camera_orientation, camera_position, camera_transform, relative_transform, target_transform
    
    def dump(self, bop_path):
        '''save data to bop format'''
        # just copy from other folder for now. Part dimension and symmetries
        # self.create_model_info()
        # self.create_trainpbr()


    def create_model_info(self):
        model_path_05_04 = Path(self.bop_path) / 'syn_fos_j_assembly_left_centered_05_04_2023_15_15' / 'models'
        shutil.copytree(model_path_05_04, self.models_path)

    def create_camera_json(self):
        # camera_json_path = self.dataset_path / 'camera.json'
        # with open(camera_json_path, "w+") as f:
        #     json.dump(self.camera_json, f)
        pass
    def create_scene_hierarcy(self):
        scene_depth_path = self.scene_path / 'depth'
        scene_depth_path = self.scene_path / 'depth'
        if not os.path.exists(self.scene_path):
            os.makedirs(self.scene_path)

    def create_trainpbr(self):
        if not os.path.exists(self.trainpbr_path):
            os.makedirs(self.trainpbr_path)
        if not os.path.exists(self.scene_path):
            os.makedirs(self.scene_path)
        # camera intrinsic & extrinsic
        self.create_scene_camera()
        # store gt
        self.create_scene_gt()
        # store bbox info
        self.create_scene_gt_info()
        # just save images to RGB folder
        self.save_rgb()
    
    def create_scene_camera(self):
        raise NotImplementedError

    def create_scene_gt(self):
        scene_gt = {}
        for img_id, (img, nut_to_camera, cam_R, cam_t) in enumerate(self.pose_ds):
            scene_gt[img_id] = {
                'cam_R_m2c': [self.camera_json['fx'],
                          0,
                          self.camera_json['cx'],
                          0,
                          self.camera_json['fy'],
                          self.camera_json['cy'],
                          0,
                          0,
                          1],
                "cam_t_m2c": cam_R.tolist(),
                "obj_id": 0 # only nut
            }

    def create_scene_gt_info(self):
        pass
    
    def save_rgb(self):
        for img_id, (img, nut_to_camera, cam_R, cam_t) in enumerate(self.pose_ds):
            pil_im = Image.fromarray(img)
            path = self.rgb_path / f"{img_id:02d}.jpg"
            pil_im.save(str(path))


if __name__=="__main__":
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    bop_path = Path("/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/")
    dataset_name = 'real_data'
    dataset_path = Path(bop_path) / dataset_name
    models_path = Path(bop_path) / dataset_name / 'models'
    trainpbr_path = Path(bop_path) / dataset_name / 'train_pbr'
    scene_path = trainpbr_path / "000000"
    rgb_path = scene_path / "rgb"

    camera_json = {
       "cx": 377.614210,
       "cy": 245.553823,
       "depth_scale": 1.0,
       "fx": 1339.996108,
       "fy": 1339.743975,
       "height": 540,
       "width": 720
    }
    pose_ds = RelativePoseDataset(
        h5_filename=data_path,
        rotation_type=RotationType.EULER,
        grayscale=False,
        load_to_memory=True,
        rotational_order=[1, 1, 1],
        preprocess_fn=crop_numpy_image_to_torch
    )
    scene_camera_json = {}
    for img_id, (img, nut_to_camera, cam_R, cam_t) in enumerate(pose_ds):
        scene_camera_json[img_id] = {
            'cam_K': [self.camera_json['fx'],
                      0,
                      self.camera_json['cx'],
                      0,
                      self.camera_json['fy'],
                      self.camera_json['cy'],
                      0,
                      0,
                      1],
            "cam_R_w2c": cam_R.tolist(),
            "cam_t_w2c": cam_t.tolist(),
            "depth_scale": 1.0
        }
    scene_camera_path = self.scene_path / 'scene_camera.json'
    with open(scene_camera_path, "w+") as f:
        json.dump(scene_camera_json, f)