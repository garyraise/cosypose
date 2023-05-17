import h5py
import numpy as np
import torch
import torchvision
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
            relative_pose = np.zeros(3)

        relative_pose[:3] = relative_transform[:3, 3]
        if self._grayscale is not None:
            output_img = self._grayscale(output_img)
        return output_img, torch.tensor(relative_pose).float()
    
    def dump(self, bop_path):
        '''save data to bop format'''
        self.bop_path = Path(bop_path)
        scene_folder = self.bop_path / self.scene_id
        self.create_model_json()
        self.create_camera_json()
        self.create_train()
    
    def create_train(self):
        self.create_scene_camera()
        self.create_scene_gt()
        self.create_scene_gt_info()
        self.save_rgb()
    
    # def 
    
if __name__=="__main__":
    data_path = "/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/check_1"
    pose_ds = RelativePoseDataset(
        h5_filename=data_path,
        rotation_type=RotationType.EULER,
        grayscale=False,
        # load_to_memory=True,
        # rotational_order=[1, 1, 1],
        # preprocess_fn=crop_numpy_image_to_torch,
    )
    pose_ds.dump(bop_path='/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/bop_datasets/real_data/')
    for data in pose_ds:
        print(data)