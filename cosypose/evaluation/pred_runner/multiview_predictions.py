import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from collections import defaultdict
from cosypose.config import LOCAL_DATA_DIR

from cosypose.utils.logging import get_logger
from cosypose.datasets.samplers import DistributedSceneSampler
import cosypose.utils.tensor_collection as tc
from cosypose.utils.distributed import get_world_size, get_rank, get_tmp_dir

from torch.utils.data import DataLoader
from open3d.visualization import rendering


logger = get_logger(__name__)


class MultiviewPredictionRunner:
    def __init__(self, scene_ds, batch_size=1, cache_data=False, n_workers=4):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        assert batch_size == 1, 'Multiple view groups not supported for now.'
        sampler = DistributedSceneSampler(scene_ds, num_replicas=self.world_size, rank=self.rank)
        self.sampler = sampler
        dataloader = DataLoader(scene_ds, batch_size=batch_size,
                                num_workers=n_workers,
                                sampler=sampler,
                                collate_fn=self.collate_fn)

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader
        output = Path('/home/ubuntu/synthetic_pose_estimation/cosypose/local_data/visualizations/real_data_rend')
        os.makedirs(output, exist_ok=True)
        self.detection_folder = output / "detections"
        self.renders_folder = output / "renders"

    def collate_fn(self, batch):
        batch_im_id = -1

        cam_infos, K = [], []
        det_infos, bboxes = [], []
        for n, data in enumerate(batch):
            assert n == 0
            images, masks, obss = data
            for c, obs in enumerate(obss):
                batch_im_id += 1
                frame_info = obs['frame_info']
                im_info = {k: frame_info[k] for k in ('scene_id', 'view_id', 'group_id')}
                im_info.update(batch_im_id=batch_im_id)
                cam_info = im_info.copy()

                K.append(obs['camera']['K'])
                cam_infos.append(cam_info)

                for o, obj in enumerate(obs['objects']):
                    obj_info = dict(
                        label=
                        ['name'],
                        score=1.0,
                    )
                    obj_info.update(im_info)
                    bboxes.append(obj['bbox'])
                    det_infos.append(obj_info)
        if len(bboxes)==0:
            gt_detections = tc.PandasTensorCollection(
            infos=pd.DataFrame(det_infos),
            bboxes=torch.as_tensor([]),
        )
        else:
            gt_detections = tc.PandasTensorCollection(
                infos=pd.DataFrame(det_infos),
                bboxes=torch.as_tensor(np.stack(bboxes)),
            )
        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
        )
        data = dict(
            images=images,
            cameras=cameras,
            gt_detections=gt_detections,
        )
        return data

    def get_predictions(self, pose_predictor, mv_predictor,
                        detections=None,
                        n_coarse_iterations=1, n_refiner_iterations=1,
                        sv_score_th=0.0, skip_mv=True, detector=None,
                        use_detections_TCO=False):
        # assert detections is not None
        # if detections is not None:
        #     mask = (detections.infos['score'] >= sv_score_th)
        #     detections = detections[np.where(mask)[0]]
        #     detections.infos['det_id'] = np.arange(len(detections))
        #     det_index = detections.infos.set_index(['scene_id', 'view_id']).sort_index()
        
        predictions = defaultdict(list)
        
        DEBUG_DATA_DIR = LOCAL_DATA_DIR / 'debug_data'
        renders_folder = output / "renders"
        detections_debug = dict()
        for frame_id, data in enumerate(tqdm(self.dataloader)):
            images = data['images'].cuda().float().permute(0, 3, 1, 2) / 255
            cameras = data['cameras'].cuda().float()
            gt_detections = data['gt_detections'].cuda().float()
            n_gt_dets = len(gt_detections)
            if n_gt_dets==0:
                continue
            scene_id = np.unique(gt_detections.infos['scene_id'])
            view_ids = np.unique(gt_detections.infos['view_id'])
            group_id = np.unique(gt_detections.infos['group_id'])
            
            detections_debug[frame_id] = dict()
            detections_debug[frame_id]['gt_detections'] = gt_detections.tensors
            logger.debug(f"{'-'*80}")
            logger.debug(f'Scene: {scene_id}')
            logger.debug(f'Views: {view_ids}')
            logger.debug(f'Group: {group_id}')
            logger.debug(f'Image has {n_gt_dets} gt detections. (not used)')

            detections = detector.get_detections(
            images=images,
            one_instance_per_class=False,
            )
            boxes = np.array(detections.bboxes.cpu()).astype(int)
            np_img = (
                np.array(images[0].permute(1, 2, 0).cpu()).astype(np.uint8).copy()
                )
            fig, ax = plt.subplots()
            ax.imshow(np_img)
            for i in range(boxes.shape[0]):
                top_left = tuple(boxes[i, :2].tolist())
                bottom_right = tuple(boxes[i, 2:].tolist())
                
                rect = patches.Rectangle((top_left[0], bottom_right[1]), bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                fig.text(top_left[0], top_left[1], f"{detections.infos['label'][i]}")
                # cv2.putText(np_img, detections.infos['label'][i], (bottom_right[0] - 70, bottom_right[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            detection_path = self.detection_folder / f"{frame_id}.png"
            fig.savefig(detection_path)
            plt.clf()
            plt.imshow(np_img, interpolation='nearest')
            plt.savefig(self.detection_folder / f'original_image{frame_id}.png')
            # det_index = detections.infos.set_index(['scene_id', 'view_id']).sort_index()

            # if detections is not None:
            #     print("detections", detections, "cameras", cameras)
            #     keep_ids, batch_im_ids = [], []
            #     for group_name, group in cameras.infos.groupby(['scene_id', 'view_id']):
            #         if group_name in det_index.index:
            #             other_group = det_index.loc[group_name]
            #             print("other_group", other_group)
            #             keep_ids_ = other_group['det_id']
            #             batch_im_id = np.unique(group['batch_im_id']).item()
            #             print("len(keep_ids_)", keep_ids_)
            #             if isinstance(keep_ids_, np.int64):
            #                 keep_ids_ = np.asarray([keep_ids_])
            #                 batch_im_ids.append(np.ones(1) * batch_im_id)
            #             else:
            #                 batch_im_ids.append(np.ones(len(keep_ids_)) * batch_im_id)
            #             keep_ids.append(keep_ids_)
            #     if len(keep_ids) > 0:
            #         keep_ids = np.concatenate(keep_ids)
            #         batch_im_ids = np.concatenate(batch_im_ids)
            #     detections_ = detections[keep_ids]
            #     detections_debug[frame_id]['detections'] = detections_.tensors # already cpu numpy
            #     detections_.infos['batch_im_id'] = np.array(batch_im_ids).astype(np.int)
            # else:
            #     raise ValueError('No detections')
            # detections_ = detections_.cuda().float()
            # detections_.infos['group_id'] = group_id.item()
            
            mask = detections.infos['label'].isin(['obj_000001','obj_000004'])
            detections = tc.PandasTensorCollection(
                infos=detections.infos[mask],
                bboxes=detections.bboxes[mask])

            sv_preds, mv_preds = dict(), dict()
            if len(detections) > 0:
                # just load detection from model output
                data_TCO_init = detections if use_detections_TCO else None
                detections__ = detections if not use_detections_TCO else None
                candidates, sv_preds = pose_predictor.get_predictions(
                    images, cameras.K, detections=detections__,
                    n_coarse_iterations=n_coarse_iterations,
                    data_TCO_init=data_TCO_init,
                    n_refiner_iterations=n_refiner_iterations,
                )
                candidates.register_tensor('initial_bboxes', detections.bboxes)

                if not skip_mv:
                    mv_preds = mv_predictor.predict_scene_state(
                        candidates, cameras,
                    )
            logger.debug(f"{'-'*80}")
            logger.debug(f'len of detections. {len(detections)}')
            for k, v in sv_preds.items():
                predictions[k].append(v.cpu())

            for k, v in mv_preds.items():
                predictions[k].append(v.cpu())
            print("predictions", predictions)
        detection_debug_fpath = DEBUG_DATA_DIR / "detection_debug.pth.tar"
        # torch.save(detections_debug, detection_debug_fpath)
        predictions = dict(predictions)
        logger.debug(f'predictions. {predictions}')
        width, height = 224, 224
        render = rendering.OffscreenRenderer(width, height, headless=True)
        pinhole = o3d.camera.PinholeCameraIntrinsic(width, height, camera_json['fx'], camera_json['fy'], camera_json['cx'], camera_json['cy'])
        render.scene.camera.set_projection(pinhole.intrinsic_matrix, 0.01, 10.0, float(width), float(height))
        center = [0, 0, 1]  # look_at target
        eye = [0, 0, 0]  # camera position
        up = [0, 1, 0]  # camera orientation
        render.scene.camera.look_at(center, eye, up)
        o3d_render_path = str(renders_folder / f"output_{img_id}.png")
        o3d.io.write_image(o3d_render_path, img_o3d, 9)

        mtl = o3d.visualization.rendering.Material()  # or MaterialRecord(), for later versions of Open3D
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        for pose in predictions.poses:
            mesh_t = deepcopy(mesh).transform(pose)
        for k, v in predictions.items():
            predictions[k] = tc.concatenate(v)
        return predictions
