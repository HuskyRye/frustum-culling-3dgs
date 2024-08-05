#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import open3d as o3d


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args)

        # Build frustum culling index
        print("Building frustum culling index, will happen only the first time you open the scene.")
        self.frustum_index = defaultdict(set)
        for camera in tqdm(self.getTrainCameras(), "Culling progress"):
            culled_points_mask = self.frustum_culling(scene_info.point_cloud.points, camera)
            # print(camera.image_name)
            self.visualize_culling(scene_info.point_cloud, culled_points_mask.cpu(), camera)
        #     for gaussian in gaussians:
        #         if gaussian in caemra:
        #             self.frustum_index[camera.image_name].add(gaussian)

    def visualize_culling(self, point_cloud, mask, camera):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.points)
        colors = point_cloud.colors.copy()
        colors[mask] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = camera.R.T
        extrinsic[:3, 3] = camera.T
        view_control = vis.get_view_control()
        camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_parameters.intrinsic = view_control.convert_to_pinhole_camera_parameters().intrinsic
        camera_parameters.extrinsic = extrinsic
        view_control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

        vis.run()
        vis.destroy_window()

    def frustum_culling(self, points, camera):
        points_homogeneous = torch.cat([torch.tensor(points, device="cuda"), torch.ones(points.shape[0], 1, device="cuda")], dim=1)
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/826
        points_ndc = torch.mm(points_homogeneous, camera.full_proj_transform)
        points_ndc[:, :3] /= points_ndc[:, 3].unsqueeze(1)
        mask = (
            (points_ndc[:, 0].abs() <= 1)
            & (points_ndc[:, 1].abs() <= 1)
            & (points_ndc[:, 2].abs() <= 1)
        )
        return mask

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
