"""SceneManager compatibility for gsplat v1.5.3 on pycolmap 4.1.x.

gsplat v1.5.3's example loader imports the unrelated legacy
``pycolmap.SceneManager`` API.  The reconstruction pipeline needs the official
COLMAP bindings instead, so this adapter exposes only the attributes consumed
by ``examples/datasets/colmap.py`` while retaining official pycolmap globally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


def _as_dict(map_like):
    return {int(key): value for key, value in map_like.items()}


def _model_name(camera) -> str:
    name = getattr(camera, "model_name", "")
    if name:
        return str(name).replace("CameraModelId.", "")
    model = getattr(camera, "model", "")
    return getattr(model, "name", str(model)).replace("CameraModelId.", "")


def _cam_from_world_matrix(image) -> np.ndarray:
    transform = image.cam_from_world
    if callable(transform):
        transform = transform()
    return np.asarray(transform.matrix(), dtype=np.float64)


@dataclass
class _ImageAdapter:
    image: object

    @property
    def camera_id(self) -> int:
        return int(self.image.camera_id)

    @property
    def name(self) -> str:
        return str(self.image.name)

    @property
    def tvec(self) -> np.ndarray:
        return _cam_from_world_matrix(self.image)[:, 3].copy()

    def R(self) -> np.ndarray:
        return _cam_from_world_matrix(self.image)[:, :3].copy()


class _CameraAdapter:
    def __init__(self, camera):
        self._camera = camera
        self.width = int(camera.width)
        self.height = int(camera.height)
        self.camera_type = _model_name(camera)

        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        params = np.asarray(camera.params, dtype=np.float64)
        self.k1 = self.k2 = self.k3 = self.k4 = 0.0
        self.p1 = self.p2 = 0.0
        if self.camera_type == "SIMPLE_RADIAL":
            self.k1 = float(params[3])
        elif self.camera_type == "RADIAL":
            self.k1, self.k2 = map(float, params[3:5])
        elif self.camera_type in {"OPENCV", "OPENCV_FISHEYE"}:
            self.k1, self.k2, self.p1, self.p2 = map(float, params[4:8])
            if self.camera_type == "OPENCV_FISHEYE":
                self.k3, self.k4 = self.p1, self.p2


class SceneManager:
    """Subset of the legacy SceneManager API used by gsplat's loader."""

    def __init__(self, colmap_dir):
        self.colmap_dir = Path(colmap_dir)
        self._reconstruction = pycolmap.Reconstruction(str(self.colmap_dir))

        cameras = _as_dict(self._reconstruction.cameras)
        images = _as_dict(self._reconstruction.images)
        registered = {int(i) for i in self._reconstruction.reg_image_ids()}

        self.cameras = {
            camera_id: _CameraAdapter(camera)
            for camera_id, camera in cameras.items()
        }
        self.images = {
            image_id: _ImageAdapter(image)
            for image_id, image in images.items()
            if image_id in registered
        }
        self.name_to_image_id = {
            image.name: image_id for image_id, image in self.images.items()
        }

        points = _as_dict(self._reconstruction.points3D)
        point_ids = sorted(points)
        self.points3D = np.asarray(
            [points[point_id].xyz for point_id in point_ids], dtype=np.float32
        ).reshape(-1, 3)
        self.point3D_errors = np.asarray(
            [points[point_id].error for point_id in point_ids], dtype=np.float32
        )
        self.point3D_colors = np.asarray(
            [points[point_id].color for point_id in point_ids], dtype=np.uint8
        ).reshape(-1, 3)
        self.point3D_id_to_point3D_idx = {
            point_id: index for index, point_id in enumerate(point_ids)
        }
        self.point3D_id_to_images = {
            point_id: [
                (int(element.image_id), int(element.point2D_idx))
                for element in points[point_id].track.elements
                if int(element.image_id) in self.images
            ]
            for point_id in point_ids
        }

    def load_cameras(self):
        return None

    def load_images(self):
        return None

    def load_points3D(self):
        return None
