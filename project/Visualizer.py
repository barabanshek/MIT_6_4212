# Bookkeeping
import numpy as np
from pydrake.all import (MeshcatVisualizer, BaseField, Fields,
                         MeshcatVisualizerParams, Rgba,
                         StartMeshcat, PointCloud)
from Trajectory import *
from IKOptimizationController import *

#
# Visualizer for things
#
class Visualizer:
    def __init__(self, meshcat, brick_geom):
        self.meshcat_ = meshcat
        self.brick_geom_ = brick_geom

    def visualize_brick(self, brick_geom):
        kStep = 0.0075
        kMargin = 0.02
        points = []
        brick_geom = brick_geom - kMargin
        for i in np.arange(-brick_geom[0]/2, brick_geom[0]/2 + kStep, kStep):
            for j in np.arange(-brick_geom[1]/2, brick_geom[1]/2 + kStep, kStep):
                for k in np.arange(-brick_geom[2]/2, brick_geom[2]/2 + kStep, kStep):
                    points.append([i, j, k])
        points = np.array(points)
        cloud = PointCloud(points.shape[0])
        cloud.mutable_xyzs()[:] = points.T

        return cloud

    def visualize_bricks(self, bricks, col):
        brick = self.visualize_brick(self.brick_geom_)
        for idx, b in enumerate(bricks):
            self.meshcat_.SetObject("/brick_vis/" + str(idx) + str(col), brick, rgba=col, point_size=0.005)
            self.meshcat_.SetTransform("/brick_vis/" + str(idx) + str(col), RigidTransform(b))

    def clear_vis(self):
        self.meshcat_.Delete("/brick_vis")
