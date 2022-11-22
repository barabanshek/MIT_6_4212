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

    def visualize_brick(self, brick_geom, with_margin = True):
        kStep = 0.0075
        kMargin = 0.02 if with_margin else 0
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

    def visualize_bricks(self, bricks, col, with_margin = True):
        brick = self.visualize_brick(self.brick_geom_, with_margin)
        for idx, b in enumerate(bricks):
            self.meshcat_.SetObject("/brick_vis/" + str(idx) + str(col), brick, rgba=col, point_size=0.005)

            #
            R = RotationMatrix.MakeZRotation(-np.pi/2) if b[3] == 0 else RotationMatrix()
            self.meshcat_.SetTransform("/brick_vis/" + str(idx) + str(col), RigidTransform(R, b[:3]))

    def clear_vis(self):
        self.meshcat_.Delete("/brick_vis")

    def visualize_coverage(self, covered_bricks, not_covered_bricks):
        # Visualize successful bricks first
        kColGreen = Rgba(0.61, 1, 0.60)
        self.visualize_bricks(covered_bricks, kColGreen)

        # Visualize failed bricks next
        kColRead = Rgba(1, 0.44, 0.33)
        self.visualize_bricks(not_covered_bricks, kColRead)

    def visualize_plan(self, bricks):
        kColGreen = Rgba(0.61, 1, 0.60)
        self.visualize_bricks(bricks, kColGreen, False)
