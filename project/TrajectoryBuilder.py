# Bookkeeping
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis, BasicVector,
                         ConstantVectorSource, DiagramBuilder,
                         FindResourceOrThrow, Integrator, JacobianWrtVariable,
                         LeafSystem, MeshcatVisualizer,
                         MeshcatVisualizerParams, MultibodyPlant,
                         MultibodyPositionToGeometryPose, Parser,
                         PiecewisePose, Quaternion, RigidTransform,
                         RollPitchYaw, RotationMatrix, SceneGraph, Simulator,
                         StartMeshcat, TrajectorySource, GenerateHtml, GetDrakePath, PiecewisePolynomial, Solve)
from Trajectory import *
from IKOptimizationController import *
from Visualizer import *

#
# Trajectory builder: make trajectories described by `class Trajectory`
#
class TrajectoryBuilder:
    # Init trajectory builder with the brick source at X_WBrickSource
    def __init__(self, system, X_WBrickSource):
        #
        self.X_WBrickSource = X_WBrickSource
        self.system = system

        # define some constant reference poses
        self.X_BrickSourcePreG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),
                                               np.array([0, 0, 0.25]))
        self.X_BrickSourceG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),
                                            np.array([0, 0, 0.09]))
        self.X_BrickTargetPreG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),
                                               np.array([0, 0, 0.25]))
        self.X_BrickTargetG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),
                                            np.array([0, 0, 0.09]))
        self.finger_opened = np.array([0.15])
        self.finger_closed = np.array([0.05])

        #
        self.trajectory = Trajectory()

    def gen_initial_traj(self):
        self.trajectory.append_point(0,
                                     self.system.get_X_WG(),
                                     self.finger_opened,
                                     0,
                                     'initial',
                                     0.0)
        self.trajectory.append_point(2,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_opened,
                                     0,
                                     'calibration',
                                     0.0,
                                     interpolate=True)

    def gen_grab_brick_traj(self, brick_n=0):
        # Initial point
        self.trajectory.append_point(0.00001,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_opened,
                                     brick_n,
                                     'grab brick #' + str(brick_n) + ' initial ',
                                      0.0,
                                      breakpoint=True,
                                      interpolate=False)

        # Approach
        self.trajectory.append_point(2,
                                     self.X_WBrickSource @ self.X_BrickSourceG,
                                     self.finger_opened,
                                     brick_n,
                                     'grab brick #' + str(brick_n) + ' approach',
                                      0.0,
                                      interpolate=True)

        # Grab
        self.trajectory.append_point(0.5,
                                     self.X_WBrickSource @ self.X_BrickSourceG,
                                     self.finger_closed,
                                     brick_n,
                                     'grab brick #' + str(brick_n) + ' finger close',
                                     0.0,
                                     interpolate=False)

        # Withdraw   
        self.trajectory.append_point(2,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_closed,
                                     brick_n,
                                     'grab brick #' + str(brick_n) + ' withdraw',
                                     0.0,
                                     interpolate=True)

    # Generate move trajectory to the point X_WBrickTarget
    def gen_move_to_place_traj(self, X_WBrickTarget, brick_n=0):
        self.trajectory.append_point(2,
                                     X_WBrickTarget @ self.X_BrickTargetPreG,
                                     self.finger_closed,
                                     brick_n,
                                     'grab brick #' + str(brick_n) + ' withdraw',
                                     0.0,
                                     interpolate=True)

    def gen_place_brick_traj(self, X_WBrickTarget, orientation, brick_n=0):
        # Turn if needed
        R_Orientation = RotationMatrix.MakeYRotation(-np.pi/2) if orientation == 0 else RotationMatrix()
        R = RigidTransform(R_Orientation, np.array([0.0, 0.0, 0.0]))

        if orientation == 0:
            self.trajectory.append_point(0.5,
                                        X_WBrickTarget @ (self.X_BrickTargetPreG @ R),
                                        self.finger_closed,
                                        brick_n,
                                        'place brick #' + str(brick_n) + ' turn',
                                        0.0,
                                        interpolate=True)

        # Approach
        self.trajectory.append_point(2,
                                     X_WBrickTarget @ (self.X_BrickTargetG @ R),
                                     self.finger_closed,
                                     brick_n,
                                     'place brick #' + str(brick_n) + ' approach',
                                     0.0,
                                     interpolate=True)

        # Open
        self.trajectory.append_point(0.5,
                                     X_WBrickTarget @ (self.X_BrickTargetG @ R),
                                     self.finger_opened,
                                     brick_n,
                                     'place brick #' + str(brick_n) + ' open',
                                     0.0,
                                     interpolate=False)

        # Withdraw
        self.trajectory.append_point(2,
                                     X_WBrickTarget @ (self.X_BrickTargetPreG @ R),
                                     self.finger_opened,
                                     brick_n,
                                     'place brick #' + str(brick_n) + ' withdraw',
                                     0.0,
                                     interpolate=True)

        # Rotate back if needed
        if orientation == 0:
            self.trajectory.append_point(0.5,
                                        X_WBrickTarget @ self.X_BrickTargetPreG,
                                        self.finger_opened,
                                        brick_n,
                                        'place brick #' + str(brick_n) + ' turn',
                                        0.0,
                                        interpolate=True)

    def gen_return_to_source_traj(self, X_WBrickTarget, brick_n=0):
        self.trajectory.append_point(2,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_opened,
                                     brick_n,
                                     'move back #' + str(brick_n),
                                     0.0,
                                     interpolate=True)

    def get_trajectories(self):
        return self.trajectory

    # Solve IK offline using IKOptimizationController and store the result in the trajecotry[6]
    def solve_IK(self, q_nominal, stop_on_fail=True):
        ik_controller = IKOptimizationController()

        # Solve for all other positions
        num_err = 0
        q_prev = q_nominal
        for i in range(len(self.trajectory.get_traj())):
            q_knots = ik_controller.solve(self.trajectory.get_traj()[i], q_prev)

            if q_knots is None:
                print("Failed to solve IK for trajectory point: ", self.trajectory.get_traj()[i][4])
                if stop_on_fail:
                    return 1
                else:
                    num_err = num_err + 1
            else:
                q_prev = q_knots
                self.trajectory.set_ik_solution(i, q_knots)

        #
        return num_err

    # Merge trajectory builders
    def merge(self, trj_builder):
        self.trajectory.merge_in(trj_builder.trajectory)
