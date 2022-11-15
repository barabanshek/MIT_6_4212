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
                                            np.array([0, 0, 0.08]))
        self.X_BrickTargetPreG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),# @ RotationMatrix.MakeYRotation(-np.pi/2),
                                               np.array([0, 0, 0.25]))
        self.X_BrickTargetG = RigidTransform(RotationMatrix.MakeXRotation(-np.pi/2),# @ RotationMatrix.MakeYRotation(-np.pi/2),
                                            np.array([0, 0, 0.08]))
        self.finger_opened = np.array([0.15])
        self.finger_closed = np.array([0.05])

        #
        self.trajectory = Trajectory()

    def gen_initial_traj(self):
        self.trajectory.append_point(0,
                                     self.system.get_X_WG(),
                                     self.finger_opened,
                                     'initial',
                                     0.0)
        self.trajectory.append_point(2,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_opened,
                                     'calibration',
                                     0.0,
                                     True)

    def gen_grab_brick_traj(self, brick_n=0):
        # Two points
        X_B1 = self.X_WBrickSource @ self.X_BrickSourcePreG
        X_B2 = self.X_WBrickSource @ self.X_BrickSourceG
        # Open grip
        self.trajectory.append_point(0.00001,
                                     X_B1,
                                     self.finger_opened,
                                     'grab brick #' + str(brick_n) + ' finger open',
                                     0.0)
        # Approach
        traj1 = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[X_B1.translation()],
                                     [X_B2.translation()]]).T)
        for i in np.arange(0.0, 4.0, 0.5):
            self.trajectory.append_point(0.5,
                                         RigidTransform(X_B2.rotation(), traj1.value(i)),
                                         self.finger_opened,
                                         'grab brick #' + str(brick_n) + ' approach (' + str(i) + ')',
                                         0.0)
        # Grab
        self.trajectory.append_point(2,
                                     X_B2,
                                     self.finger_closed,
                                     'grab brick #' + str(brick_n) + ' finger close',
                                     0.0)
        # Withdraw
        traj2 = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[X_B2.translation()],
                                     [X_B1.translation()]]).T)
        for i in np.arange(0.0, 4.0, 0.5):
            self.trajectory.append_point(0.5,
                                         RigidTransform(X_B1.rotation(), traj2.value(i)),
                                         self.finger_closed,
                                         'grab brick #' + str(brick_n) + ' withdraw (' + str(i) + ')',
                                         0.0)

    # Generate move trajectory to the point X_WBrickTarget
    def gen_move_to_place_traj(self, X_WBrickTarget, brick_n=0):
        traj = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[(self.X_WBrickSource @ self.X_BrickSourcePreG).translation()],
                                     [(X_WBrickTarget @ self.X_BrickTargetPreG).translation()]]).T)

        for i in np.arange(0.0, 4.0, 0.1):
            R = RigidTransform(self.X_BrickSourcePreG.rotation(), traj.value(i))
            self.trajectory.append_point(0.1,
                                         R,
                                         self.finger_closed,
                                         'move #' + str(brick_n) + ' time: ' + str(i),
                                         0.1)

    def gen_place_brick_traj(self, X_WBrickTarget, brick_n=0):
        # Two points
        X_B1 = X_WBrickTarget @ self.X_BrickTargetPreG
        X_B2 = X_WBrickTarget @ self.X_BrickTargetG
        # Open grip
        self.trajectory.append_point(0.00001,
                                     X_B1,
                                     self.finger_closed,
                                     'grab brick #' + str(brick_n) + ' finger open',
                                     0.0)
        # Approach
        traj1 = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[X_B1.translation()],
                                     [X_B2.translation()]]).T)
        for i in np.arange(0.0, 4.0, 0.5):
            self.trajectory.append_point(0.5,
                                         RigidTransform(X_B2.rotation(), traj1.value(i)),
                                         self.finger_closed,
                                         'grab brick #' + str(brick_n) + ' approach (' + str(i) + ')',
                                         0.0)
        # Grab
        self.trajectory.append_point(2,
                                     X_B2,
                                     self.finger_opened,
                                     'grab brick #' + str(brick_n) + ' finger close',
                                     0.0)
        # Withdraw
        traj2 = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[X_B2.translation()],
                                     [X_B1.translation()]]).T)
        for i in np.arange(0.0, 4.0, 0.5):
            self.trajectory.append_point(0.5,
                                         RigidTransform(X_B1.rotation(), traj2.value(i)),
                                         self.finger_opened,
                                         'grab brick #' + str(brick_n) + ' withdraw (' + str(i) + ')',
                                         0.0)




        # self.trajectory.append_point(0.0001,
        #                              X_WBrickTarget @ self.X_BrickTargetPreG,
        #                              self.finger_closed,
        #                              'pre-put #' + str(brick_n),
        #                              0.0)
        # self.trajectory.append_point(4,
        #                              X_WBrickTarget @ self.X_BrickTargetG,
        #                              self.finger_closed,
        #                              'put #' + str(brick_n),
        #                              0.0)
        # self.trajectory.append_point(2,
        #                              X_WBrickTarget @ self.X_BrickTargetG,
        #                              self.finger_opened,
        #                              'put and open #' + str(brick_n),
        #                              0.0)
        # self.trajectory.append_point(4,
        #                              X_WBrickTarget @ self.X_BrickTargetPreG,
        #                              self.finger_opened,
        #                              'pre-put #' + str(brick_n),
        #                              0.0)

    def gen_return_to_source_traj(self, X_WBrickTarget, brick_n=0):
        traj = PiecewisePolynomial.FirstOrderHold(
                      [0.0, 4.0],
                          np.vstack([[(X_WBrickTarget @ self.X_BrickTargetPreG).translation()],
                                     [(self.X_WBrickSource @ self.X_BrickSourcePreG).translation()]]).T)

        for i in np.arange(0.0, 4.0, 0.1):
            R = RigidTransform(self.X_BrickSourcePreG.rotation(), traj.value(i))
            self.trajectory.append_point(0.1,
                                         R,
                                         self.finger_opened,
                                         'move_back #' + str(brick_n) + ' time: ' + str(i),
                                         0.1)
        self.trajectory.append_point(4,
                                     self.X_WBrickSource @ self.X_BrickSourcePreG,
                                     self.finger_opened,
                                     'go_back_end/calibration #' + str(brick_n),
                                     0.0,
                                     True)

    def get_trajectories(self):
        return self.trajectory
    
    # Solve IK offline using IKOptimizationController and store the result in the trajecotry[6]
    def solve_IK(self):
        ik_controller = IKOptimizationController()
        q_nominal = np.array([ 0., 0.6, 0., 0.0, 0.0, 1, 0.0, 0., 0.]) # nominal joint for joint-centering.

        # Solve for all other positions
        num_errors = 0
        q_prev = q_nominal
        for i in range(len(self.trajectory.get_traj())):
            q_knots = ik_controller.solve(self.trajectory.get_traj()[i], q_prev)

            if q_knots is None:
                print("Failed to solve IK for trajectory point: ", self.trajectory.get_traj()[i][3])
                num_errors = num_errors + 1
                self.trajectory.get_traj()[i][6] = q_prev
            else:
                q_prev = q_knots
                self.trajectory.get_traj()[i][6] = q_knots

        # Print stat
        print("IK solver: failed to solve for ", num_errors, " positions")
