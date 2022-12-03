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

    # Check if the line connecting P1 and P2 penetrates the robot defined as a circle R at Rc
    # Return True is there is penetration, False otherwise
    @staticmethod
    def check_for_penetration_with_the_robot(Rc, R, P1, P2):
        P1_X = P1[0] - Rc[0]
        P1_Y = P1[1] - Rc[1]
        P2_X = P2[0] - Rc[0]
        P2_Y = P2[1] - Rc[1]
        #
        k = (P1_X - P2_X)/(P1_Y - P2_Y)
        b = P1_X - P1_Y * k
        L = (k, b)
        #
        D = 4 * (R ** 2 - b ** 2 + (k ** 2)*(R ** 2))
        return False if D < 0 else True

    # Get an equation of a tangent line to the radius R in center Rc passing throug the point Pa
    @staticmethod
    def get_tangental_line(Pa, Rc, R):
        Pa_X = Pa[0] - Rc[0]
        Pa_Y = Pa[1] - Rc[1]
        #
        k1 = (-Pa_X*Pa_Y + R * np.sqrt(Pa_X * Pa_X + Pa_Y * Pa_Y - R * R))/(R * R - Pa_X * Pa_X)
        b1 = Pa_Y - k1 * Pa_X + Rc[1] - k1 * Rc[0]
        #
        k2 = (-Pa_X*Pa_Y - R * np.sqrt(Pa_X * Pa_X + Pa_Y * Pa_Y - R * R))/(R * R - Pa_X * Pa_X)
        b2 = Pa_Y - k2 * Pa_X + Rc[1] - k2 * Rc[0]
        #
        return (k1, b1), (k2, b2)

    # Get the point of intersection of two lines L1 and L2 specified as (k, b)
    @staticmethod
    def get_line_intersection_point(L1, L2):
        x = (L2[1] - L1[1])/(L1[0] - L2[0])
        y = L1[0] * x + L1[1]
        #
        return (x, y)

    # Generate a trajectory from point P1 to P2 such that the grip never penetrates the area defined as a circle R at Rc
    @staticmethod
    def generate_bypass_trajectory(R, Rc, P1, P2, plot=False):
        # Solve penetration
        L1, L11 = TrajectoryBuilder.get_tangental_line(P1, Rc, R)
        L2, L22 = TrajectoryBuilder.get_tangental_line(P2, Rc, R)
        # Get four points
        points = []
        points.append(TrajectoryBuilder.get_line_intersection_point(L1, L2))
        points.append(TrajectoryBuilder.get_line_intersection_point(L1, L22))
        points.append(TrajectoryBuilder.get_line_intersection_point(L11, L2))
        points.append(TrajectoryBuilder.get_line_intersection_point(L11, L22))

        return points

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
                                     'move brick #' + str(brick_n),
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

    def gen_return_to_source_traj(self, brick_n=0):
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
    def solve_IK(self, q_nominal, debug=False):
        ik_controller = IKOptimizationController()

        # Solve for all other positions
        q_prev = q_nominal
        for i in range(len(self.trajectory.get_traj())):
            try:
                q_knots = ik_controller.solve(self.trajectory.get_traj()[i], q_prev)
            except:
                if debug:
                    print("Failed to solve IK for trajectory point: ", self.trajectory.get_traj()[i][4])
                return (False, q_nominal)

            if q_knots is None:
                if debug:
                    print("Failed to solve IK for trajectory point: ", self.trajectory.get_traj()[i][4])
                return (False, q_nominal)
            else:
                q_prev = q_knots
                self.trajectory.set_ik_solution(i, q_knots)
        #
        return (True, q_prev)

    # Merge trajectory builders
    def merge(self, trj_builder):
        self.trajectory.merge_in(trj_builder.trajectory)

    def get_trajectory_points_for_brick(self, brick):
        return np.array([t[1].translation() for t in self.trajectory.get_traj() if t[3] == brick])
