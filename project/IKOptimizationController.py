# Bookkeeping
import numpy as np
from pydrake.multibody import inverse_kinematics
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis, BasicVector,
                         ConstantVectorSource, DiagramBuilder,
                         FindResourceOrThrow, Integrator, JacobianWrtVariable,
                         LeafSystem, MeshcatVisualizer,
                         MeshcatVisualizerParams, MultibodyPlant,
                         MultibodyPositionToGeometryPose, Parser,
                         PiecewisePose, Quaternion, RigidTransform,
                         RollPitchYaw, RotationMatrix, SceneGraph, Simulator,
                         StartMeshcat, TrajectorySource, GenerateHtml, GetDrakePath, PiecewisePolynomial, Solve)

#
# IK-based controller class
#
class IKOptimizationController():
    def __init__(self):
        self.plant, _ = self.CreateIiwaControllerPlant()
        self.world_frame = self.plant.world_frame()
        self.gripper_frame = self.plant.GetFrameByName("body")

    def CreateIiwaControllerPlant(self):
        robot_sdf_path = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")
        gripper_sdf_path = FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf")
        sim_timestep = 1e-3
        plant_robot = MultibodyPlant(sim_timestep)
        parser = Parser(plant=plant_robot)
        parser.AddModelFromFile(robot_sdf_path)
        parser.AddModelFromFile(gripper_sdf_path)
        plant_robot.WeldFrames(
            frame_on_parent_F=plant_robot.world_frame(),
            frame_on_child_M=plant_robot.GetFrameByName("iiwa_link_0"))
        plant_robot.WeldFrames(
            frame_on_parent_F=plant_robot.GetFrameByName("iiwa_link_7"),
            frame_on_child_M=plant_robot.GetFrameByName("body"),
            X_FM=RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), np.array([0, 0, 0.114]))
        )
        plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
        plant_robot.Finalize()

        link_frame_indices = []
        for i in range(8):
            link_frame_indices.append(
                plant_robot.GetFrameByName("iiwa_link_" + str(i)).index())

        return plant_robot, link_frame_indices

    def AddOrientationConstraint(self, ik, R_WG, bounds):
        ik.AddOrientationConstraint(
            frameAbar=self.world_frame, R_AbarA=R_WG,
            frameBbar=self.gripper_frame, R_BbarB=RotationMatrix(),
            theta_bound=bounds
        )

    def AddPositionConstraint(self, ik, p_WG_lower, p_WG_upper):
        ik.AddPositionConstraint(
            frameA=self.world_frame, frameB=self.gripper_frame, p_BQ=np.zeros(3),
            p_AQ_lower=p_WG_lower, p_AQ_upper=p_WG_upper)

    def solve(self, traj_fragment, q_guess):
        ik = inverse_kinematics.InverseKinematics(self.plant, with_joint_limits=True)
        q_variables = ik.q()
        prog = ik.prog()

        self.AddPositionConstraint(ik, traj_fragment[1].translation(), traj_fragment[1].translation())
        self.AddOrientationConstraint(ik, traj_fragment[1].rotation(), traj_fragment[5])

        prog.SetInitialGuess(q_variables, q_guess)
        result = Solve(prog)
        if not result.is_success():
            return None
        else:
            return result.GetSolution(q_variables)
