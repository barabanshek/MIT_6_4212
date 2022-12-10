from multiprocessing import Process, Manager, cpu_count
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
from TrajectoryBuilder import *

class HighPerformanceTrajectoryBuilder:
    def __init__(self, planer, X_WBrickSource, brick_to_cover, num_of_workers, q0, robot_pose=None):
        self.planer = planer
        self.q0 = q0
        self.brick_to_cover = brick_to_cover
        self.num_of_bricks = len(brick_to_cover)
        self.X_WBrickSource = X_WBrickSource
        self.robot_pose = robot_pose

        #
        print("Initializing HPC trajectory builder with ", num_of_workers, " workers on ", cpu_count(), " core machine")

        # Instantiate multiple trajectory builders and partition work
        bricks_per_worker = (int)(self.num_of_bricks / num_of_workers)
        current_bricks = 0
        self.traj_builders = {}
        for i in range(num_of_workers):
            if i == num_of_workers - 1:
                remaining_bricks = self.num_of_bricks - current_bricks
                self.traj_builders[i] = brick_to_cover[current_bricks : current_bricks + remaining_bricks]
            else:
                self.traj_builders[i] = brick_to_cover[current_bricks : current_bricks + bricks_per_worker]
            current_bricks = current_bricks + bricks_per_worker

    def worker_(self, worker_id, brick_poses, covered_traj, X_WBrickSource, q0, robot_pose):
        for b_id, b_pose in brick_poses:
            # Set points
            b = b_pose
            X_WBrickTarget = RigidTransform(RotationMatrix(), b[:3])

            # Generate trajectory options based on different bypass trajectories
            R = 0.4  # bypass radius
            if not robot_pose == None:
                Rc = (robot_pose.translation()[0], robot_pose.translation()[1])  # bypass center
            else:
                Rc = (0, 0)
            P1 = (X_WBrickSource.translation()[0], X_WBrickSource.translation()[1])
            P2 = (X_WBrickTarget.translation()[0], X_WBrickTarget.translation()[1])
            bypass_P_pairs = TrajectoryBuilder.generate_bypass(R, Rc, P1, P2)

            # Generate trajectories
            covered_traj_collection = []
            for bypass_p in bypass_P_pairs:
                trj_builder_for_brick = TrajectoryBuilder(X_WBrickSource)
                # Grab
                trj_builder_for_brick.gen_grab_brick_traj(b_id)
                # Move to place
                if not bypass_p == [P1, P2]:
                    trj_builder_for_brick.gen_move_to_place_traj(
                        RigidTransform(np.array([bypass_p[0][0], bypass_p[0][1], X_WBrickTarget.translation()[2]])),
                        b_id)
                    trj_builder_for_brick.gen_move_to_place_traj(
                        RigidTransform(np.array([bypass_p[1][0], bypass_p[1][1], X_WBrickTarget.translation()[2]])),
                        b_id)
                    trj_builder_for_brick.gen_move_to_place_traj(X_WBrickTarget, b_id)
                else:
                    trj_builder_for_brick.gen_move_to_place_traj(X_WBrickTarget, b_id)
                # Place
                trj_builder_for_brick.gen_place_brick_traj(X_WBrickTarget, b[3], b_id)
                # Return
                if not bypass_p == [P1, P2]:
                    trj_builder_for_brick.gen_move_to_place_traj(
                        RigidTransform(np.array([bypass_p[1][0], bypass_p[1][1], X_WBrickTarget.translation()[2]])),
                        b_id)
                    trj_builder_for_brick.gen_move_to_place_traj(
                        RigidTransform(np.array([bypass_p[0][0], bypass_p[0][1], X_WBrickTarget.translation()[2]])),
                        b_id)
                    trj_builder_for_brick.gen_return_to_source_traj(b_id)
                else:
                    trj_builder_for_brick.gen_return_to_source_traj(b_id)

                # Solve IK for the trajectories
                res, q = trj_builder_for_brick.solve_IK(q0, robot_pose)
                if res == False:
                    continue
                else:
                    covered_traj_collection.append(trj_builder_for_brick.get_trajectories())

            # If there are multiple feasible trajectories, select the shortest
            if (len(covered_traj_collection) == 1):
                covered_traj.append((b_id, covered_traj_collection[0]))
            elif (len(covered_traj_collection) > 1):
                shortest = 999999
                shortest_traj = None
                for t in covered_traj_collection:
                    length = t.get_length()
                    if length < shortest:
                        shortest = length
                        shortest_traj = t
                covered_traj.append((b_id, shortest_traj))

    def solve(self, initial_traj):
        # {thread -> covered_traj}
        threads = {}
        # Map
        for trj_b_id, bricks in self.traj_builders.items():
            manager = Manager()
            covered_traj = manager.list()

            brick_poses = []
            for b_id in bricks:
                brick_poses.append((b_id, self.planer.get_brick_poses()[b_id]))

            t = Process(target=self.worker_, args=(trj_b_id, brick_poses, covered_traj, self.X_WBrickSource, self.q0, self.robot_pose))
            t.start()
            threads[t] = covered_traj

        # Reduce
        total_trajectories = {}
        for tk, tv in threads.items():
            tk.join()
            for tv_0, tv_1 in tv:
                total_trajectories[tv_0] = tv_1

        total_bricks_failed = []
        total_bricks_covered = []
        for b in self.brick_to_cover:
            if b in total_trajectories.keys():
                initial_traj.get_trajectories().merge_in(total_trajectories[b])
                total_bricks_covered.append(b)
            else:
                total_bricks_failed.append(b)

        return total_bricks_covered, total_bricks_failed
