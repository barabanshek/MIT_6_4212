# Bookkeeping
import numpy as np
from pydrake.all import (PiecewisePolynomial, PiecewisePose)

#
# Trajectory class: define and manipulate trajectories
#
class Trajectory():
    def __init__(self):
        # [timestamp, pose, grip_pose, brick_n, metainfo, bounds, qs]
        self.traj = []

    def get_traj(self):
        return self.traj

    # Add intermediate point in the trajectory
    def append_point(self,
                     timestamp, 
                     pose,
                     grip_pose,
                     brick_n,
                     metainfo,
                     bounds,
                     interpolate = False):
        # Adjust time
        base_t = 0
        ts = base_t + timestamp

        # Append points (interpolate if requested)
        if interpolate:
            kStep = 0.1
            k = PiecewisePose.MakeLinear([base_t + kStep, ts], 
                                        [self.traj[-1][1], pose])
            for i in np.arange(base_t + kStep, ts + kStep, kStep):
                self.traj.append([kStep,
                                 k.GetPose(i),
                                 grip_pose,
                                 brick_n,
                                 metainfo,
                                 bounds,
                                 None])
        else:
            self.traj.append([ts, pose, grip_pose, brick_n, metainfo, bounds, None])

    def append_point_simple(self, point):
        self.traj.append(point)

    # Merge this trajectory with `traj_to_merge`, putting traj after
    def merge_in(self, traj_to_merge):
        # Merge trajectories
        for trj in traj_to_merge.get_traj():
            self.traj.append(trj)

    # Shrink trajectories by scaling all timestamps for each point
    def slow_down(self, k):
        for trj in self.traj:
            trj[0] = trj[0] * k

    # Dump each point based on the mask
    def dump_trajectories(self, mask=[False, False, False, False, False, False, False]):
        for trj in self.traj:
            dump = list(filter(lambda x: x[1] == True, zip(trj, mask)))
            print([x[0] for x in dump])

    def form_iiwa_finger_traj(self):
        finger_traj = PiecewisePolynomial.FirstOrderHold([self.traj[0][0], self.traj[1][0]], 
                                                         np.hstack([[self.traj[0][2]], [self.traj[1][2]]]))
        for i in range(2, len(self.traj)):
            finger_traj.AppendFirstOrderSegment(self.traj[i][0], self.traj[i][2])
        
        #
        return finger_traj

    def set_ik_solution(self, i, q_sol):
        self.traj[i][6] = q_sol

    def get_qs(self, i):
        return self.traj[i][6]

    # Form iiwa grip and finger trajectories, grop trajectories are identified as joint positions (q);
    # Use this for IK-based control
    def form_iiwa_traj_q(self):
        #
        traj = PiecewisePolynomial.CubicShapePreserving(np.array([t[0] for t in self.traj]),
                                                        np.array([t[6] for t in self.traj])[:, 0:7].T)
        #
        finger_traj = self.form_iiwa_finger_traj()
        #
        return (traj, finger_traj)
