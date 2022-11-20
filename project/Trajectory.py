# Bookkeeping
import numpy as np
from pydrake.all import (PiecewisePolynomial)

#
# Trajectory class: define and manipulate trajectories
#
class Trajectory():
    def __init__(self):
        # [timestamp, pose, grip_pose, brick_n, metainfo, breakpoint, bounds, qs]
        self.traj = []
        # [timestamp]
        self.breakpoints = []

    def get_traj(self):
        return self.traj
    
    def get_breakpoints(self):
        return self.breakpoints

    # Add intermediate point in the trajectory
    def append_point(self, timestamp, pose, grip_pose, brick_n, metainfo, bounds, breakpoint = False, calibration_q0 = None, with_merge = True):
        base_t = self.traj[-1][0] if len(self.traj) else 0
        ts = base_t + timestamp if with_merge else timestamp
        self.traj.append([ts, pose, grip_pose, brick_n, metainfo, breakpoint, bounds, None])
        if breakpoint:
            self.breakpoints.append(ts)

    # Merge this trajectory with `traj_to_merge`, putting traj after
    # Add a small constant as a gap between trajectories
    def merge_in(self, traj_to_merge):
        # Merge trajectories
        gap = 0.0001
        base_t = self.traj[-1][0] + gap
        for trj in traj_to_merge.get_traj():
            self.append_point(base_t + trj[0], trj[1], trj[2], trj[3], trj[4], trj[5], trj[6], trj[7])

    # Shrink trajectories by scaling all timestamps for each point
    def slow_down(self, k):
        for trj in self.traj:
            trj[0] = trj[0] * k
        self.breakpoints = [x * k for x in self.breakpoints]

    # Dump each point based on mask corresponding to
    # [timestamp, pose, grip_pose, metainfo, breakpoint]
    def dump_trajectories(self, mask=[False, False, False, False, False, False, False, False]):
        for trj in self.traj:
            dump = list(filter(lambda x: x[1] == True, zip(trj, mask)))
            print([x[0] for x in dump])

    def dump_breakpoints(self):
        for bp in self.breakpoints:
            print(bp)

    def form_iiwa_finger_traj(self):
        finger_traj = PiecewisePolynomial.FirstOrderHold([self.traj[0][0], self.traj[1][0]], 
                                                         np.hstack([[self.traj[0][2]], [self.traj[1][2]]]))
        for i in range(2, len(self.traj)):
            finger_traj.AppendFirstOrderSegment(self.traj[i][0], self.traj[i][2])
        
        #
        return finger_traj

    def set_ik_solution(self, i, q_sol):
        self.traj[i][7] = q_sol

    # Form iiwa grip and finger trajectories, grip trajectories are identified as end-effector poses;
    # Use this for PseudoInverseController-based control
    def form_iiwa_traj(self):
        #
        traj = PiecewisePose.MakeLinear(np.array([t[0] for t in self.traj]), 
                                        np.array([t[1] for t in self.traj]))
        #
        finger_traj = self.form_iiwa_finger_traj()
        #
        return (traj, finger_traj)

    # Form iiwa grip and finger trajectories, grop trajectories are identified as joint positions (q);
    # Use this for IK-based control
    def form_iiwa_traj_q(self):
        #
        traj = PiecewisePolynomial.CubicShapePreserving(np.array([t[0] for t in self.traj]),
                                                        np.array([t[7] for t in self.traj])[:, 0:7].T)
        #
        finger_traj = self.form_iiwa_finger_traj()
        #
        return (traj, finger_traj)
