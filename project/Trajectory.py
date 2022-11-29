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
        # [timestamp]
        self.breakpoints = []

    def get_traj(self):
        return self.traj
    
    def get_breakpoints(self):
        return self.breakpoints

    # Add intermediate point in the trajectory
    def append_point(self,
                     timestamp, 
                     pose,
                     grip_pose,
                     brick_n,
                     metainfo,
                     bounds,
                     breakpoint = False,
                     interpolate = False):
        # Adjust time
        base_t = self.traj[-1][0] if len(self.traj) else 0
        ts = base_t + timestamp

        # Append points (interpolate if requested)
        if interpolate:
            kStep = 0.1
            k = PiecewisePose.MakeLinear([base_t + kStep, ts], 
                                        [self.traj[-1][1], pose])
            for i in np.arange(base_t + kStep, ts + kStep, kStep):
                self.traj.append([i,
                                 k.GetPose(i),
                                 grip_pose,
                                 brick_n,
                                 metainfo,
                                 bounds,
                                 None])
        else:
            self.traj.append([ts, pose, grip_pose, brick_n, metainfo, bounds, None])

        # Add to breakpoints
        if breakpoint:
            self.breakpoints.append(ts)

    # Merge this trajectory with `traj_to_merge`, putting traj after
    def merge_in(self, traj_to_merge):
        base_t = self.traj[-1][0]

        # Merge trajectories
        for trj in traj_to_merge.get_traj():
            trj[0] = base_t + trj[0]
            self.traj.append(trj)

        # Merge breakpooints
        for bp in traj_to_merge.get_breakpoints():
            self.breakpoints.append(base_t + bp)

    # Shrink trajectories by scaling all timestamps for each point
    def slow_down(self, k):
        for trj in self.traj:
            trj[0] = trj[0] * k
        self.breakpoints = [x * k for x in self.breakpoints]

    # Dump each point based on mask corresponding to
    # [timestamp, pose, grip_pose, metainfo, breakpoint]
    def dump_trajectories(self, mask=[False, False, False, False, False, False, False]):
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
        self.traj[i][6] = q_sol

    def get_qs(self, i):
        return self.traj[i][6]

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
                                                        np.array([t[6] for t in self.traj])[:, 0:7].T)
        #
        finger_traj = self.form_iiwa_finger_traj()
        #
        return (traj, finger_traj)
