# Bookkeeping
import mpld3
import numpy as np
from IPython.display import HTML, display
from manipulation import running_as_notebook, FindResource
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import MakeManipulationStation
from pydrake.all import (AddMultibodyPlantSceneGraph, AngleAxis, BasicVector,
                         ConstantVectorSource, DiagramBuilder,
                         FindResourceOrThrow, JacobianWrtVariable,
                         LeafSystem, MeshcatVisualizer,
                         MeshcatVisualizerParams, MultibodyPlant,
                         MultibodyPositionToGeometryPose, Parser,
                         PiecewisePose, Quaternion, RigidTransform,
                         RollPitchYaw, RotationMatrix, SceneGraph, Simulator,
                         StartMeshcat, TrajectorySource, GenerateHtml, GetDrakePath, PiecewisePolynomial, Solve)


#
# Model definition
#
model_directives = """
directives:
"""

gModelScene = """
- add_frame:
    name: ground_origin
    X_PF:
      base_frame: world
      rotation: !Rpy { deg: [0.0, 0.0, 0.0 ]}
      translation: [0.0, 0.0, 0.0]

- add_model:
    name: ground
    file: package://manipulation/ground_model.sdf

- add_weld:
    parent: ground_origin
    child: ground::ground_base
"""

# Robot template
gRobotTemplate = """
- add_model:
    name: iiwa$
    file: package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa$::iiwa_link_0
    X_PC:
        translation: [X_PC$, Y_PC$, Z_PC$]
        rotation: !Rpy { deg: [0, 0, 0]}
- add_model:
    name: wsg$
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa$::iiwa_link_7
    child: wsg$::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
"""

# Brick template
gBrickString = """
- add_model:
    name: brick$
    file: package://manipulation/real_brick.sdf
"""


#
# iiwa robot class
#
class IIWA_Ilyich():
    def __init__(self, meshcat, num_of_bricks, brick_geom, robot_s, traj_s = None):
        self.brick_geom = brick_geom

        global gModelDerectives
        global gBrickString

        # set up robots
        self.robot_idxs_string = []
        model_directive = model_directives
        for i, robot in enumerate(robot_s):
            self.robot_idxs_string.append(str(i))
            robot_new = gRobotTemplate.replace("X_PC$", str(robot[0]))
            robot_new = robot_new.replace("Y_PC$", str(robot[1]))
            robot_new = robot_new.replace("Z_PC$", str(robot[2]))
            robot_new = robot_new.replace("iiwa$", 'iiwa' + str(i))
            robot_new = robot_new.replace("wsg$", 'wsg' + str(i))
            model_directive = model_directive + robot_new

        # set up scene
        model_directive = model_directive + gModelScene
        for i in range(0, num_of_bricks):
            brick_string_new = gBrickString.replace("brick$", "brick_" + str(i))
            model_directive = model_directive + brick_string_new

        # set up the system of manipulation station
        self.meshcat = meshcat
        self.station = MakeManipulationStation(model_directives=model_directive)
        builder = DiagramBuilder()
        builder.AddSystem(self.station)

        # set up plant
        self.plant = self.station.GetSubsystemByName("plant")

        # optionally add trajectory source
        if traj_s is not None:
            for i, traj_in in enumerate(traj_s):
                # traj for robot
                traj = traj_in[0]
                V_G_source = builder.AddSystem(TrajectorySource(traj))
                builder.Connect(V_G_source.get_output_port(),
                                self.station.GetInputPort("iiwa" + str(i) + "_position"))

                # and trajectory source for the grip fingers as well
                finger_traj = traj_in[1]
                wsg_source = builder.AddSystem(TrajectorySource(finger_traj))
                wsg_source.set_name("wsg" + str(i) + "_command")
                builder.Connect(wsg_source.get_output_port(), self.station.GetInputPort("wsg" + str(i) + "_position"))

        # visualization
        params = MeshcatVisualizerParams()
        params.delete_on_initialization_event = False
        self.visualizer = MeshcatVisualizer.AddToBuilder(
            builder, self.station.GetOutputPort("query_object"), self.meshcat, params)

        # build and add diagram
        self.diagram = builder.Build()
        self.world_frame = self.plant.world_frame()
        self.gripper_frame = []
        for i_s in self.robot_idxs_string:
            self.gripper_frame.append(self.plant.GetFrameByName('body', model_instance=self.plant.GetModelInstanceByName("wsg" + i_s)))

        # resolve context
        context = self.CreateDefaultContext()
        self.diagram.Publish(context)

    # Helper to visualize frame
    def visualize_frame(self, name, X_WF, length=0.15, radius=0.006):
        AddMeshcatTriad(self.meshcat, "painter/" + name,
                        length=length, radius=radius, X_PT=X_WF)

    # Helper to create default context
    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, context)
        station_context = self.diagram.GetMutableSubsystemContext(
            self.station, context)

        # provide initial states
        q0 = np.array([ 1.40666193e-05,  1.56461165e-01, -3.82761069e-05,
                       -1.32296976e+00, -6.29097287e-06,  1.61181157e+00, -2.66900985e-05])
        for i_s in self.robot_idxs_string:
            iiwa = self.plant.GetModelInstanceByName("iiwa" + i_s)
            self.plant.SetPositions(plant_context, iiwa, q0)
            self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
            wsg = self.plant.GetModelInstanceByName("wsg" + i_s)
            self.plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
            self.plant.SetVelocities(plant_context, wsg, [0, 0])        

        return context

    # Helper to get current grip position
    def get_X_WG(self, context=None):

        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)

        ret = []
        for gf in self.gripper_frame:
            X_WG = self.plant.CalcRelativeTransform(
                        plant_context,
                        frame_A=self.world_frame,
                        frame_B=gf)
            ret.append(X_WG)
        return ret

    # Lock bricks
    def lock_brick(self, context, brick_num):
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        brick_body = self.plant.GetBodyByName("base_link", self.plant.GetModelInstanceByName("brick_" + str(brick_num)))
        brick_body.Lock(plant_context)
        
    # Unlock bricks
    def unlock_brick(self, context, brick_num):
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        brick_body = self.plant.GetBodyByName("base_link", self.plant.GetModelInstanceByName("brick_" + str(brick_num)))
        brick_body.Unlock(plant_context)
    
    # Generate new brick
    def move_brick(self, context, X_WBrickSource, brick_num):
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        brick_body = self.plant.GetBodyByName("base_link", self.plant.GetModelInstanceByName("brick_" + str(brick_num)))
        self.plant.SetFreeBodyPose(plant_context, brick_body, X_WBrickSource)
        self.visualize_frame("brick_source", X_WBrickSource)

    def put_bricks_in_warehouse(self, context, wh_location, wh_size, brick_cnt):
        # Warehouse cell geometry -- a little bigger than bricks
        wh_cell_geom = self.brick_geom + np.array([0.02, 0.02, 0.02])

        # Put all bricks in the warehouse and lock them there
        # Grid cell spec: each warehouse cell is slightly bigger than the bricks
        brick_cnt = brick_cnt - 1
        for z in range (0, wh_size[2]):
            for i in range (0, wh_size[1]):
                for j in range (0, wh_size[0]):
                    X_WBrickWareHouse = RigidTransform(wh_location + wh_cell_geom * np.array([i, j, z]))
                    self.move_brick(context, X_WBrickWareHouse, brick_cnt)
                    self.lock_brick(context, brick_cnt)
                    if brick_cnt == 0:
                        return
                    else:
                        brick_cnt = brick_cnt - 1

    # Run simulation
    def work(self, simulator, context, sim_duration=20.0):
        if context == None:
            context = self.CreateDefaultContext()
        
        if simulator == None:
            simulator = Simulator(self.diagram, context)
            simulator.set_target_realtime_rate(1.0)

        duration = sim_duration if running_as_notebook else 0.01
        simulator.AdvanceTo(duration)
        
        return (simulator, context)
