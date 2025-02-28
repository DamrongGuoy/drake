"""
Modified from Sergey's run_demo.py to Drake's source built.

bazel run //tmp_Sergey_run_demo:py/run_demo -- --path /path/to/gso_mini/

For example,

bazel run //tmp_Sergey_run_demo:run_demo -- --path /home/damrongguoy/project/2025-02-26_Sergey_run_demo_gso_mini/gso_mini/

"""
import os
import glob
import math
import argparse

from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ApplyVisualizationConfig
from pydrake.visualization import VisualizationConfig


def create_scene(path, sim_time_step):
    # Clean up the Meshcat instance.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Weld the table to the world so that it's fixed during the simulation.
    parser.AddModels(table_top_sdf_file)
    table_frame = plant.GetFrameByName("table_top_center")
    plant.WeldFrames(plant.world_frame(), table_frame)

    # Loading models.
    mesh_paths = glob.glob(os.path.join(path, '*/meshes/*.sdf'))
    models_used = []
    for mesh_path in mesh_paths:
        try:
            parser.AddModels(mesh_path)
            models_used.append(mesh_path)
        except Exception as e:
            print(f"Failed to load {mesh_path}: {e}")

    # Finalize the plant after loading the scene.
    plant.Finalize()
    # We use the default context to calculate the transformation of the table
    # in world frame but this is NOT the context the Diagram consumes.
    plant_context = plant.CreateDefaultContext()
    # Set the initial pose for the free bodies
    X_WorldTable = table_frame.CalcPoseInWorld(plant_context)

    # Grid parameter
    num_objects = len(models_used)
    num_columns = math.ceil(math.sqrt(num_objects))  # Closest square layout
    num_rows = math.ceil(num_objects / num_columns)  # Compute required rows
    spacing = 0.4  # Adjust spacing

    # Compute offsets to center the grid at (0, 0)
    x_offset = (num_columns - 1) * spacing / 2  # Center horizontally
    y_offset = (num_rows - 1) * spacing / 2  # Center vertically

    # Set pose
    for i, mesh_path in enumerate(models_used):
        # Compute grid position
        row = i // num_columns
        col = i % num_columns
        x = col * spacing - x_offset  # Centering in x
        y = row * spacing - y_offset  # Centering in y
        z = 0.4 * (x + 1)

        name = os.path.basename(mesh_path).split('.')[0].replace('_tetwild', '')
        print("Adding", name, "at", x, y)
        body_name = f"{name}_body_link"  # Replace with actual body names if known
        obj_body = plant.GetBodyByName(body_name)
        X_TableObj = RigidTransform(p=[x, y, z])  # Aligned to xz grid
        X_WorldObj = X_WorldTable.multiply(X_TableObj)
        plant.SetDefaultFreeBodyPose(obj_body, X_WorldObj)

    # Add visualization to see the geometries.
    config = VisualizationConfig(publish_contacts=False)
    ApplyVisualizationConfig(config=config, builder=builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram


def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)
    return simulator


def run_simulation(path, sim_time_step):
    diagram = create_scene(path, sim_time_step)
    simulator = initialize_simulation(diagram)
    meshcat.StartRecording()
    finish_time = 0.1 if test_mode else 2.0
    simulator.AdvanceTo(finish_time)
    meshcat.StopRecording()
    meshcat.PublishRecording()


def bazel_chdir():
    """When using `bazel run`, the current working directory ("cwd") of the
    program is set to a deeply-nested runfiles directory, not the actual cwd.
    In case relative paths are given on the command line, we need to restore
    the original cwd so that those paths resolve correctly.
    """
    if 'BUILD_WORKSPACE_DIRECTORY' in os.environ:
        os.chdir(os.environ['BUILD_WORKSPACE_DIRECTORY'])


if __name__ == '__main__':

    bazel_chdir()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to mesh files.",
    )
    args = parser.parse_args()

    meshcat = StartMeshcat()
    # Run the simulation with a small time step. Try gradually increasing it!

    temp_dir = temp_directory()

    # Create a table top SDFormat model.
    table_top_sdf_file = os.path.join(temp_dir, "table_top.sdf")
    table_top_sdf = """<?xml version="1.0"?>
    <sdf version="1.7">
      <model name="table_top">
        <link name="table_top_link">
          <visual name="visual">
            <pose>0 0 0.445 0 -0.4 0</pose>
            <geometry>
              <box>
                <size>3 3 0.05</size>
              </box>
            </geometry>
            # <material>
            #  <diffuse>0.9 0.8 0.7 1.0</diffuse>
            # </material>
          </visual>
          <collision name="collision">
            <pose>0 0 0.445  0 -0.4 0</pose>
            <geometry>
              <box>
                <size>3 3 0.05</size>
              </box>
            </geometry>
          <drake:proximity_properties>
            <drake:compliant_hydroelastic/>
            <drake:hydroelastic_modulus> 1e7 </drake:hydroelastic_modulus>
          </drake:proximity_properties>
          </collision>
          
        </link>
        <frame name="table_top_center">
          <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
        </frame>
      </model>
    </sdf>

    """

    with open(table_top_sdf_file, "w") as f:
        f.write(table_top_sdf)
    # test_mode = True if "TEST_SRCDIR" in os.environ else False
    test_mode = False
    run_simulation(args.path, sim_time_step=1e-2)
