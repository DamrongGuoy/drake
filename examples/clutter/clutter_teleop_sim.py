import argparse
import numpy as np
import os

from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.multibody.plant import (
    MultibodyPlant,
    AddMultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.meshcat import (
    ContactVisualizer,
    ContactVisualizerParams
)
from pydrake.multibody.parsing import (
    ProcessModelDirectives,
    LoadModelDirectives,
    Parser,
)
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    PrintSimulatorStatistics,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import (
    LeafSystem,
    DiagramBuilder,
)
from pydrake.systems.primitives import (
    ConstantVectorSource,
    TrajectorySource,
    Multiplexer,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)

def run(*, local_dir):
    """Runs a simulation from the given model directives.
    """
    if not local_dir:
        local_dir = os.path.dirname(__file__)

    builder = DiagramBuilder()

    # Plant configuration (timestep and contact parameters).
    plant_config = MultibodyPlantConfig()
    plant_config.time_step = 11e-3
    plant_config.discrete_contact_solver = "sap"

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=plant_config,
        builder=builder)

    # Load simulation model directives.
    parser = Parser(sim_plant)
    directives = LoadModelDirectives(local_dir + "/sim_directives.dmd.yaml")

    # Add a custom package map to the local repository location.
    package_map = parser.package_map()
    package_map.Add("clutter", local_dir)

    ProcessModelDirectives(
        directives=directives,
        plant=sim_plant,
        parser=parser)

    # Now the simulation plant is complete.
    sim_plant.Finalize()

    paddle_model_index = sim_plant.GetModelInstanceByName("paddle")

    # Create the Meshcat visualizer.
    meshcat = StartMeshcat()

    # Add and connect the paddle's controller.
    paddle_pid_controller = builder.AddSystem(
        PidController(kp=[20, 20, 20], ki=[0, 0, 0], kd=[2.5, 2.5, 0.075]))

    # Add trajectory source to the builder, and connect to the controller.
    path = PiecewisePolynomial.FirstOrderHold(
        breaks=np.array([5, 15, 25, 35]),   # time points
        samples=np.array(([[0,  0,  0, 0],  # paddle x values
                           [0, -1, -1, 1],  # paddle y values
                           [0, -1,  1, 1]   # paddle theta values
                           ])))
    trajectory_source = builder.AddSystem(TrajectorySource(
        trajectory=path, output_derivative_order=1))
    builder.Connect(trajectory_source.get_output_port(),
                    paddle_pid_controller.get_input_port_desired_state())

    sim_paddle_instance = sim_plant.GetModelInstanceByName("paddle")
    builder.Connect(
        sim_plant.get_state_output_port(sim_paddle_instance),
        paddle_pid_controller.get_input_port_estimated_state())
    builder.Connect(
        paddle_pid_controller.get_output_port_control(),
        sim_plant.get_actuation_input_port())

    # Set visualization configuration.
    visualization_config = VisualizationConfig()
    # We will enable contact visualization with our custom code.
    # fine mesh, publish_{period default 1/64} _contact{True}  40 seconds
    # fine mesh, publish_{period default 1/64} _contact{False} 28 seconds
    # fine mesh, publish_{period 0.05} _contact{True}          41 seconds
    # fine mesh, publish_{period 0.05} _contact{False}         27 seconds
    # medium mesh, publish_{period default 1/64} _contact{True} 22 seconds
    # medium mesh, publish_{period default 1/64} _contact{False} 17 seconds
    # medium mesh, publish_{period 0.05} _contact{True}        23 seconds
    # medium mesh, publish_{period 0.05} _contact{False}       17 seconds
    visualization_config.publish_contacts = True
    visualization_config.publish_proximity = False
    visualization_config.enable_alpha_sliders = True
    visualization_config.publish_period = 1.0 / 16.0;

    # Add visualization.
    ApplyVisualizationConfig(visualization_config, builder, meshcat=meshcat)

    # Simulator configuration (integrator and publisher parameters).
    simulator_config = SimulatorConfig(
        target_realtime_rate=1)

    # Build the diagram and its simulator.
    diagram = builder.Build()
    simulator = Simulator(diagram)
    ApplySimulatorConfig(simulator_config, simulator)

    simulator.AdvanceTo(15)

    PrintSimulatorStatistics(simulator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working_directory", required=False,
        help="Full path to the local directory for directives file and models folder.")
    args = parser.parse_args()
    run(local_dir=args.working_directory)


if __name__ == "__main__":
    main()
