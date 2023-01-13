#include <memory>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/fem/deformable_body_config.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/deformable_model.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/multibody/tree/rigid_body.h"
#include "drake/multibody/tree/spatial_inertia.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

DEFINE_double(simulation_time, 2.0, "Desired duration of the simulation [s].");
DEFINE_double(realtime_rate, 0.1, "Desired real time rate.");
DEFINE_double(time_step, 1.0e-2,
              "Discrete time step for the system [s]. Must be positive.");
// Larger E, stiffer materials. Rubber is about 1e7 to 1e8 Pascals.
// Steel is about 2e11 Pascals.
DEFINE_double(E, 1e6, "Young's modulus of the deformable body [Pa].");
DEFINE_double(nu, 0.4, "Poisson's ratio of the deformable body, unitless.");
DEFINE_double(density, 1000, "Mass density of the deformable body [kg/m³].");
DEFINE_double(beta, 0.005,
              "Stiffness damping coefficient for the deformable body [1/s].");

using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::GeometryInstance;
using drake::geometry::IllustrationProperties;
using drake::geometry::Mesh;
using drake::geometry::ProximityProperties;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::Body;
using drake::multibody::CoulombFriction;
using drake::multibody::DeformableBodyId;
using drake::multibody::DeformableModel;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using drake::multibody::fem::DeformableBodyConfig;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

namespace drake {
namespace examples {
namespace multibody {
namespace bunny {
namespace {

int do_main() {
  systems::DiagramBuilder<double> builder;

  MultibodyPlantConfig plant_config;
  plant_config.time_step = FLAGS_time_step;
  /* Deformable simulation only works with SAP solver. */
  plant_config.discrete_contact_solver = "sap";

  auto [plant, scene_graph] = AddMultibodyPlant(plant_config, &builder);

  /* Minimum required proximity properties for rigid bodies to interact with
   deformable bodies.
   1. A valid Coulomb friction coefficient, and
   2. A resolution hint. (Rigid bodies need to be tesselated so that collision
   queries can be performed against deformable geometries.) */
  ProximityProperties rigid_proximity_props;
  /* Set the friction coefficient close to that of rubber against rubber. */
  const CoulombFriction<double> kSurfaceFriction(1.0, 1.0);
  {
    AddContactMaterial({}, {}, kSurfaceFriction, &rigid_proximity_props);
    rigid_proximity_props.AddProperty(geometry::internal::kHydroGroup,
                                      geometry::internal::kRezHint, 1.0);
  }

  /* Set up an orange ground. */
  {
    // 30 x 30 x 1 centimeter is about twice the bunny in width.
    Box ground{0.30, 0.30, 0.01};
    const RigidTransformd X_WG(Eigen::Vector3d{0, 0, -0.005});
    plant.RegisterCollisionGeometry(plant.world_body(), X_WG, ground,
                                    "ground_collision", rigid_proximity_props);
    IllustrationProperties illustration_props;
    illustration_props.AddProperty("phong", "diffuse",
                                   Vector4d(0.7, 0.5, 0.4, 0.8));
    plant.RegisterVisualGeometry(plant.world_body(), X_WG, ground,
                                 "ground_visual", illustration_props);
  }

  /* Drop a yellow brick. */
  // 15 x 15 x 4 centimeter is about the same as the bunny's width.
  {
    Box brick_geometry{0.15, 0.15, 0.04};
    const RigidBody<double>& brick_body = plant.AddRigidBody(
        "Brick",
        SpatialInertia<double>{/*mass*/ 0.5, /*p_BoBcm*/ Vector3d::Zero(),
                                        UnitInertia<double>::SolidBox(
                                            brick_geometry.width(),
                                            brick_geometry.depth(),
                                            brick_geometry.height())});
    const RigidTransformd X_WC(Eigen::Vector3d{0, 0, 0.20});
    plant.RegisterCollisionGeometry(brick_body, X_WC, brick_geometry,
                                    "brick_collision", rigid_proximity_props);
    IllustrationProperties brick_illustration_props;
    brick_illustration_props.AddProperty("phong", "diffuse",
                                         Vector4d(0.7, 0.7, 0, 0.8));
    plant.RegisterVisualGeometry(brick_body, X_WC, brick_geometry,
                                 "brick_visual", brick_illustration_props);
  }

  /* Set up the deformable bunny. */
  auto owned_deformable_model =
      std::make_unique<DeformableModel<double>>(&plant);
  DeformableModel<double>* deformable_model = owned_deformable_model.get();

  DeformableBodyConfig<double> deformable_config;
  deformable_config.set_youngs_modulus(FLAGS_E);
  deformable_config.set_poissons_ratio(FLAGS_nu);
  deformable_config.set_mass_density(FLAGS_density);
  deformable_config.set_stiffness_damping_coefficient(FLAGS_beta);

  auto bunny_mesh = std::make_unique<Mesh>(
      FindResourceOrThrow("drake/examples/multibody/bunny/bunny_fixed.vtk"),
      1.0);
  // Bunny's bounding box is about 15 x 12 x 16 centimeters like this.
  // X: -0.09, 0.06
  // Y: -0.06, 0.06
  // Z:  0.03, 0.19
  // The center of bounding box is around (-0.015, 0, 0.11).
  //
  // To pose the bunny upside down on the floor. Rotate 180 degrees around X,
  // then move it up by 19 centimeters.
  // RollPitchYawd(M_PI, 0, 0), Vector3d(0, 0, 0.19)
  //
  // To pose the bunny upside down above the floor. Rotate 180 degrees around X,
  // then move it up by 29 centimeters.
  // RollPitchYawd(M_PI, 0, 0), Vector3d(0, 0, 0.29)
  //
  // To pose the bunny on the floor, displace Z down 3.5 centimeters
  // (RollPitchYawd(0, 0, 0), Vector3d(0, 0, -0.035)).
  const RigidTransformd X_WB(RollPitchYawd(0, 0, 0),
                             Vector3d(0, 0, -0.035));

  auto bunny_instance =
      std::make_unique<GeometryInstance>(X_WB, std::move(bunny_mesh), "bunny");
  /* Minimumly required proximity properties for deformable bodies: A valid
   Coulomb friction coefficient. */
  {
    ProximityProperties deformable_proximity_props;
    AddContactMaterial({}, {}, kSurfaceFriction, &deformable_proximity_props);
    bunny_instance->set_proximity_properties(deformable_proximity_props);
  }

  deformable_model->RegisterDeformableBody(std::move(bunny_instance),
                                           deformable_config, 1.0);

  plant.AddPhysicalModel(std::move(owned_deformable_model));

  /* All rigid and deformable models have been added. Finalize the plant. */
  plant.Finalize();

  /* It's essential to connect the vertex position port in DeformableModel to
   the source configuration port in SceneGraph when deformable bodies are
   present in the plant. */
  builder.Connect(
      deformable_model->vertex_positions_port(),
      scene_graph.get_source_configuration_port(plant.get_source_id().value()));

  /* Add a visualizer that emits LCM messages for visualization. */
  geometry::DrakeVisualizerParams params;
  params.role = geometry::Role::kIllustration;
  geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                           params);

  auto diagram = builder.Build();
  std::unique_ptr<Context<double>> diagram_context =
      diagram->CreateDefaultContext();

  /* Build the simulator and run! */
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.Initialize();
  simulator.set_target_realtime_rate(FLAGS_realtime_rate);
  simulator.AdvanceTo(FLAGS_simulation_time);

  return 0;
}

}  // namespace
}  // namespace bunny
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "This is a demo used to showcase deformable body simulations in Drake. "
      "Launch meldis before running this example. "
      "Refer to README for instructions on meldis as well as optional flags.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::bunny::do_main();
}
