#include "drake/geometry/query_object.h"
#include "drake/geometry/query_results/signed_distance_pair.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {

void do_main() {
  // Three boxes.  Two on the outside are fixed.  One in the middle on a
  // prismatic joint.  The configuration space is a (convex) line segment
  // q ∈ (−1,1).
  const char urdf[] = R"""(
  <robot name="boxes">
    <link name="fixed">
      <collision name="right">
        <origin rpy="0 0 0" xyz="2 0 0"/>
        <geometry><box size="1 1 1"/></geometry>
      </collision>
      <collision name="left">
        <origin rpy="0 0 0" xyz="-2 0 0"/>
        <geometry><box size="1 1 1"/></geometry>
      </collision>
    </link>
    <joint name="fixed_link_weld" type="fixed">
      <parent link="world"/>
      <child link="fixed"/>
    </joint>
    <link name="movable">
      <collision name="center">
        <geometry><sphere radius="0.5"/></geometry>
      </collision>
    </link>
    <joint name="movable" type="prismatic">
      <axis xyz="1 0 0"/>
      <limit lower="-2" upper="2"/>
      <parent link="world"/>
      <child link="movable"/>
    </joint>
  </robot>
  )""";
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
  multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
  multibody::Parser parser(&plant);
  parser.AddModelsFromString(urdf, "urdf");
  plant.Finalize();
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      plant.GetMyMutableContextFromRoot(context.get());
  systems::Context<double>& scene_graph_context =
      scene_graph.GetMyMutableContextFromRoot(context.get());

  // Set the movable box to the boundary of collision.
  plant.SetPositions(&plant_context, Vector1d(1.0));

  const double kInfluenceDistance{5.0};
  auto query_object = scene_graph.get_query_output_port().Eval<
      geometry::QueryObject<double>>(scene_graph_context);
  const std::vector<geometry::SignedDistancePair<double>>
      signed_distance_pairs =
      query_object.ComputeSignedDistancePairwiseClosestPoints(
          kInfluenceDistance);
  for (const auto& pair : signed_distance_pairs) {
    log()->info("id_A = {}", pair.id_A);
    log()->info("id_B = {}", pair.id_B);
    log()->info("p_ACa = {}", fmt_eigen(pair.p_ACa.transpose()));
    log()->info("p_BCb = {}", fmt_eigen(pair.p_BCb.transpose()));
    log()->info("distance = {}", pair.distance);
    log()->info("nhat_BA_W = {}", fmt_eigen(pair.nhat_BA_W.transpose()));
    log()->info("");
  }
  // The first pair has distance zero because the sphere (id_B=25) touches the
  // right box (id_A=18) with nhat_BA_W = (1, 0, 0) pointing from the sphere
  // towards the box. In World X, the sphere covers [0.5, 1.5], and the
  // right box covers [1.5, 2.5]
  //                     ⦿[] →
  //
  // The second pair has distance +2 with the sphere (id_B=25) outside the
  // left box (id_A=21) with nhat_BA_W = (-1, 0, 0) pointing from the sphere
  // towards the box. In World X, the sphere covers [0.5, 1.5] and the left
  // box covers [-2.5,-1.5]
  //                 [] ←⦿
  //
  // [2023-07-18 17:45:31.971] [console] [info] id_A = 18
  // [2023-07-18 17:45:31.972] [console] [info] id_B = 25
  // [2023-07-18 17:45:31.972] [console] [info] p_ACa = -0.5    0    0
  // [2023-07-18 17:45:31.972] [console] [info] p_BCb = 0.5  -0  -0
  // [2023-07-18 17:45:31.972] [console] [info] distance = 0
  // [2023-07-18 17:45:31.972] [console] [info] nhat_BA_W =  1 -0 -0
  // [2023-07-18 17:45:31.972] [console] [info]
  // [2023-07-18 17:45:31.972] [console] [info] id_A = 21
  // [2023-07-18 17:45:31.972] [console] [info] id_B = 25
  // [2023-07-18 17:45:31.972] [console] [info] p_ACa = 0.5   0   0
  // [2023-07-18 17:45:31.972] [console] [info] p_BCb = -0.5   -0   -0
  // [2023-07-18 17:45:31.972] [console] [info] distance = 2
  // [2023-07-18 17:45:31.972] [console] [info] nhat_BA_W = -1 -0 -0
  // [2023-07-18 17:45:31.972] [console] [info]
}

}  // namespace drake

int main(int, char**) {
  drake::do_main();
  return 0;
}



