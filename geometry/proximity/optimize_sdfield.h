#pragma once

#include <unordered_map>
#include <vector>

#include "drake/common/sorted_pair.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/sorted_triplet.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_mesh_refiner.h"

namespace drake {
namespace geometry {
namespace internal {

Eigen::Vector3d CalcSimpleAveragePosition(
    int v, const VolumeMesh<double>& mesh,
    const std::vector<std::vector<int>>& vertex_to_tetrahedra);

// Signed Distance Field Optimizer.
// Quick hack to reuse code in VolumeMeshRefiner. For simplicity, we abuse
// class hierarchy.
class SDFieldOptimizer : VolumeMeshRefiner {
 public:
  explicit SDFieldOptimizer(
      const VolumeMeshFieldLinear<double, double>& sdfield_M,
      const TriangleSurfaceMesh<double>& original_M)
      : VolumeMeshRefiner(sdfield_M.mesh()),
        original_boundary_(TriangleSurfaceMesh<double>(original_M)) {}

  struct RelaxationParameters {
    double alpha_exterior = 0.03;            // dimensionless
    double alpha = 0.3;                      // dimensionless
    double beta = 0.3;                       // dimensionless
    double target_boundary_distance = 1e-3;  // meters
  };

  VolumeMesh<double> OptimizeVertex(const struct RelaxationParameters& params);

 protected:
  void LaplacianBoundary(double alpha);
  void SpringBoundary(double target_boundary_distance, double beta);
  void LaplacianInterior(double alpha);

  // Map to a vertex in the support volume mesh Ω from a vertex in its
  // triangulated boundary mesh ∂Ω.
  // The boundary_to_volume[i] = v means the i-th vertex of ∂Ω corresponds to
  // the v-th vertex of Ω.
  std::vector<int> boundary_to_volume_;
  std::unordered_map<int, int> volume_to_boundary_;
  // TODO(DamrongGuoy):  For simplicity, we initialize support_boundary_mesh_
  //  and evolving_volume_mesh_ to temporary meshes because
  //  TriangleSurfaceMesh and VolumeMesh do not have default constructors
  //  (no concepts of "empty" meshes). They will be override when the
  //  optimization starts. Consider a cleaner scheme.
  //
  // TODO(DamrongGuoy): Remove them if they turn out to have no use.
  //
  TriangleSurfaceMesh<double> support_boundary_mesh_{
      {{0, 1, 2}},
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
       Eigen::Vector3d::UnitY()}};
  VolumeMesh<double> evolving_volume_mesh_{
      {{0, 1, 2, 3}},
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
       Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()}};
  VolumeMeshFieldLinear<double, double> evolving_sdfield_{
      {0, 0.1, 0.2, 0.3}, &evolving_volume_mesh_};

  // For signed-distance query to the original mesh during optimization.
  const MeshDistanceBoundary original_boundary_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
