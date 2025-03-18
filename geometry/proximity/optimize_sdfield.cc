#include "drake/geometry/proximity/optimize_sdfield.h"

#include <array>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/geometry/proximity/volume_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

Vector3d SDFieldOptimizer::CalcVariationalNewPosition(int v) {
  // If we allow a dangling vertex, remove this throw and skip the
  // dangling vertex v from the optimization.
  DRAKE_THROW_UNLESS(vertex_to_tetrahedra_.at(v).size() > 0);
  Vector3d accumulator{0, 0, 0};
  for (int tet : vertex_to_tetrahedra_.at(v)) {
    Vector3d tet_centroid =
        (evolving_volume_mesh_.vertex(
             evolving_volume_mesh_.element(tet).vertex(0)) +
         evolving_volume_mesh_.vertex(
             evolving_volume_mesh_.element(tet).vertex(1)) +
         evolving_volume_mesh_.vertex(
             evolving_volume_mesh_.element(tet).vertex(2)) +
         evolving_volume_mesh_.vertex(
             evolving_volume_mesh_.element(tet).vertex(3))) /
        4;
    accumulator += tet_centroid;
  }
  return accumulator / vertex_to_tetrahedra_.at(v).size();
}

VolumeMesh<double> SDFieldOptimizer::OptimizeMeshVertices() {
  tetrahedra_ = input_mesh_.tetrahedra();
  vertices_ = input_mesh_.vertices();
  ResetVertexToTetrahedra();

  // TODO(DamrongGuoy): Move initialization into its own function.
  boundary_to_volume_.clear();
  volume_to_boundary_.clear();
  support_boundary_mesh_ = ConvertVolumeToSurfaceMeshWithBoundaryVertices(
      input_mesh_, &boundary_to_volume_, nullptr);
  for (int i = 0; i < support_boundary_mesh_.num_vertices(); ++i) {
    const int v = boundary_to_volume_.at(i);
    volume_to_boundary_[v] = i;
  }
  evolving_volume_mesh_ =
      VolumeMesh<double>{std::vector<VolumeElement>{tetrahedra_},
                         std::vector<Vector3d>{vertices_}};

  // Sanity check: perform simple Laplacian smoothing, fixing the boundary mesh.
  const double alpha = 0.3;
  const double beta = 0.3;
  // TODO(DamrongGuoy): Should this constant be the same as the out_offset
  //  tolerance in MakeEmPressMesh() { const double out_offset = 1e-3; } ?
  //  or a fraction of grid resolution?
  //
  // 5mm tolerance.
  const double target_boundary_distance = 5e-3;
  for (int time_iteration = 0; time_iteration < 20; ++time_iteration) {
    // Smooth the boundary subject to distance constraints.
    for (int i = 0; i < std::ssize(boundary_to_volume_); ++i) {
      int bv = boundary_to_volume_.at(i);
      SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
          vertices_.at(bv), original_boundary_.tri_mesh(),
          original_boundary_.tri_bvh(),
          std::get<FeatureNormalSet>(original_boundary_.feature_normal()));
      // x^{k+1} = x^{k} - beta * (phi - d) * normal, beta is dimensionless
      vertices_.at(bv) =
          vertices_.at(bv) -
          beta * (d.signed_distance - target_boundary_distance) * d.gradient;
    }
    // TODO(DamrongGuoy): Call VolumeMesh::SetAllPositions(VectorX<double>)
    //  instead.  I'm not fluent in Egien::Ref<const VectorX<T>> yet.
    evolving_volume_mesh_ =
        VolumeMesh<double>{std::vector<VolumeElement>{tetrahedra_},
                           std::vector<Vector3d>{vertices_}};

    for (int v = 0; v < std::ssize(vertices_); ++v) {
      // Fix the boundary vertices for now.
      if (volume_to_boundary_.contains(v)) {
        continue;
      }
      Vector3d target = CalcVariationalNewPosition(v);
      // x^k (current iterate), xv (varitional new position):
      // x^{k+1} = (1-alpha) * x^k + alpha* xv
      vertices_.at(v) = (1.0 - alpha) * vertices_.at(v) + alpha * target;
    }
    // TODO(DamrongGuoy): Call VolumeMesh::SetAllPositions(VectorX<double>)
    //  instead.  I'm not fluent in Egien::Ref<const VectorX<T>> yet.
    evolving_volume_mesh_ =
        VolumeMesh<double>{std::vector<VolumeElement>{tetrahedra_},
                           std::vector<Vector3d>{vertices_}};
  }

  return evolving_volume_mesh_;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
