#include "drake/geometry/proximity/optimize_sdfield.h"

#include <array>
#include <iostream>
#include <utility>
#include <vector>

#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

Vector3d SDFieldOptimizer::CalcLaplacianNewPosition(int v) {
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

void SDFieldOptimizer::LaplacianBoundary(const double alpha) {
  for (int i = 0; i < std::ssize(boundary_to_volume_); ++i) {
    int bv = boundary_to_volume_.at(i);
    Vector3d target = CalcLaplacianNewPosition(bv);
    // x^k (current iterate), xv (varitional new position):
    // x^{k+1} = (1-alpha) * x^k + alpha* xv
    vertices_.at(bv) = (1.0 - alpha) * vertices_.at(bv) + alpha * target;
  }
}

void SDFieldOptimizer::SpringBoundary(const double target_boundary_distance,
                                      const double beta) {
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
}

void SDFieldOptimizer::LaplacianInterior(const double alpha) {
  for (int v = 0; v < std::ssize(vertices_); ++v) {
    // Fix the boundary vertices in this step.
    if (volume_to_boundary_.contains(v)) {
      continue;
    }
    Vector3d target = CalcLaplacianNewPosition(v);
    // x^k (current iterate), xv (varitional new position):
    // x^{k+1} = (1-alpha) * x^k + alpha* xv
    vertices_.at(v) = (1.0 - alpha) * vertices_.at(v) + alpha * target;
  }
}

VolumeMesh<double> SDFieldOptimizer::OptimizeVertex() {
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
  evolving_sdfield_ =
      MakeEmPressSDField(evolving_volume_mesh_, original_boundary_.tri_mesh());
  WriteVolumeMeshFieldLinearToVtk(fmt::format("iteration_{:04d}.vtk", 0),
                                  "SignedDistance(meters)", evolving_sdfield_,
                                  "Optimized EmbeddedSignedDistanceField");

  double previous_rms_error =
      CalcRMSErrorOfSDField(evolving_sdfield_, original_boundary_.tri_mesh());
  std::cout << previous_rms_error << std::endl;

  // These parameters need tuning.
  const double alpha = 0.3;
  const double beta = 0.3;
  const double target_boundary_distance = 1e-3;  // 1mm tolerance.
  const int num_global_iterations = 100;
  for (int time = 1; time <= num_global_iterations; ++time) {
    // Laplacian smoothing the boundary vertices "tangentially". It tends
    // to move the boundary vertices into the interior of the original
    // surface. Next step, we will move them back outside the original surface.
    LaplacianBoundary(beta / 10);

    // Use the spring model to move the boundary vertices towards the
    // target_boundary_distance outside the original surface.
    SpringBoundary(target_boundary_distance, beta);

    LaplacianInterior(alpha);

    evolving_volume_mesh_ =
        VolumeMesh<double>{std::vector<VolumeElement>{tetrahedra_},
                           std::vector<Vector3d>{vertices_}};
    evolving_sdfield_ = MakeEmPressSDField(evolving_volume_mesh_,
                                           original_boundary_.tri_mesh());
    const double rms_error =
        CalcRMSErrorOfSDField(evolving_sdfield_, original_boundary_.tri_mesh());
    std::cout << rms_error << std::endl;
    WriteVolumeMeshFieldLinearToVtk(fmt::format("iteration_{:04d}.vtk", time),
                                    "SignedDistance(meters)", evolving_sdfield_,
                                    "Optimized EmbeddedSignedDistanceField");

    if (std::abs(rms_error - previous_rms_error) < 1e-6) {
      break;
    } else {
      previous_rms_error = rms_error;
    }
  }

  return evolving_volume_mesh_;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
