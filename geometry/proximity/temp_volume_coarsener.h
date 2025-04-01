#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_mesh_refiner.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

/* Return a new coarsen tetrahedral mesh supporting the field.

 @param[in] sdf_M      signed distance field on a support mesh.
 @param[in] fraction   a number between 0 and 1 that control how aggressive
                       the coarsening process is. The target number of
                       tetrahedra equals the `fraction` times the number of
                       the given tetrahedra.  */
VolumeMesh<double> TempCoarsenVolumeMeshOfSdField(
    const VolumeMeshFieldLinear<double, double>& sdf_M, double fraction);

// Quadric Error Metric (similar to vtkUnstructuredGridQuadricDecimationQEF)
struct QEM {
  Eigen::Matrix4d A;
  Eigen::Vector4d b;
  double c;
};

class VolumeMeshCoarsener : VolumeMeshRefiner {
 public:
  VolumeMeshCoarsener(const VolumeMeshFieldLinear<double, double>& sdfield_M,
                      const TriangleSurfaceMesh<double>& original_M);

  VolumeMesh<double> coarsen(double fraction);

 protected:
  // Contract the edge between the v0-th vertex and the v1-th vertex to the
  // `new_position` with `new_scalar` value.
  //
  // The v1-th vertex will lose all its incident tetrahedra.
  // The v0-th vertex will gain all incident tetrahedra from v1-th vertex,
  // except the "deleted" tetrahedra sharing both vertices.
  //
  bool ContractEdge(int v0, int v1, const Eigen::Vector3d& new_position,
                    double new_scalar);

  static double CalcTetrahedronVolume(
      int tetrahedron_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices);

  static bool AreAllIncidentTetrahedraPositive(
      int vertex_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices,
      const std::vector<std::vector<int>>& vertex_to_tetrahedra,
      const double kTinyVolume);

  // Return true if all incident tetrahedra of the given vertex,
  // excluding the ones also incident to the excluded vertex, have
  // positive volumes.
  static bool AreAllMorphedTetrahedraPositive(
      int vertex_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices,
      const std::vector<std::vector<int>>& vertex_to_tetrahedra,
      int exclude_vertex_index, double kTinyVolume);

  const double kTinyVolume = 1e-12;  // cubic meters, 0.1x0.1x0.1-mm cube.

  // signed_distances[i] := the signed distance value of the i-th vertex.
  // As we perform edge contraction, the value of `signed_distances[i]` can
  // change since we might move the i-th vertex during edge contraction.
  // Or `signed_distances[i]` becomes irrelevant because the i-th vertex
  // was deleted by the edge contraction.
  std::vector<double> signed_distances_;

  // "is_vertex_deleted_[i]==true" means vertices_[i] was marked for deletion
  //                               from an edge contraction.
  std::vector<bool> is_vertex_deleted_;

  // "is_tet_deleted[i]==true" means tetrahedra_[i] was marked for deletion
  //                           from an edge contraction.
  std::vector<bool> is_tet_deleted_;
  int num_tet_deleted_{0};
  // "is_tet_morphed[i]==true"  means tetrahedra_[i] was changed by edge
  //                            contraction. One of its vertex has changed to
  //                            a new vertex or new position.
  std::vector<bool> is_tet_morphed_;

  // "boundary_to_volume_[u]==v" means support_boundary_mesh_.vertex(u)
  // corresponds to vertices_[v].
  std::vector<int> boundary_to_volume_;
  // "volume_to_boundary_.at(u)==v" means vertices_[u] corresponds to
  // support_boundary_mesh.vertex(v).  Since it's a map, some vertices_[i]
  // may not have a corresponding support_boundary_mesh.vertex(j), i.e.,
  // volume_to_boundary_.contains(i) == false.
  std::unordered_map<int, int> volume_to_boundary_;
  TriangleSurfaceMesh<double> support_boundary_mesh_{
      {{0, 1, 2}},
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
       Eigen::Vector3d::UnitY()}};

  // For signed-distance query to the original mesh during optimization.
  const MeshDistanceBoundary original_boundary_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
