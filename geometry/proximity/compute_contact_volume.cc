#include "drake/geometry/proximity/compute_contact_volume.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/default_scalars.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/mesh_intersection.h"
#include "drake/geometry/proximity/make_mesh_field.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

namespace {

VolumeMeshFieldLinear<double, double> MakeVolumeMeshUnsignedDistanceField(
    const VolumeMesh<double>* mesh_M) {
  DRAKE_DEMAND(mesh_M != nullptr);
  std::vector<int> boundary_vertices;
  // The subscript _d is for the scalar type double.
  TriangleSurfaceMesh<double> surface_M =
      ConvertVolumeToSurfaceMeshWithBoundaryVertices(*mesh_M,
                                                     &boundary_vertices);
  std::vector<double> distance_values;
  const Bvh<Obb, TriangleSurfaceMesh<double>> bvh(surface_M);
  auto boundary_iter = boundary_vertices.begin();
  for (int v = 0; v < ssize(mesh_M->vertices()); ++v) {
    if (boundary_iter != boundary_vertices.end() && *boundary_iter == v) {
      ++boundary_iter;
      distance_values.push_back(0);
      continue;
    }
    const Vector3<double>& p_MV = mesh_M->vertex(v);
    double distance = internal::CalcDistanceToSurfaceMesh(p_MV, surface_M, bvh);
    distance_values.push_back(distance);
  }

  return {std::move(distance_values), mesh_M};
}

template <typename T>
std::unique_ptr<ContactSurface<T>>
HackToCallComputeContactSurfaceFromSoftVolumeRigidSurface(
    const GeometryId id_S, const VolumeMesh<double>& mesh_S,
    const Bvh<Obb, VolumeMesh<double>>& bvh_S,
    const math::RigidTransform<T>& X_WS, const GeometryId id_R,
    const VolumeMesh<double>& mesh_R,
    const Bvh<Obb, VolumeMesh<double>>& /*bvh_R*/,
    const math::RigidTransform<T>& X_WR,
    HydroelasticContactRepresentation representation) {
  // This is a hack to reuse ComputeContactSurfaceFromSoftVolumeRigidSurface.
  // 1. It needs VolumeMeshFieldLinear of S instead of VolumeMesh.
  // 2. It needs TriangleSurfaceMesh of R instead of VolumeMesh.

  const VolumeMeshFieldLinear<double, double> field_S =
      MakeVolumeMeshUnsignedDistanceField(&mesh_S);
  const TriangleSurfaceMesh<double> surface_mesh_R =
      ConvertVolumeToSurfaceMesh(mesh_R);
  const Bvh<Obb, TriangleSurfaceMesh<double>> bvh_surface_R(surface_mesh_R);

  // TODO(DamrongGuoy): Change unsigned distance to signed distance by
  //  "multiplying the field by -1".
  return ComputeContactSurfaceFromSoftVolumeRigidSurface(
      id_S, field_S, bvh_S, X_WS, id_R, surface_mesh_R, bvh_surface_R, X_WR,
      representation);
}

}

template <typename T>
std::pair<std::unique_ptr<ContactSurface<T>>,
          std::unique_ptr<ContactSurface<T>>>
ComputeContactVolume(const GeometryId id_M, const VolumeMesh<double>& mesh_M,
                     const Bvh<Obb, VolumeMesh<double>>& bvh_M,
                     const math::RigidTransform<T>& X_WM, const GeometryId id_N,
                     const VolumeMesh<double>& mesh_N,
                     const Bvh<Obb, VolumeMesh<double>>& bvh_N,
                     const math::RigidTransform<T>& X_WN,
                     HydroelasticContactRepresentation representation) {
  return {
      HackToCallComputeContactSurfaceFromSoftVolumeRigidSurface<T>(
          id_M, mesh_M, bvh_M, X_WM, id_N, mesh_N, bvh_N, X_WN, representation),
      HackToCallComputeContactSurfaceFromSoftVolumeRigidSurface<T>(
          id_N, mesh_N, bvh_N, X_WN, id_M, mesh_M, bvh_M, X_WM,
          representation)};
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeContactVolume<T>))

}  // namespace internal
}  // namespace geometry
}  // namespace drake