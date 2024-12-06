#include "drake/geometry/proximity/compute_contact_volume.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/default_scalars.h"
#include "drake/common/extract_double.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/calc_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/calc_signed_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/make_mesh_field.h"
#include "drake/geometry/proximity/mesh_intersection.h"
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

}  // namespace

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

template <typename T>
PolygonSurfaceMesh<T> HackToIntersectSurfaceWithVolume(
    // Provide triangle mesh
    const GeometryId id_R, const MeshDistanceBoundary& boundary_R,
    const math::RigidTransform<T>& X_WR,
    // Provide tetrahedral mesh
    const GeometryId id_S, const hydroelastic::SoftGeometry& volume_S,
    const math::RigidTransform<T>& X_WS) {
  std::unique_ptr<ContactSurface<T>> hydro_contact_surface =
      ComputeContactSurfaceFromSoftVolumeRigidSurface(
          // Compliant volume
          id_S, volume_S.pressure_field(), volume_S.bvh(), X_WS,
          // Rigid surface
          id_R, boundary_R.tri_mesh(), boundary_R.tri_bvh(), X_WR,
          HydroelasticContactRepresentation::kPolygon);

  // Copy the contact mesh because ContactSurface::poly_mesh_W() is read-only.
  PolygonSurfaceMesh<T> contact_mesh_W(hydro_contact_surface->poly_mesh_W());

  // TODO(DamrongGuoy): Reverse face winding of the contact mesh
  //  according to the order of id_R and id_S if needed.

  return contact_mesh_W;
}

template <typename T>
std::unique_ptr<ContactSurface<T>> MakeSignedDistanceContactSurface(
    const GeometryId id_A, const GeometryId id_B,
    const PolygonSurfaceMesh<T>& bdΩₐ_W, const MeshDistanceBoundary& boundary_B,
    const math::RigidTransform<T>& X_WB) {
  if (!std::holds_alternative<FeatureNormalSet>(boundary_B.feature_normal())) {
    throw std::runtime_error(
        "MakeSignedDistanceContactSurface: " +
        std::get<std::string>(boundary_B.feature_normal()));
  }
  const FeatureNormalSet& feature_normals_B =
      std::get<FeatureNormalSet>(boundary_B.feature_normal());
  const math::RigidTransform<T> X_BW = X_WB.inverse();
  int num_vertices = bdΩₐ_W.num_vertices();
  std::vector<T> signed_distance_at_vertices;
  for (int i = 0; i < num_vertices; ++i) {
    Vector3<T> p_WV = bdΩₐ_W.vertex(i);
    Vector3<T> p_BV = X_BW * p_WV;
    Vector3<double> p_BVd = ExtractDoubleOrThrow(p_BV);
    SignedDistanceToSurfaceMesh sd = CalcSignedDistanceToSurfaceMesh(
        p_BVd, boundary_B.tri_mesh(), boundary_B.tri_bvh(), feature_normals_B);
    // The values of signed distances are frame-independent.
    signed_distance_at_vertices.push_back(sd.signed_distance);
  }
  std::vector<T> signed_distance_at_centroids;
  std::vector<Vector3<T>> gradient_at_centroids_W;
  const int num_faces = bdΩₐ_W.num_faces();
  for (int i = 0; i < num_faces; ++i) {
    Vector3<T> p_WC = bdΩₐ_W.element_centroid(i);
    Vector3<T> p_BC = X_BW * p_WC;
    Vector3<double> p_BCd = ExtractDoubleOrThrow(p_BC);
    SignedDistanceToSurfaceMesh sd = CalcSignedDistanceToSurfaceMesh(
        p_BCd, boundary_B.tri_mesh(), boundary_B.tri_bvh(), feature_normals_B);

    signed_distance_at_centroids.push_back(sd.signed_distance);

    Vector3<double> gradient_B_double = sd.gradient;
    Vector3<T> gradient_B = gradient_B_double;
    Vector3<T> gradient_W = X_WB.rotation() * gradient_B;

    gradient_at_centroids_W.push_back(gradient_W);
  }

  // TODO(DamrongGuoy) Avoid copying the contact mesh by std::move the mesh
  //  into ContactSurface and change the input-parameter bdΩₐ_W to be an
  //  in-out parameter.  Document that bdΩₐ_W is reset on return.
  auto mesh_W = std::make_unique<PolygonSurfaceMesh<T>>(bdΩₐ_W);

  auto mesh_field_W = std::make_unique<PolygonSurfaceMeshFieldLinear<T, T>>(
      std::move(signed_distance_at_vertices), mesh_W.get(),
      // TODO(DamrongGuoy) Avoid copying the gradients.  Perhaps switch to
      //  another constructor with gradient_mode = MeshGradientMode::kNone if
      //  there is no use. For now, we provide it for downstream
      //  verification.
      std::vector<Vector3<T>>(gradient_at_centroids_W));

  // We have evaluated signed distances and gradients from B to vertices or
  // face centroids on A (bdΩₐ_W to be precise), so all the quantities are
  // associated with A.
  return std::make_unique<ContactSurface<T>>(
      id_A, id_B, std::move(mesh_W), std::move(mesh_field_W),
      std::make_unique<std::vector<Vector3<T>>>(
          std::move(gradient_at_centroids_W)),
      // grad_eN_W
      std::unique_ptr<std::vector<Vector3<T>>>(nullptr),
      // e_MN_at_face_centroids
      std::make_unique<std::vector<T>>(
          std::move(signed_distance_at_centroids)));
}

template <typename T>
std::pair<std::unique_ptr<ContactSurface<T>>,
          std::unique_ptr<ContactSurface<T>>>
ComputeContactVolumeNew(const GeometryId id_M,
                        const MeshDistanceBoundary& boundary_M,
                        const math::RigidTransform<T>& X_WM,
                        const GeometryId id_N,
                        const MeshDistanceBoundary& boundary_N,
                        const math::RigidTransform<T>& X_WN,
                        const hydroelastic::SoftGeometry& volume_M,
                        const hydroelastic::SoftGeometry& volume_N) {
  // TODO(DamrongGuoy): Check consistency of normal directions in both this
  //  function and in MakeSignedDistanceContactSurface().
  PolygonSurfaceMesh<T> bdΩₘ_W = HackToIntersectSurfaceWithVolume(
      id_M, boundary_M, X_WM, id_N, volume_N, X_WN);
  PolygonSurfaceMesh<T> bdΩₙ_W = HackToIntersectSurfaceWithVolume(
      id_N, boundary_N, X_WN, id_M, volume_M, X_WM);

  std::unique_ptr<ContactSurface<T>> contact_bdΩₘ_W =
      MakeSignedDistanceContactSurface(id_M, id_N, bdΩₘ_W, boundary_N, X_WN);
  std::unique_ptr<ContactSurface<T>> contact_bdΩₙ_W =
      MakeSignedDistanceContactSurface(id_N, id_M, bdΩₙ_W, boundary_M, X_WM);

  return {std::move(contact_bdΩₘ_W), std::move(contact_bdΩₙ_W)};
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeContactVolume<T>, &ComputeContactVolumeNew<T>));

}  // namespace internal
}  // namespace geometry
}  // namespace drake
