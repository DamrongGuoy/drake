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
// TODO(DamrongGuoy):  Consider other conventions for the direction of face
//  normals of the returned mesh if it's more efficient. These are some
//  possibilities:
//  - The face normals point out of S and into R.
//  - The face normals point out of S and into R when id_R < id_S, and
//    they point out of R and into S, otherwise.
//  - The face normals point out of R and into S when id_R < id_S, and
//    they point out of S and into R, otherwise.

// Return the intersecting surface mesh between a triangle mesh R and a
// tetrahedral mesh S.
//
// @note The returned mesh has face normals pointing *out of* R and *into* S.
template <typename T>
std::unique_ptr<PolygonSurfaceMesh<T>> HackToIntersectSurfaceWithVolume(
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
          HydroelasticContactRepresentation::kPolygon,
          // filter_face_normal_along_field_gradient
          false);

  if (hydro_contact_surface == nullptr) {
    return nullptr;
  }

  // Copy the contact mesh because ContactSurface::poly_mesh_W() is read-only.
  auto mesh_W = std::make_unique<PolygonSurfaceMesh<T>>(
      hydro_contact_surface->poly_mesh_W());

  if (hydro_contact_surface->id_M() == id_R) {
    // The ContactSurface documentation says the face normals in `mesh_W`
    // point *out of* geometry N and *into* M, so it points out of S and into R.
    // We will swap the direction, so it points out of R and into S.
    mesh_W->ReverseFaceWinding();
  }

  return mesh_W;
}

// @pre  The face normals of bdΩₐ_W point *out of* geometry A and *into*
//       geometry B.
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

  std::vector<T> signed_distance_at_vertices;
  int num_vertices = bdΩₐ_W.num_vertices();
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

  // TODO(DamrongGuoy) Avoid copying the contact mesh. Change the
  //  in-parameter bdΩₐ_W to be an in-out parameter.  Document that
  //  bdΩₐ_W is reset on return.
  auto mesh_W = std::make_unique<PolygonSurfaceMesh<T>>(bdΩₐ_W);

  auto mesh_field_W = std::make_unique<PolygonSurfaceMeshFieldLinear<T, T>>(
      std::move(signed_distance_at_vertices), mesh_W.get(),
      // TODO(DamrongGuoy) Avoid copying the gradients.  Perhaps switch to
      //  another constructor with gradient_mode = MeshGradientMode::kNone if
      //  there is no use of surface gradients. For now, we provide it for
      //  downstream verification.
      std::vector<Vector3<T>>(gradient_at_centroids_W));

  // The constructor ContactSurface(id_M, id_N,...) requires the mesh_W to
  // have face normals point *out of* N ad *into* M.  The precondition of
  // this function demands the face normals of bdΩₐ_W point *out of* A
  // and *into* B, so we have to call ContactSurface(id_B, id_A,...).
  return std::make_unique<ContactSurface<T>>(
      id_B, id_A, std::move(mesh_W), std::move(mesh_field_W),
      // We only evaluated gradients of B on bdΩₐ
      std::make_unique<std::vector<Vector3<T>>>(
          std::move(gradient_at_centroids_W)),
      std::unique_ptr<std::vector<Vector3<T>>>(nullptr),
      std::make_unique<std::vector<T>>(
          std::move(signed_distance_at_centroids)));
}

}  // namespace

template <typename T>
std::pair<std::unique_ptr<ContactSurface<T>>,
          std::unique_ptr<ContactSurface<T>>>
ComputeContactVolume(const GeometryId id_M,
                     const MeshDistanceBoundary& boundary_M,
                     const math::RigidTransform<T>& X_WM, const GeometryId id_N,
                     const MeshDistanceBoundary& boundary_N,
                     const math::RigidTransform<T>& X_WN,
                     const hydroelastic::SoftGeometry& volume_M,
                     const hydroelastic::SoftGeometry& volume_N) {
  // The document of HackToIntersectSurfaceWithVolume(id_R,...,id_S,...) says
  // the returned mesh has face normals pointing out of R and into S.
  // Therefore, the face normals of bdΩₘ_W point out of M and into N.
  std::unique_ptr<PolygonSurfaceMesh<T>> bdΩₘ_W =
      HackToIntersectSurfaceWithVolume(id_M, boundary_M, X_WM, id_N, volume_N,
                                       X_WN);
  // Similarly, the face normals of bdΩₙ_W point out of N and into M.
  std::unique_ptr<PolygonSurfaceMesh<T>> bdΩₙ_W =
      HackToIntersectSurfaceWithVolume(id_N, boundary_N, X_WN, id_M, volume_M,
                                       X_WM);

  if (bdΩₘ_W == nullptr || bdΩₙ_W == nullptr) {
    return {nullptr, nullptr};
  }

  // MakeSignedDistanceContactSurface(id_A, id_B,...) requires the face
  // normals of the surface mesh point *out of* geometry A and *into*
  // geometry B.
  std::unique_ptr<ContactSurface<T>> contact_bdΩₘ_W =
      MakeSignedDistanceContactSurface(id_M, id_N, *bdΩₘ_W, boundary_N, X_WN);
  std::unique_ptr<ContactSurface<T>> contact_bdΩₙ_W =
      MakeSignedDistanceContactSurface(id_N, id_M, *bdΩₙ_W, boundary_M, X_WM);

  // Note that contact_bdΩₘ_W->id_M() is not necessarily id_M (ditto for N).
  // The ContactSurface constructor might have switched the GeometryId's.
  DRAKE_ASSERT(contact_bdΩₘ_W->id_M() < contact_bdΩₘ_W->id_N());
  DRAKE_ASSERT(contact_bdΩₙ_W->id_M() < contact_bdΩₙ_W->id_N());

  if (contact_bdΩₘ_W->HasGradE_N()) {
    // The contact surface s = contact_bdΩₘ_W has the signed-distance gradient
    // from geometry s.Id_N(), so s lies on the geometry s.Id_M(). Therefore,
    // s should come first.
    return {std::move(contact_bdΩₘ_W), std::move(contact_bdΩₙ_W)};
  } else {
    // The contact surface s = contact_bdΩₘ_W does not have the
    // signed-distance gradient from geometry s.Id_N(), so s lies on the
    // geometry s.Id_N(). Therefore, s should come second.
    return {std::move(contact_bdΩₙ_W), std::move(contact_bdΩₘ_W)};
  }
}

DRAKE_DEFINE_FUNCTION_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    (&ComputeContactVolume<T>));

}  // namespace internal
}  // namespace geometry
}  // namespace drake
