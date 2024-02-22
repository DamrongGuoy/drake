#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/bvh.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/polygon_surface_mesh.h"
#include "drake/geometry/proximity/polygon_surface_mesh_field.h"
#include "drake/geometry/proximity/posed_half_space.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh_field.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

/* This method computes the volume of intersection between overlapping
 geometries. The contact volume is represented by the boundary surface that
 encloses the volume.

 For a given overlapping volume, hereinafter we denote its domain with Ω and its
 boundary with ∂Ω. For each of these volumes, the boundary can be split into two
 sub-domains ∂Ωₘ and ∂Ωₙ, each associated with one of the original geometries M
 and N respectively. ∂Ωₘ is included in the boundary of geometry M and ∂Ωₙ is
 included in the boundary of geometry N. ∂Ωₘ and ∂Ωₙ are disjoint (∂Ωₘ ∩ ∂Ωₙ =
 ∅), and their union is the full boundary ∂Ω of the overlapping volume.

 The ContactSurface refers to a linear field that at each point x ∈ ∂Ω provides
 the signed distance from x to N, when x ∈ ∂Ωₘ, or from x to M, when x ∈ ∂Ωₙ.

 Return the boundary surface of overlapping geometries, represented as a
 std::pair of ContactSurface(s).  The pair stores ∂Ωₘ and ∂Ωₙ separately.
 Each of the ContactSurface has polygonal representation.

 For the returned pair of contact surfaces {∂Ωₘ, ∂Ωₙ}, the one with smaller
 GeometryId comes first, i.e., for id_M < id_N, the pair is {∂Ωₘ, ∂Ωₙ},
 not {∂Ωₙ, ∂Ωₘ}.

 Return a pair of nullptr if the two geometries do not overlap.

 Warning: Be careful with the convention explained in ContactSurface
 documentation regarding the order of the two GeometryId's and the normal
 direction of the contact surfaces.
 https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_contact_surface.html#:~:text=.-,Note,-If
 It said:
   The face normals in mesh_W point out of geometry N and into M.
   If id_M > id_N, the labels will be swapped and the normals of the mesh
   reversed (to maintain the documented invariants). Comparing the input
   parameters with the members of the resulting ContactSurface will reveal if
   such a swap has occurred.

 @note Assume the meshes involved are *double* valued -- in other words, they
 are constant parameters in the calculation. If derivatives are to be found,
 the point of injection is through the definition of the relative position
 of the two meshes. */
template <typename T>
std::pair<std::unique_ptr<ContactSurface<T>>,
          std::unique_ptr<ContactSurface<T>>>
ComputeContactVolume(const GeometryId id_M,
                     const MeshDistanceBoundary& boundary_M,
                     const math::RigidTransform<T>& X_WM, const GeometryId id_N,
                     const MeshDistanceBoundary& boundary_N,
                     const math::RigidTransform<T>& X_WN,
                     // We need these two additional parameters because we
                     // don't have code to mutually clip two boundary surface
                     // meshes. Their pressure fields are ignored because we
                     // resampling the signed distances from boundary_M and
                     // boundary_N on the contact surfaces.
                     const hydroelastic::SoftGeometry& volume_M,
                     const hydroelastic::SoftGeometry& volume_N);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
