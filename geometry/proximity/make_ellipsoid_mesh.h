#pragma once

#include <cmath>
#include <utility>

#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

/**
 Approximates the perimeter of an ellipse with lengths of semi-axes `a` and `b`.
 */
double ApproximateEllipsePerimeter(double a, double b) {
  // Kepler's approximation using geometric mean.
  return 2.0 * M_PI * std::sqrt(a * b);
}

/** Creates a volume mesh for the given `ellipsoid`; the level of
 tessellation is guided by the `resolution_hint` parameter.

 `resolution_hint` influences the resolution of the mesh. Smaller values
 create higher-resolution meshes with smaller tetrahedra.

 The resolution of the final mesh will change discontinuousely. Small changes
 to `resolution_hint` will likely produce the same mesh. However, in the
 current implementation, cutting `resolution_hint` in half _will_ increase
 the number of tetrahedra.

 Ultimately, successively smaller values of `resolution_hint` will no longer
 change the output mesh. This algorithm will not produce a tetrahedral mesh with
 more than approximately 100 million tetrahedra.

 @param ellipsoid
 @param resolution_hint
*/
template <typename T>
VolumeMesh<T> MakeEllipsoidVolumeMesh(const Ellipsoid& ellipsoid,
                                      double resolution_hint) {
  const double ellipse_perimeter =
      ApproximateEllipsePerimeter(ellipsoid.get_a(), ellipsoid.get_b());
  const double unit_circle_perimeter = 2.0 * M_PI;
  const double unit_sphere_resolution =
      resolution_hint * unit_circle_perimeter / ellipse_perimeter;
  const Sphere unit_sphere(1.0);
  auto unit_sphere_mesh = MakeSphereVolumeMesh<T>(unit_sphere,
  unit_sphere_resolution);

  const double a = ellipsoid.get_a();
  const double b = ellipsoid.get_b();
  const double c = ellipsoid.get_c();
  const Vector3<T> scale{a, b, c};
  std::vector<VolumeVertex<T>> vertices;
  vertices.reserve(unit_sphere_mesh.num_vertices());
  for (const auto& sphere_vertex : unit_sphere_mesh.vertices()) {
    vertices.emplace_back(scale.cwiseProduct(sphere_vertex.r_MV()));
  }

  std::vector<VolumeElement> tetrahedra = unit_sphere_mesh.tetrahedra();

  return VolumeMesh<T>(std::move(tetrahedra), std::move(vertices));
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
