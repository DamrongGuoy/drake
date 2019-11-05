#include "drake/geometry/proximity/make_ellipsoid_mesh.h"

#include <gtest/gtest.h>

#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

GTEST_TEST(MakeEllipsoidMeshTest, MakeEllipsoidVolumeMesh) {
  const Ellipsoid ellipsoid(4.0, 5.0, 1.0);
  const double resolution_hint = 1.0;
  auto mesh = MakeEllipsoidVolumeMesh<double>(ellipsoid, resolution_hint);
  EXPECT_GE(mesh.num_vertices(), 6);
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
