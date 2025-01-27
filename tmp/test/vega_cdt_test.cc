#include "../vega_cdt.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Vector3d;

GTEST_TEST(VegaCdtTest, Tetrahedron) {
  // A four-triangle mesh of a standard tetrahedron.
  const TriangleSurfaceMesh<double> drake_surface_mesh{
      {// The triangle windings give outward normals.
       SurfaceTriangle{0, 2, 1}, SurfaceTriangle{0, 1, 3},
       SurfaceTriangle{0, 3, 2}, SurfaceTriangle{1, 2, 3}},
      {Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
       Vector3d::UnitZ()}};

  VolumeMesh<double> drake_tetrahedral_mesh = VegaCdt(drake_surface_mesh);

  // Expect a one-tetrahedron mesh with four vertices.
  EXPECT_EQ(drake_tetrahedral_mesh.num_vertices(), 4);
  EXPECT_EQ(drake_tetrahedral_mesh.num_elements(), 1);
}

}  // namespace
}  // namespace geometry
}  // namespace drake