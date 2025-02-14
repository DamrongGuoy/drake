#include "drake/geometry/proximity/surface_to_volume_mesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

namespace fs = std::filesystem;

using Eigen::Vector3d;

// 2025-02-13: 7 Ok, 1 ThrowNullptr, 1 InfiniteLoop, 1 DegenratePointOnBall.
// 1 Excluded due to self-intersection (mustard_bottle).
// 1 Excluded due to multi-object .obj file (two_cube_objects).

GTEST_TEST(ConvertSurfaceToVolumeMeshTest, OneTetrahedron) {
  // A four-triangle mesh of a standard tetrahedron.
  const TriangleSurfaceMesh<double> drake_surface_mesh{
      {// The triangle windings give outward normals.
       SurfaceTriangle{0, 2, 1}, SurfaceTriangle{0, 1, 3},
       SurfaceTriangle{0, 3, 2}, SurfaceTriangle{1, 2, 3}},
      {Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
       Vector3d::UnitZ()}};

  VolumeMesh<double> volume_mesh =
      ConvertSurfaceToVolumeMesh(drake_surface_mesh);

  // Expect a one-tetrahedron mesh with four vertices.
  EXPECT_EQ(volume_mesh.num_vertices(), 4);
  EXPECT_EQ(volume_mesh.num_elements(), 1);
}

GTEST_TEST(bad_geometry_volume_zero, OK) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/bad_geometry_volume_zero.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 6);
  ASSERT_EQ(surface.num_triangles(), 8);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 7);
  EXPECT_EQ(volume.tetrahedra().size(), 8);
}

GTEST_TEST(convex, OK) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/convex.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 20);
  ASSERT_EQ(surface.num_triangles(), 36);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 20);
  EXPECT_EQ(volume.tetrahedra().size(), 38);
}

// Did it get into troubles because the input has 8 components instead of one
// connected piece? Each component is just a symmetric version of a
// triangular prism. Removing all but one prism ran successfully.
GTEST_TEST(cube_corners, ThrowNullptr) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/cube_corners.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);

  EXPECT_EQ(surface.num_vertices(), 48);
  EXPECT_EQ(surface.num_triangles(), 64);

  DRAKE_EXPECT_THROWS_MESSAGE(
      ConvertSurfaceToVolumeMesh(surface),
      "DelaunayMesher::getOneBallBySegment::nullptr ball");

  // This is for one triangular prism.
  // EXPECT_EQ(surface.num_vertices(), 48);
  // EXPECT_EQ(surface.num_triangles(), 8);
  // VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
  // EXPECT_EQ(volume.vertices().size(), 6);
  // EXPECT_EQ(volume.tetrahedra().size(), 3);
}

// Did it get into infinite loop because it's not a topological ball? It's a
// topological torus (donut).
// GTEST_TEST(cube_with_hole, InfiniteLoop) {
//   const fs::path filename =
//       FindResourceOrThrow("drake/geometry/test/cube_with_hole.obj");
//   const TriangleSurfaceMesh<double> surface =
//       ReadObjToTriangleSurfaceMesh(filename);
//   ASSERT_EQ(surface.num_vertices(), 16);
//   ASSERT_EQ(surface.num_triangles(), 32);
//   VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
// }

GTEST_TEST(non_convex_mesh, Ok) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/non_convex_mesh.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 5);
  ASSERT_EQ(surface.num_triangles(), 6);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 5);
  EXPECT_EQ(volume.tetrahedra().size(), 3);
}

GTEST_TEST(octahedron, Ok) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/octahedron.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 6);
  ASSERT_EQ(surface.num_triangles(), 8);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 6);
  EXPECT_EQ(volume.tetrahedra().size(), 4);
}

GTEST_TEST(quad_cube, Ok) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/quad_cube.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 8);
  ASSERT_EQ(surface.num_triangles(), 12);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 9);
  EXPECT_EQ(volume.tetrahedra().size(), 11);
}

// Exclude from the benchmark. The file contains two objects.
// Removing one object manually ran successfully.
// GTEST_TEST(two_cube_objects, InfiniteLoop) {
//   const fs::path filename =
//       FindResourceOrThrow("drake/geometry/test/two_cube_objects.obj");
//   const TriangleSurfaceMesh<double> surface =
//       ReadObjToTriangleSurfaceMesh(filename);
//
//   // If we keep only one of the two cubes at a time, it passes.
//   EXPECT_EQ(surface.num_vertices(), 8);
//   EXPECT_EQ(surface.num_triangles(), 12);
//
//   VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
//
//   EXPECT_EQ(volume.vertices().size(), 12);
//   EXPECT_EQ(volume.tetrahedra().size(), 25);
//
//   WriteVolumeMeshToVtk("unit_test.vtk", volume,
//                        "test");
// }

// Degenerated in-sphere test: DelaunayBall::contains() reached an ambiguity
// case.  It may need coin flips to decide.
GTEST_TEST(evo_bowl_col, DegenratePointOnBall) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/dishes/assets/evo_bowl_col.obj");
  ASSERT_EQ(rlocation.error, "");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
  EXPECT_EQ(surface.num_vertices(), 3957);
  EXPECT_EQ(surface.num_triangles(), 7910);

  DRAKE_EXPECT_THROWS_MESSAGE(
      ConvertSurfaceToVolumeMesh(surface),
      "vegafem::DelaunayMesher::DelaunayBall::contains: undecidable case");
}

GTEST_TEST(plate_8in_col, Ok) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/dishes/assets/plate_8in_col.obj");
  ASSERT_EQ(rlocation.error, "");

  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
  EXPECT_EQ(surface.num_vertices(), 450);
  EXPECT_EQ(surface.num_triangles(), 896);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 450);
  EXPECT_EQ(volume.tetrahedra().size(), 1263);
}

GTEST_TEST(sugar_box, Ok) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/ycb/meshes/004_sugar_box_textured.obj");
  ASSERT_EQ(rlocation.error, "");

  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
  ASSERT_EQ(surface.num_vertices(), 8194);
  ASSERT_EQ(surface.num_triangles(), 16384);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 8194);
  EXPECT_EQ(volume.tetrahedra().size(), 27189);
}

// Exclude from the benchmark.
//
// Meshlab identified 3 self-intersecting faces out of 16,384 triangles in
// the mustard_bottle.
// GTEST_TEST(mustard_bottle, CoreDumped) {
//   const RlocationOrError rlocation =
//       FindRunfile("drake_models/ycb/meshes/006_mustard_bottle_textured.obj");
//   ASSERT_EQ(rlocation.error, "");
//
//   const TriangleSurfaceMesh<double> surface =
//       ReadObjToTriangleSurfaceMesh(rlocation.abspath);
//   EXPECT_EQ(surface.num_vertices(), 8194);
//   EXPECT_EQ(surface.num_triangles(), 16384);
//
//   VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
// }

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
