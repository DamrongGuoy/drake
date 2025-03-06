#include "drake/geometry/proximity/surface_to_volume_mesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

namespace fs = std::filesystem;

using Eigen::Vector3d;

// 2025-03-05: all 17 OK, 47 seconds total.
//
// 1. The main solution was to apply random perturbation (10-micron) in
// ConvertSurfaceToVolumeMesh(const TriangleSurfaceMesh<double>&) near
// lines 75-85 of surface_to_volume_mesh.cc.
// Previously all failures happened when vegafem::TetMesher::compute()
// calls TetMesher::initializeCDT(bool recovery = true).
//
// 2. Guard against nullptr in TetMesher::removeOutside() near
// line 535 of tetMesher.cpp.
//
// 1 Excluded due to self-intersection (mustard_bottle).
// 1 Excluded due to multi-object .obj file (two_cube_objects).
//
// Note: There are two initializeCDT():
// - TetMesher::initializeCDT(bool recovery), and
// - DelaunayMesher::initializeCDT(TetMesh * inputMesh, double ep).

GTEST_TEST(ConvertSurfaceToVolumeMeshTest, OK) {
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

// I had a hypothesis that VegaFEM/tetMesher had troubles with
// non-topological-ball cases; however, I used this trivial
// non-topological-ball case, and it passed.  It's simply a set of two
// disjoint tetrahedra.
GTEST_TEST(NonTopologicalBall, OK) {
  // Start with a volume mesh of two disconnected tetrahedra.
  const VolumeMesh<double> two_tetrahedra{
      {// This is a standard tetrahedron at the origin.
       VolumeElement{0, 1, 2, 3},
       // This second tetrahedron is the translation of the first one.
       // The translation is enough to separate them (see vertex coordinates).
       VolumeElement{4, 5, 6, 7}},
      {// Vertex 0-3 belongs to the first tetrahedron.
       Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
       Vector3d::UnitZ(),
       // Vertex 4-7 belongs to the second tetrahedron. They are
       // adequate translation of the first four vertices, so the second
       // tetrahedron doesn't overlap the first tetrahedron.
       Vector3d(0, 0, 1.1), Vector3d(1, 0, 1.1), Vector3d(0, 1, 1.1),
       Vector3d(0, 0, 2.1)}};
  const TriangleSurfaceMesh<double> surface =
      ConvertVolumeToSurfaceMesh(two_tetrahedra);

  EXPECT_EQ(surface.num_vertices(), 8);
  EXPECT_EQ(surface.num_triangles(), 8);

  const VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 8);
  EXPECT_EQ(volume.tetrahedra().size(), 2);
}

// Compare VegaFEM-v4.0.5/tetMesher and our successful customized code.
//     $ tetMesher bad_geometry_volume_zero.obj bad_geometry_volume_zero.veg
//     Refinement quality is: 1.1
//     Alpha is: 1
//     Minimal dihedral is: 0
//     Running the tet mesher...
//     ^C
//     $ bazel-bin/geometry/proximity/mtetm bad_geometry_volume_zero.obj
//                                          bad_geometry_volume_zero.vtk
//     Starting tet mesh refinement.
//     Checking if mesh is triangular... yes
//     Total number of triangles is: 8
//     Building the octree data structure...
//     All tets are good quality after refining.
//
//
//     0 steiner points inserted.
//     [2025-02-18 19:10:01.969] [console] [info] wrote tetrahedral mesh to
//     file 'bad_geometry_volume_zero.vtk' with 8 tets and 7 vertices.
GTEST_TEST(bad_geometry_volume_zero, OK) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/bad_geometry_volume_zero.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 6);
  ASSERT_EQ(surface.num_triangles(), 8);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 6);
  EXPECT_EQ(volume.tetrahedra().size(), 5);
}

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(convex, OK) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/convex.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 20);
  ASSERT_EQ(surface.num_triangles(), 36);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 20);
  EXPECT_EQ(volume.tetrahedra().size(), 40);
}

// Did it get into troubles because the input has 8 components instead of one
// connected piece? Each component is just a symmetric version of a
// triangular prism. Removing all but one prism ran successfully.
//
// VegaFEM-v4.0.5/tetMesher said it's not 2-manifold.
//     $ tetMesher geometry/test/cube_corners.obj cube_corners.veg
//     Refinement quality is: 1.1
//     Alpha is: 1
//     Minimal dihedral is: 0
//     Running the tet mesher...
//     The input mesh must be a 2-manifold mesh
//
// TetGen is ok (after we triangulate the quadrilateral faces).
//
// Stack trace to the throw:
// DelaunayMesher::getOneBallBySegment
// DelaunayMesher::segmentRecoveryUsingFlip
// TetMesher::segmentRecovery
// TetMesher::initializeCDT
// TetMesher::compute
GTEST_TEST(cube_corners, OK_ThrowNullptr) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/cube_corners.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);

  EXPECT_EQ(surface.num_vertices(), 48);
  EXPECT_EQ(surface.num_triangles(), 64);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 48);
  EXPECT_EQ(volume.tetrahedra().size(), 32);
}

// Stack trace to the throw:
// DelaunayMesher::DelaunayBall::contains
// DelaunayMesher::getBallsContainingPoint
// DelaunayMesher::update
// DelaunayMesher::computeDelaunayTetrahedralization
// TetMesher::initializeCDT
// TetMesher::compute
GTEST_TEST(cube_corners_Tet2Tri2Tet, OK_UndecidableCase) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/cube_corners_tet.vtk");
  const VolumeMesh<double> in_volume = ReadVtkToVolumeMesh(filename);
  const TriangleSurfaceMesh<double> surface =
      ConvertVolumeToSurfaceMesh(in_volume);

  WriteSurfaceMeshToVtk("cube_corners_64triangles.vtk", surface,
                        "CubeCorners64Triangles");

  EXPECT_EQ(surface.num_vertices(), 48);
  EXPECT_EQ(surface.num_triangles(), 64);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 48);
  EXPECT_EQ(volume.tetrahedra().size(), 31);
}

// Did it get into infinite loop because it's not a topological ball? It's a
// topological torus (donut).
//
// VegaFEM-v4.0.5/tetMesher said it's not 2-manifold.
//     $ tetMesher geometry/test/cube_with_hole.obj cube_with_hole.veg
//     Refinement quality is: 1.1
//     Alpha is: 1
//     Minimal dihedral is: 0
//     Running the tet mesher...
//     The input mesh must be a 2-manifold mesh
//
// Tetgen is ok (after we split all input faces into triangles).
//
GTEST_TEST(cube_with_hole, OK_InfiniteLoop) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/cube_with_hole.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 16);
  ASSERT_EQ(surface.num_triangles(), 32);

  // Before random perturbation in ConvertSurfaceToVolumeMesh(), we got
  // infinite loops.
  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
  EXPECT_EQ(volume.vertices().size(), 16);
  EXPECT_EQ(volume.tetrahedra().size(), 32);
}
//
// TetMesher::compute()->
// TetMesher::initializeCDT() ->
//  TetMesher::segmentRecovery() ->
//   DelaunayMesher::segmentRecoveryUsingFlip(lineSegment, depth=3) ->
//    DelaunayMesher::segmentRemovalUsingFlip(edge, depth=3->2) ->
//     DelaunayMesher::segmentRemovalUsingFlip(edge, depth=2->1) ->
//      DelaunayMesher::segmentRemovalUsingFlip(edge, depth=1->0) ->
//       DelaunayMesher::getTetsAroundEdge() ->
//        DelaunayMesher::getOneBallBySegment(start = 7, end = 0)
//
// Typical stack trace in the infinite loop:
// DelaunayMesher::getOneBallBySegment
// DelaunayMesher::getTetsAroundEdge
// DelaunayMesher::segmentRemovalUsingFlip
// DelaunayMesher::segmentRemovalUsingFlip
// DelaunayMesher::segmentRemovalUsingFlip
// DelaunayMesher::segmentRecoveryUsingFlip
// TetMesher::segmentRecovery
// TetMesher::initializeCDT
// TetMesher::compute
//
GTEST_TEST(cube_with_hole_Tet2Tri2Tet, OK_InfiniteLoop) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/cube_with_hole_tet.vtk");
  const VolumeMesh<double> in_volume = ReadVtkToVolumeMesh(filename);
  const TriangleSurfaceMesh<double> surface =
      ConvertVolumeToSurfaceMesh(in_volume);

  WriteSurfaceMeshToVtk("cube_with_hole_32triangles.vtk", surface,
                        "CubeWithHole32Triangles");

  EXPECT_EQ(surface.num_vertices(), 16);
  EXPECT_EQ(surface.num_triangles(), 32);

  // Before random perturbation in ConvertSurfaceToVolumeMesh(), we got
  // infinite loops.
  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
  EXPECT_EQ(volume.vertices().size(), 16);
  EXPECT_EQ(volume.tetrahedra().size(), 32);
}

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(non_convex_mesh, OK) {
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

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(octahedron, OK) {
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

// Our customized code is ok, but VegaFEM-v4.0.5/tetMesher said it's not
// 2-manifold.
//     $ tetMesher geometry/test/quad_cube.obj quad_cube.veg
//     Refinement quality is: 1.1
//     Alpha is: 1
//     Minimal dihedral is: 0
//     Running the tet mesher...
//     The input mesh must be a 2-manifold mesh
GTEST_TEST(quad_cube, OK) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/quad_cube.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  ASSERT_EQ(surface.num_vertices(), 8);
  ASSERT_EQ(surface.num_triangles(), 12);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 8);
  EXPECT_EQ(volume.tetrahedra().size(), 7);
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

// The input is non-manifold. Our code got into degenerated in-sphere test:
// ambiguous DelaunayBall::contains().
//
// VegaFEM-v4.0.5/tetMesher said it's not 2-manifold.
//     $ tetMesher
//     ~/GitHub/DamrongGuoy/RobotLocomotion_models/dishes/assets/evo_bowl_col.obj
//     evo_bowl_col.veg Refinement quality is: 1.1 Alpha is: 1 Minimal dihedral
//     is: 0 Running the tet mesher... The input mesh must be a 2-manifold mesh
//
// TetGen is ok (after we fixed manifold-ness).
//
// GTEST_TEST(evo_bowl_col, UndecidableCase) {
//  const RlocationOrError rlocation =
//      FindRunfile("drake_models/dishes/assets/evo_bowl_col.obj");
//  ASSERT_EQ(rlocation.error, "");
//  const TriangleSurfaceMesh<double> surface =
//      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
//  EXPECT_EQ(surface.num_vertices(), 3957);
//  EXPECT_EQ(surface.num_triangles(), 7910);
//
//  DRAKE_EXPECT_THROWS_MESSAGE(
//      ConvertSurfaceToVolumeMesh(surface),
//      "vegafem::DelaunayMesher::DelaunayBall::contains: undecidable case");
//}

// We got "DelaunayBall::contains: undecidable case".  It's like a coin-flip
// problem.
//
// VegaFEM-v4.0.5/tetMesher is ok with this "cleaned" bowl.
//     $ tetMesher geometry/test/evo_bowl_fine_7910triangles.obj
//                 evo_bowl_fine_7910triangles.veg
//     ...
//     Running the tet mesher...
//     Starting tet mesh refinement.
//     Checking if mesh is triangular... yes
//     Total number of triangles is: 7910
//     ...
//     184 steiner points inserted.
//     Saving the output mesh to evo_bowl_fine_7910triangles.veg.
//
// TetGen is ok.
//     $ python3 ~/gh.ssv/damrong-guoy/3d_mesh_generator/venv_tetgen.py
//       --input geometry/test/evo_bowl_fine_7910triangles.obj
//       --output evo_bowl_fine_7910triangles_tetgen.vtk
//     Write a temporary file as TetGen input:
//     evo_bowl_fine_7910triangles_tetgen.ply.
//     Call TetGen.
//     Wrote tetrahedral mesh to evo_bowl_fine_7910triangles_tetgen.vtk
GTEST_TEST(evo_bowl_fine_7910triangles, OK_UndecidableCase_5Seconds) {
  const fs::path filename = FindResourceOrThrow(
      "drake/geometry/test/evo_bowl_fine_7910triangles.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);
  EXPECT_EQ(surface.num_vertices(), 3957);
  EXPECT_EQ(surface.num_triangles(), 7910);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);
  // It throw before.
  // DRAKE_EXPECT_THROWS_MESSAGE(
  //     ConvertSurfaceToVolumeMesh(surface),
  //     "vegafem::DelaunayMesher::DelaunayBall::contains: undecidable case");

  EXPECT_EQ(volume.vertices().size(), 3957);
  EXPECT_EQ(volume.tetrahedra().size(), 13715);
}

GTEST_TEST(evo_bowl_coarse3k_Tet2Tri2Tet, OK1Second) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/dishes/assets/evo_bowl_coarse3k.vtk");
  ASSERT_EQ(rlocation.error, "");
  const VolumeMesh<double> in_volume =
      ReadVtkToVolumeMesh(std::filesystem::path(rlocation.abspath));
  const TriangleSurfaceMesh<double> surface =
      ConvertVolumeToSurfaceMesh(in_volume);

  EXPECT_EQ(surface.num_vertices(), 249);
  EXPECT_EQ(surface.num_triangles(), 494);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 249);
  EXPECT_EQ(volume.tetrahedra().size(), 750);
}

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(plate_8in_col, OK1Second) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/dishes/assets/plate_8in_col.obj");
  ASSERT_EQ(rlocation.error, "");

  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
  EXPECT_EQ(surface.num_vertices(), 450);
  EXPECT_EQ(surface.num_triangles(), 896);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 450);
  EXPECT_EQ(volume.tetrahedra().size(), 1420);
}

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(sugar_box, OK18Seconds) {
  const RlocationOrError rlocation =
      FindRunfile("drake_models/ycb/meshes/004_sugar_box_textured.obj");
  ASSERT_EQ(rlocation.error, "");

  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(rlocation.abspath);
  ASSERT_EQ(surface.num_vertices(), 8194);
  ASSERT_EQ(surface.num_triangles(), 16384);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 8194);
  EXPECT_EQ(volume.tetrahedra().size(), 27229);
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

// Both VegaFEM-v4.0.5/tetMesher and our customized code are ok.
GTEST_TEST(Android_Lego, OK16Seconds) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/Android_Lego.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);

  EXPECT_EQ(surface.num_vertices(), 7109);
  EXPECT_EQ(surface.num_triangles(), 14214);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 7109);
  EXPECT_EQ(volume.tetrahedra().size(), 24407);
}

GTEST_TEST(gso_RoLoPM_adizero_F50_TRX_FG_LEA, OK_PreviousCoreDumped) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/"
                          "gso_RoLoPM_adizero_F50_TRX_FG_LEA.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);

  EXPECT_EQ(surface.num_vertices(), 177);
  EXPECT_EQ(surface.num_triangles(), 350);

  VolumeMesh<double> volume = ConvertSurfaceToVolumeMesh(surface);

  EXPECT_EQ(volume.vertices().size(), 177);
  EXPECT_EQ(volume.tetrahedra().size(), 524);
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
