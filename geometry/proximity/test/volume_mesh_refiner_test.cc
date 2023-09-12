#include "drake/geometry/proximity/volume_mesh_refiner.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/proximity/detect_zero_simplex.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;

GTEST_TEST(VolumeMeshRefinerTest, TestRefinePunyoChest) {
  const std::string test_file = FindResourceOrThrow(
      "drake/geometry/test/"
      "MA-081-PT-0003_chest_centerslice_plus5mm_fine_fTetWild.vtk");
  const VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);
  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 130);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(),
            224);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 94);
  EXPECT_EQ(test_mesh.num_vertices(), 1579);
  EXPECT_EQ(test_mesh.num_elements(), 6741);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 1673);
  EXPECT_EQ(refined_mesh.num_elements(), 7188);

  WriteVolumeMeshToVtk("MA-081-PT-0003_chest_centerslice_plus5mm_fine.vtk",
                       refined_mesh,
                       "Refined tetrahedral mesh for collision geometry");
}

/* This test takes only a second.
GTEST_TEST(VolumeMeshRefinerTest, TestRefinePunyoPaw) {
  const std::string test_file = FindResourceOrThrow(
      "drake/geometry/test/"
      "MA-027-AY-0006_bubble35mm_coarse_250faces_fTetWild.vtk");
  const VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);
  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 67);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(),
            117);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 50);
  EXPECT_EQ(test_mesh.num_vertices(), 935);
  EXPECT_EQ(test_mesh.num_elements(), 3893);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 985);
  EXPECT_EQ(refined_mesh.num_elements(), 4130);

  WriteVolumeMeshToVtk("MA-027-AY-0006_bubble35mm_coarse_250faces.vtk",
                       refined_mesh,
                       "Refined tetrahedral mesh for collision geometry");
}
*/

/* This test takes about one hour to run. Hide it in the comment for now.
GTEST_TEST(VolumeMeshRefinerTest, TestRefinePunyoPaw) {
  const std::string test_file = FindResourceOrThrow(
      "drake/geometry/test/MA-027-AY-0006_bubble35mm_fTetWild.vtk");
  const VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);
  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 99881);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(),
            166654);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 66733);
  EXPECT_EQ(test_mesh.num_vertices(), 44530);
  EXPECT_EQ(test_mesh.num_elements(), 147345);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 111355);
  EXPECT_EQ(refined_mesh.num_elements(), 514322);

  WriteVolumeMeshToVtk("MA-027-AY-0006_bubble35mm.vtk", refined_mesh,
                       "Refined tetrahedral mesh for collision geometry");
}
*/

/* This test takes about a minute to run. Hide it in the comment for now.
GTEST_TEST(VolumeMeshRefinerTest, TestRefinePunyoPaw) {
  const std::string test_file = FindResourceOrThrow(
      "drake/geometry/test/MA-027-AY-0006_pawframe_fTetWild.vtk");
  const VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);
  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 3379);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(),
            6412);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 3086);
  EXPECT_EQ(test_mesh.num_vertices(), 16011);
  EXPECT_EQ(test_mesh.num_elements(), 59123);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 19097);
  EXPECT_EQ(refined_mesh.num_elements(), 74860);

  WriteVolumeMeshToVtk("MA-027-AY-0006_pawframe.vtk", refined_mesh,
                       "Refined tetrahedral mesh for collision geometry");
}
*/

GTEST_TEST(VolumeMeshRefinerTest, TestRefineTetrahedron) {
  //      +Z
  //       |
  //       v3
  //       |
  //       |
  //     v0+------v2---+Y
  //      /
  //     /
  //   v1
  //   /
  // +X
  //
  const VolumeMesh<double> test_mesh(
      std::vector<VolumeElement>{{0, 1, 2, 3}},
      std::vector<Vector3d>{Vector3d::Zero(), Vector3d::UnitX(),
                            Vector3d::UnitY(), Vector3d::UnitZ()});
  ASSERT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 1);
  ASSERT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(), 0);
  ASSERT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 0);
  ASSERT_EQ(test_mesh.num_vertices(), 4);
  ASSERT_EQ(test_mesh.num_elements(), 1);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 5);
  EXPECT_EQ(refined_mesh.num_elements(), 4);
}

GTEST_TEST(VolumeMeshRefinerTest, TestRefineTriangle) {
  // The interior triangle v0v1v2 is shared by two tetrahedra comprising the
  // mesh.
  //
  //      +Z
  //       |
  //       v3
  //       |
  //       |
  //     v0+------v2---+Y
  //      /|
  //     / |
  //   v1  v4
  //   /   |
  // +X    |
  //      -Z
  //
  const VolumeMesh<double> test_mesh(
      std::vector<VolumeElement>{{0, 1, 2, 3}, {2, 1, 0, 4}},
      std::vector<Vector3d>{Vector3d::Zero(), Vector3d::UnitX(),
                            Vector3d::UnitY(), Vector3d::UnitZ(),
                            -Vector3d::UnitZ()});
  ASSERT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 2);
  ASSERT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(), 1);
  ASSERT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 0);
  ASSERT_EQ(test_mesh.num_vertices(), 5);
  ASSERT_EQ(test_mesh.num_elements(), 2);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 6);
  EXPECT_EQ(refined_mesh.num_elements(), 6);
}

GTEST_TEST(VolumeMeshRefinerTest, TestRefineEdge) {
  // The interior edge v0v2 is shared by three tetrahedra comprising the mesh.
  // Vertex v3(-1,1,1) is in the -X+Y+Z octant, and vertex v4(-1,1,-1) is in
  // the -X+Y-Z octant.
  // quadrant.
  //
  //      +Z
  //       |   -X
  //       |   /
  //       |  /   v3(-1,1,1)
  //       | /
  //       |/
  //     v0+-----+------v2---+Y
  //      /|
  //     / |      v4(-1,1,-1)
  //   v1  |
  //   /   |
  // +X    |
  //      -Z
  //
  const VolumeMesh<double> test_mesh(
      std::vector<VolumeElement>{{0, 1, 2, 3}, {2, 1, 0, 4}, {0, 2, 4, 3}},
      std::vector<Vector3d>{Vector3d::Zero(), Vector3d::UnitX(),
                            2 * Vector3d::UnitY(), Vector3d(-1, 1, 1),
                            Vector3d(-1, 1, -1)});
  ASSERT_EQ(DetectTetrahedronWithAllBoundaryVertices(test_mesh).size(), 3);
  ASSERT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(test_mesh).size(), 3);
  ASSERT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(test_mesh).size(), 1);
  ASSERT_EQ(test_mesh.num_vertices(), 5);
  ASSERT_EQ(test_mesh.num_elements(), 3);

  VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.Refine();

  EXPECT_EQ(DetectTetrahedronWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(DetectInteriorTriangleWithAllBoundaryVertices(refined_mesh).size(),
            0);
  EXPECT_EQ(DetectInteriorEdgeWithAllBoundaryVertices(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 6);
  EXPECT_EQ(refined_mesh.num_elements(), 6);
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
