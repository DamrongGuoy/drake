#include "drake/geometry/proximity/feature_normal_set.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Vector3d;
using math::RigidTransformd;
using math::RollPitchYawd;

class FeatureNormalSetTest : public ::testing::Test {
 public:
  FeatureNormalSetTest()
      :  // Surface mesh of a tetrahedron, expressed in the mesh's frame M.
         //
         //              Mz
         //              ┆
         //           v3 ●
         //              ┆
         //              ┆      v2
         //           v0 ●┄┄┄┄┄┄┄●┄┄┄ My
         //             ╱
         //            ╱
         //        v1 ●
         //         ╱
         //        Mx
         //
        mesh_M_(
            // The triangles have outward face winding.
            {SurfaceTriangle{0, 2, 1}, SurfaceTriangle{0, 1, 3},
             SurfaceTriangle{0, 3, 2}, SurfaceTriangle{1, 2, 3}},
            {Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
             Vector3d::UnitZ()}) {}

 protected:
  const TriangleSurfaceMesh<double> mesh_M_;
  const double kEps{std::numeric_limits<double>::epsilon()};
};

TEST_F(FeatureNormalSetTest, EdgeNormalIsAverageFaceNormal) {
  const auto dut =
      std::get<FeatureNormalSet>(FeatureNormalSet::MaybeCreate(mesh_M_));
  // Edge v1v2 is shared by face 0 (v0,v2,v1) and face 3 (v1, v2,v3).
  const Vector3d kExpectEdgeNormal =
      (mesh_M_.face_normal(0) + mesh_M_.face_normal(3)).normalized();
  EXPECT_TRUE(
      CompareMatrices(dut.edge_normal({1, 2}), kExpectEdgeNormal, kEps));
}

TEST_F(FeatureNormalSetTest, VertexNormalIsAngleWeightedAverage) {
  const auto dut =
      std::get<FeatureNormalSet>(FeatureNormalSet::MaybeCreate(mesh_M_));
  // Vertex v1 is shared by face 0 (v0,v2,v1), face 1 (v0,v1,v3), and
  // face 3 (v1,v2,v3).  We expect vertex_normal() of v1 to be their
  // angle-weighted average normal.
  //
  //   face      triangle        angle at v1 in the triangle
  //    0      {v0, v2, v1}                π/4
  //    1      {v0, v1, v3}                π/4
  //    3      {v1, v2, v3}                π/3
  const Vector3d kExpectVertexNormal =
      (M_PI_4 * mesh_M_.face_normal(0) + M_PI_4 * mesh_M_.face_normal(1) +
       (M_PI / 3) * mesh_M_.face_normal(3))
          .normalized();
  EXPECT_TRUE(CompareMatrices(dut.vertex_normal(1), kExpectVertexNormal, kEps));
}

// Test the error message when the mesh has two triangles that make a "sharp
// knife" with a very small dihedral angle.
GTEST_TEST(FeatureNormalSet, ErrorSharpKnife) {
  // This is a "flatten" tetrahedron with all four vertices on the same plane.
  // Its surface mesh has zero dihedral angle.
  //
  //              Mz
  //              ┆
  //              ┆       v2
  //           v0 ●┄┄┄┄┄┄┄●┄┄┄ My
  //             ╱       ╱
  //            ╱       ╱
  //        v1 ●┄┄┄┄┄┄┄● v3
  //         ╱
  //        Mx
  //
  const VolumeMesh<double> one_flat_tetrahedron_M(
      {VolumeElement(0, 1, 2, 3)}, {Vector3d::Zero(), Vector3d::UnitX(),
                                    Vector3d::UnitY(), Vector3d(1, 1, 0)});
  const TriangleSurfaceMesh<double> mesh_M =
      ConvertVolumeToSurfaceMesh(one_flat_tetrahedron_M);

  auto dut = FeatureNormalSet::MaybeCreate(mesh_M);
  ASSERT_TRUE(std::holds_alternative<std::string>(dut));
  EXPECT_EQ(std::get<std::string>(dut),
            "FeatureNormalSet: Cannot compute an edge normal because "
            "the two triangles sharing the edge make a very sharp edge.");
}

// Test the error message when the mesh has a "pointy, needle-like" vertex.
GTEST_TEST(FeatureNormalSet, ErrorPointyNeedleVertex) {
  // This scaling factor will creat a "needle" with aspect ratio 1:100.
  const double kSmallBase = 1e-2;
  // The apex vertex v3 becomes very pointy as the base triangle v0v1v2
  // shrinks.
  //
  //              Mz
  //              ┆
  //           v3 ●
  //              ┆
  //              ┆
  //              ┆
  //              ┆
  //              ┆   v2
  //           v0 ●┄┄┄●┄┄┄ My
  //             ╱
  //         v1 ●
  //          ╱
  //        Mx
  //
  const VolumeMesh<double> needle_tetrahedron_M(
      {VolumeElement(0, 1, 2, 3)},
      {Vector3d::Zero(), kSmallBase * Vector3d::UnitX(),
       kSmallBase * Vector3d::UnitY(), Vector3d::UnitZ()});
  const TriangleSurfaceMesh<double> mesh_M =
      ConvertVolumeToSurfaceMesh(needle_tetrahedron_M);

  auto dut = FeatureNormalSet::MaybeCreate(mesh_M);
  ASSERT_TRUE(std::holds_alternative<std::string>(dut));
  EXPECT_EQ(std::get<std::string>(dut),
            "FeatureNormalSet: Cannot compute a vertex normal because "
            "the triangles sharing the vertex form a very pointy needle.");
}

}  // namespace
}  // namespace geometry
}  // namespace drake
