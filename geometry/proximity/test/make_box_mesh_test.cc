#include "drake/geometry/proximity/make_box_mesh.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

GTEST_TEST(MakeBoxVolumeMeshTest, CalcSequentialIndex) {
  const Vector3<int> num_vertices(3, 2, 5);
  EXPECT_EQ(28, CalcSequentialIndex(2, 1, 3, num_vertices));
}

GTEST_TEST(MakeBoxVolumeMeshTest, GenerateVertices) {
  // Set up a box [-1,1]x[-2,2]x[-3,3] whose corners have integer coordinates.
  const Box box(2.0, 4.0, 6.0);
  // Request the number of vertices so that the vertices have integer
  // coordinates {-1,0,1} x {-2,-1,0,1,2} x {-3,-2,-1,0,1,2,3}.
  // Vertex[i][j][k] should have coordinates (i-1, j-2, k-3) for
  // 0 ≤ i < 3, 0 ≤ j < 5, 0 ≤ k < 7.
  const Vector3<int> num_vertices{3, 5, 7};

  auto vertices = GenerateVertices<double>(box, num_vertices);

  EXPECT_EQ(105, vertices.size());
  for (int i = 0; i < num_vertices.x(); ++i) {
    for (int j = 0; j < num_vertices.y(); ++j) {
      for (int k = 0; k < num_vertices.z(); ++k) {
        int sequential_index = CalcSequentialIndex(i, j, k, num_vertices);
        Vector3<double> expect_r_MV = Vector3<double>(i - 1, j - 2, k - 3);
        Vector3<double> r_MV = vertices[sequential_index].r_MV();
        EXPECT_TRUE(CompareMatrices(expect_r_MV, r_MV))
                    << "Incorrect vertex position.";
      }
    }
  }
}

GTEST_TEST(MakeBoxVolumeMeshTest, AddSixTetrahedraOfCell) {
  const Vector3<int> lowest(1, 2, 3);
  const Vector3<int> num_vertices(3, 4, 5);
  std::vector<VolumeElement> elements;

  AddSixTetrahedraOfCell(lowest, num_vertices, &elements);
  ASSERT_EQ(6, elements.size());

  // In a 3x4x5 grid of vertices, the vertex with (i,j,k)-index = (1,2,3) has
  // its sequential index 33. This picture shows how the rectangular cell
  // with its lowest vertex v₃₃ looks like.
  //
  //               v₃₄     v₃₉
  //               ●------●
  //              /|     /|
  //             / | v₅₉/ |
  //        v₅₄ ●------●  |
  //            |  |   |  |
  //            |  ●---|--● v₃₈
  //            | /v₃₃ | /
  //            |/     |/
  //    +K  v₅₃ ●------● v₅₈
  //     |
  //     |
  //     o------+J
  //    /
  //   /
  // +I
  //
  // This table has the expected six tetrahedra of the rectangular cell.
  // They share the main diagonal v₃₃v₅₉.
  const int expect_elements[6][4] {
      // clang-format off
      {33, 59, 53, 58},
      {33, 59, 58, 38},
      {33, 59, 38, 39},
      {33, 59, 39, 34},
      {33, 59, 34, 54},
      {33, 59, 54, 53}};
  // clang-format on
  for (int e = 0; e < 6; ++e)
    for (int v = 0; v < 4; ++v)
      EXPECT_EQ(expect_elements[e][v], elements[e].vertex(v));
}

GTEST_TEST(MakeBoxVolumeMeshTest, GenerateElements) {
  const Vector3<int> num_vertices{3, 5, 7};
  const int expect_total_num_vertex =
      num_vertices.x() * num_vertices.y() * num_vertices.z();

  const Vector3<int> num_cell = num_vertices - Vector3<int>::Ones();
  const int expect_num_cell = num_cell.x() * num_cell.y() * num_cell.z();
  const int expect_num_element = 6 * expect_num_cell;

  auto elements = GenerateElements(num_vertices);

  EXPECT_EQ(expect_num_element, elements.size());
  // TODO(DamrongGuoy): Find a better way to test `elements`. Currently we
  //  only test that each tetrahedron uses vertices with indices in the range
  //  [0, expect_total_num_vertex). Perhaps check Euler characteristic,
  //  i.e., #vertex - #edge + #triangle - #tetrahedron = 1.
  for (const auto& tetrahedron : elements) {
    for (int v = 0; v < 4; ++v) {
      EXPECT_GE(tetrahedron.vertex(v), 0);
      EXPECT_LT(tetrahedron.vertex(v), expect_total_num_vertex);
    }
  }
}

GTEST_TEST(MakeBoxVolumeMeshTest, CalcOffsetBox) {
  const Box box(0.2, 0.4, 0.8);
  // Estimate three arithmetic operations, so 3 * epsilon.
  const double kTolerance = 3.0 * std::numeric_limits<double>::epsilon();

  // Typical case. The offset is smaller than half the minimum dimension.
  // The offset box still has positive volume.
  {
    const double offset = 0.05;
    const Box offset_box = CalcOffsetBox(box, offset);
    const Vector3<double> expect_size(0.1, 0.3, 0.7);
    EXPECT_TRUE(CompareMatrices(expect_size, offset_box.size(), kTolerance))
        << "Incorrect size of offset box. Typical case.";
  }
  // Special case 1.  The offset equals half the minimum dimension. The
  // offset box degenerates to the medial surface (zero volume, positive area).
  {
    const double offset = 0.1;
    const Box medial_surface = CalcOffsetBox(box, offset);
    const Vector3<double> expect_size(0.0, 0.2, 0.6);
    EXPECT_TRUE(CompareMatrices(expect_size, medial_surface.size(), kTolerance))
        << "Incorrect size of offset box. Special case 1: the offset "
           "equals half the minimum dimension. Expect the offset box to"
           " degenerate to the medial surface.";
  }
  // Special case 2. The offset is larger than half the minimum dimension. The
  // offset box degenerates to the same as the special case 1, i.e., the
  // medial surface. No inverted box (negative dimensions).
  {
    const double offset = 0.15;
    const Box medial_surface = CalcOffsetBox(box, offset);
    const Vector3<double> expect_size(0.0, 0.2, 0.6);
    EXPECT_TRUE(CompareMatrices(expect_size, medial_surface.size(), kTolerance))
        << "Incorrect size of offset box. Special case 2: the offset is"
           " larger than half the minimum dimension. Expect the "
           "offset box to degenerate to the medial surface.";
  }
  // Special case 3: a box with the same width as depth. Its offset
  // degenerates to the medial line segment.
  {
    const Box box_same_width_depth(0.2, 0.2, 0.4);
    const double offset = 0.1;
    const Box medial_line = CalcOffsetBox(box_same_width_depth, offset);
    const Vector3<double> expect_size(0.0, 0.0, 0.2);
    EXPECT_TRUE(CompareMatrices(expect_size, medial_line.size(), kTolerance))
        << "Incorrect size of offset box. Special case 3: A box with the same "
           "width as depth. Its offset degenerates to the medial line segment.";
  }
  // Special case 4: a box with all same dimensions. Its offset degenerates
  // to the center point.
  {
    const Box box_same_width_depth_height(0.2, 0.2, 0.2);
    const double offset = 0.1;
    const Box center_point = CalcOffsetBox(box_same_width_depth_height, offset);
    const Vector3<double> expect_size(0.0, 0.0, 0.0);
    EXPECT_TRUE(CompareMatrices(expect_size, center_point.size(), kTolerance))
    << "Incorrect size of offset box. Special case 4: a box with all same "
       "dimensions. Its offset degenerates to the center point.";
  }
}

GTEST_TEST(MakeBoxVolumeMeshTest, GenerateMesh) {
  const Box box(0.2, 0.4, 0.8);
  VolumeMesh<double> box_mesh = MakeBoxVolumeMesh<double>(box, 0.1, 0.05);

  const int rectangular_cells = 2 * 4 * 8;
  const int tetrahedra_per_cell = 6;
  const int expect_num_tetrahedra = rectangular_cells * tetrahedra_per_cell;
  EXPECT_EQ(expect_num_tetrahedra, box_mesh.num_elements());

  const int expect_num_vertices = 3 * 5 * 9;
  EXPECT_EQ(expect_num_vertices, box_mesh.num_vertices());

  const double expect_volume = box.width() * box.depth() * box.height();
  double volume = 0.0;
  for (int e = 0; e < box_mesh.num_elements(); ++e) {
    double tetrahedron_volume =
        box_mesh.CalcTetrahedronVolume(VolumeElementIndex(e));
    EXPECT_GT(tetrahedron_volume, 0.0);
    volume += tetrahedron_volume;
  }
  EXPECT_NEAR(expect_volume, volume,
              2.0 * std::numeric_limits<double>::epsilon());
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
