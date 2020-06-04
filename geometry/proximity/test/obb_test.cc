#include "drake/geometry/proximity/obb.h"

#include <memory>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/shape_specification.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::AngleAxisd;
using Eigen::Vector3d;
using math::RigidTransformd;
using math::RotationMatrixd;

class ObbTester : public ::testing::Test {
 public:
  static constexpr double kTolerance = Obb::kTolerance;
};

namespace {

// Tests calculating the bounding box volume. Due to boundary padding, the
// volume is increased from 8abc to 8((a + ε)*(b + ε)*(c+ε)), i.e.:
// 8[abc + (ab + bc + ac)ε + (a + b + c)ε² + ε³].
TEST_F(ObbTester, TestVolume) {
  Obb obb = Obb(RigidTransformd(Vector3d(-1, 2, 1)), Vector3d(2, 0.5, 2.7));
  // In this case the dominating error term is 8(ab + bc + ac)ε, which caps
  // out under kTolerance * 70.
  const double volume = obb.CalcVolume();
  EXPECT_NEAR(volume, 21.6, kTolerance * 70);
  EXPECT_GT(volume, 21.6);
  Obb zero_obb = Obb(RigidTransformd(Vector3d(3, -4, 1.3)), Vector3d(0, 0, 0));
  // Since a, b and c are 0, only the ε³ term is left and kTolerance³ is
  // within kTolerance.
  const double zero_volume = zero_obb.CalcVolume();
  EXPECT_NEAR(zero_volume, 0, kTolerance);
  EXPECT_GT(zero_volume, 0);
}

// Tests padding the boundary of the bounding box volume.
TEST_F(ObbTester, TestPadBoundary) {
  Obb obb = Obb(RigidTransformd(Vector3d(-1, 0.5, 1)), Vector3d(1.2, 2.5, 0.3));
  Vector3d padded = Vector3d(1.2, 2.5, 0.3).array() + kTolerance;
  EXPECT_TRUE(CompareMatrices(obb.half_width(), padded));

  // Large boxes should have a bigger padding based on either the maximum
  // half width or position in the frame.
  const double padding = 300 * std::numeric_limits<double>::epsilon();
  ASSERT_GT(padding, kTolerance);
  // Max is set from half_width.z.
  obb = Obb(RigidTransformd(Vector3d(-1, 1.5, 1)), Vector3d(120, 250, 300));
  padded = Vector3d(120, 250, 300).array() + padding;
  // Expect the two Vector3d to be exactly equal down to the last bit.
  EXPECT_TRUE(CompareMatrices(obb.half_width(), padded));
  // Max is set from |center.x|.
  obb = Obb(RigidTransformd(Vector3d(-300, 50, 100)), Vector3d(1, 2, 0.5));
  padded = Vector3d(1, 2, 0.5).array() + padding;
  // Expect the two Vector3d to be exactly equal down to the last bit.
  EXPECT_TRUE(CompareMatrices(obb.half_width(), padded));
}

// We want to compute X_AB such that B is posed relative to A as documented in
// TestObbOverlap. We can do so by generating the rotation component, R_AB, such
// that Bq has a minimum value along the chosen axis, and we can solve for
// the translation component, p_AoBo = p_AoAf + p_AfBq + p_BqBo_A.
auto calc_corner_transform = [](const Obb& a, const Obb& b, const int axis,
                                const bool expect_overlap) -> RigidTransformd {
  const int axis1 = (axis + 1) % 3;
  const int axis2 = (axis + 2) % 3;
  // Construct the rotation matrix, R_AB, that has meaningful (non-zero)
  // values everywhere for the remaining 2 axes and no symmetry.
  RotationMatrixd R_AB =
      RotationMatrixd(AngleAxisd(M_PI / 5, Vector3d::Unit(axis1))) *
      RotationMatrixd(AngleAxisd(-M_PI / 5, Vector3d::Unit(axis2)));
  // We define p_BoBq in Frame A by taking the minimum corner and applying
  // the constructed rotation.
  Vector3d p_BoBq_A = R_AB * (b.pose().translation() - b.half_width());
  // Reality check that the center (p_BoBc_A) and the maximum corner
  // (p_BoBqprime_A) are strictly increasing along the given axis.
  Vector3d p_BoBc_A = R_AB * b.pose().translation();
  Vector3d p_BoBqprime_A = R_AB * (b.pose().translation() + b.half_width());
  DRAKE_DEMAND(p_BoBc_A[axis] > p_BoBq_A[axis]);
  DRAKE_DEMAND(p_BoBqprime_A[axis] > p_BoBc_A[axis]);
  // We construct Bq to be a small relative offset either side of Af along the
  // given axis, depending on whether we expect the boxes to overlap.
  Vector3d p_AfBq{0, 0, 0};
  p_AfBq[axis] = expect_overlap ? -0.01 : 0.01;
  // We construct Af by taking the maximum corner and offsetting it along the
  // remaining 2 axes, e.g. by a quarter across. This ensures we thoroughly
  // exercise all bits instead of simply using any midpoints or corners.
  // z
  // ^
  // |  -------------
  // |  |     |  o  |
  // |  |------------
  // |  |     |     |
  // |  -------------
  // -----------------> y
  Vector3d p_AoAf = a.half_width();
  p_AoAf[axis1] /= 2;
  p_AoAf[axis2] /= 2;
  p_AoAf += a.pose().translation();
  // We can rewrite +p_BqBo as -p_BoBq, thus solving for p_AoBo = p_AoAf +
  // p_AfBq - p_BoBq_A.
  Vector3d p_AoBo = p_AoAf + p_AfBq - p_BoBq_A;
  // Finally we combine the components to form the transform X_AB.
  return RigidTransformd(R_AB, p_AoBo);
};

// We want to compute X_AB such that B is posed relative to A as documented
// in TestObbOverlap. We can do so by generating the rotation component, R_AB,
// such that Bq lies on the minimum edge along the chosen axis, and we can solve
// for the translation component, p_AoBo = p_AoAf + p_AfBq + p_BqBo_A.
auto calc_edge_transform = [](const Obb& a, const Obb& b, const int a_axis,
                              const int b_axis,
                              const bool expect_overlap) -> RigidTransformd {
  const int a_axis1 = (a_axis + 1) % 3;
  const int a_axis2 = (a_axis + 2) % 3;
  const int b_axis1 = (b_axis + 1) % 3;
  const int b_axis2 = (b_axis + 2) % 3;
  // Construct a rotation matrix that has meaningful (non-zero) values
  // everywhere for the remaining 2 axes and no symmetry. Depending on the
  // combination of axes, we need to rotate around different axes to ensure
  // the edge remains as the minimum.
  RotationMatrixd R_AB;
  const double theta = M_PI / 5;
  // For cases Ax × Bx, Ay × By, and Az × Bz.
  if (a_axis == b_axis) {
    R_AB = RotationMatrixd(AngleAxisd(theta, Vector3d::Unit(b_axis1))) *
           RotationMatrixd(AngleAxisd(theta, Vector3d::Unit(b_axis2)));
    // For cases Ax × By, Ay × Bz, and Az × Bx.
  } else if (a_axis1 == b_axis) {
    R_AB = RotationMatrixd(AngleAxisd(theta, Vector3d::Unit(b_axis1))) *
           RotationMatrixd(AngleAxisd(-theta, Vector3d::Unit(b_axis2)));
    // For cases Ax × Bz, Ay × Bx, and Az × By.
  } else {
    R_AB = RotationMatrixd(AngleAxisd(-theta, Vector3d::Unit(b_axis2))) *
           RotationMatrixd(AngleAxisd(theta, Vector3d::Unit(b_axis1)));
  }
  // We define p_BoBq in Frame B taking a point on the minimum edge aligned
  // with the given axis, offset it to be without symmetry, then convert it
  // to Frame A by applying the rotation.
  Vector3d p_BoBQ_B = b.pose().translation() - b.half_width();
  p_BoBQ_B[b_axis] += b.half_width()[b_axis] / 2;
  Vector3d p_BoBq_A = R_AB * p_BoBQ_B;
  // Reality check that the center (p_BoBc_A) and the point on the opposite
  // edge (p_BoBqprime_A) are strictly increasing along the remaining 2 axes.
  Vector3d p_BoBc_A = R_AB * b.pose().translation();
  Vector3d p_BoBqprime_B = b.pose().translation() + b.half_width();
  p_BoBqprime_B[b_axis] -= b.half_width()[b_axis] / 2;
  Vector3d p_BoBqprime_A = R_AB * p_BoBqprime_B;
  DRAKE_DEMAND(p_BoBc_A[a_axis1] > p_BoBq_A[a_axis1]);
  DRAKE_DEMAND(p_BoBqprime_A[a_axis1] > p_BoBc_A[a_axis1]);
  DRAKE_DEMAND(p_BoBc_A[a_axis2] > p_BoBq_A[a_axis2]);
  DRAKE_DEMAND(p_BoBqprime_A[a_axis2] > p_BoBc_A[a_axis2]);
  // We construct Bq to be a small relative offset either side of Af along the
  // given axis, depending on whether we expect the boxes to overlap.
  Vector3d p_AfBq{0, 0, 0};
  const double offset = expect_overlap ? -0.01 : 0.01;
  p_AfBq[a_axis1] = offset;
  p_AfBq[a_axis2] = offset;
  // We construct Af by taking the maximum corner and offsetting it along the
  // given edge to thoroughly exercise all bits.
  Vector3d p_AoAf = a.pose().translation() + a.half_width();
  p_AoAf[a_axis] -= a.half_width()[a_axis] / 2;
  // We can rewrite +p_BqBo as -p_BoBq, thus solving for p_AoBo = p_AoAf +
  // p_AfBq - p_BoBq_A.
  Vector3d p_AoBo = p_AoAf + p_AfBq - p_BoBq_A;
  // Finally we combine the components to form the transform X_AB.
  return RigidTransformd(R_AB, p_AoBo);
};

// Tests whether OBBs overlap. There are 15 cases to test, each covering a
// separating axis between the two bounding boxes. The first 3 cases use the
// axes of Frame A, the next 3 cases use the axes of Frame B, and the remaining
// 9 cases use the axes defined by the cross product of axes from Frame A and
// Frame B. We also test that it is robust for the case of parallel boxes.
GTEST_TEST(OBBTest, TestObbOverlap) {
  // One box is fully contained in the other and they are parallel.
  Obb a = Obb(RigidTransformd(Vector3d(1, 2, 3)), Vector3d(1, 2, 1));
  Obb b = Obb(RigidTransformd(Vector3d(1, 2, 3)), Vector3d(0.5, 1, 0.5));
  RigidTransformd X_AB = RigidTransformd::Identity();
  EXPECT_TRUE(Obb::HasOverlap(a, b, X_AB));

  // To cover the cases of the axes of Frame A, we need to pose box B along
  // each axis. For example, in the case of the x-axis, in a 2D view they would
  // look like:
  // y
  // ^
  // |  -----------------       *
  // |  |               |     *   *
  // |  |      Ao       Af Bq   Bo  *
  // |  |               |     *       *
  // |  |               |       *   *
  // |  -----------------         *
  // -----------------------------------> x
  //
  // For this test, we define Point Bq as the minimum corner of the box B (i.e.,
  // center - half width). We want to pose box B so Bq is the uniquely closest
  // point to box A at a Point Af in the interior of its +x face. The rest of
  // the box extends farther along the +x axis (as suggested in the above
  // illustration). Point Bq will be a small epsilon away from the nearby face
  // either outside (if expect_overlap is false) or inside (if true).
  a = Obb(RigidTransformd(Vector3d(1, 2, 3)), Vector3d(2, 4, 3));
  b = Obb(RigidTransformd(Vector3d(2, 0.5, 4)), Vector3d(3.5, 2, 1.5));
  for (int axis = 0; axis < 3; ++axis) {
    X_AB = calc_corner_transform(a, b, axis, false /* expect_overlap */);
    EXPECT_FALSE(Obb::HasOverlap(a, b, X_AB));
    X_AB = calc_corner_transform(a, b, axis, true /* expect_overlap */);
    EXPECT_TRUE(Obb::HasOverlap(a, b, X_AB));
  }

  // To cover the local axes out of B, we can use the same method by swapping
  // the order of the boxes and then using the inverse of the transform.
  for (int axis = 0; axis < 3; ++axis) {
    X_AB =
        calc_corner_transform(b, a, axis, false /* expect_overlap */).inverse();
    EXPECT_FALSE(Obb::HasOverlap(a, b, X_AB));
    X_AB =
        calc_corner_transform(b, a, axis, true /* expect_overlap */).inverse();
    EXPECT_TRUE(Obb::HasOverlap(a, b, X_AB));
  }

  // To cover the remaining 9 cases, we need to pose an edge from box B along
  // an edge from box A. The axes that the edges are aligned with form the
  // two inputs into the cross product for the separating axis. For example,
  // in the following illustration, Af lies on the edge aligned with A's y-axis.
  // Assuming that Bq lies on an edge aligned with B's x-axis, this would form
  // the case testing the separating axis Ay × Bx.
  //                       _________
  //   +z                 /________/\              .
  //    ^                 \        \ \             .
  //    |   ______________ Bq       \ \            .
  //    |  |\             Af \  Bo   \ \           .
  //    |  | \ _____________\ \       \ \          .
  // +y |  | |      Ao      |  \_______\/          .
  //  \ |  \ |              |                      .
  //   \|   \|______________|                      .
  //    -----------------------------------> +x
  //
  // For this test, we define point Bq on the minimum edge of the box in its
  // own frame (i.e., center - half width + an offset along the edge). We want
  // to pose box B so Bq is the uniquely closest point to A at a Point Af on the
  // edge between the +x and +z face of box A. The rest of the box extends
  // farther along the +x and +z axis (as suggested in the above illustration).
  // Point Bq will be a small epsilon away from the nearby edge either outside
  // (if expect_overlap is false) or inside (if true).
  for (int a_axis = 0; a_axis < 3; ++a_axis) {
    for (int b_axis = 0; b_axis < 3; ++b_axis) {
      X_AB =
          calc_edge_transform(a, b, a_axis, b_axis, false /* expect_overlap */);
      // Separate along a's y-axis and b's x-axis.
      EXPECT_FALSE(Obb::HasOverlap(a, b, X_AB));
      X_AB =
          calc_edge_transform(a, b, a_axis, b_axis, true /* expect_overlap */);
      // Separate along a's y-axis and b's x-axis.
      EXPECT_TRUE(Obb::HasOverlap(a, b, X_AB));
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
