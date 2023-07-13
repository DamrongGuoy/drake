#include "drake/geometry/proximity/distance_to_shape_callback.h"

#include <drake_vendor/fcl/fcl.h>
#include <gtest/gtest.h>

#include "drake/common/text_logging.h"

// @file Test special cases when two shapes touch each other for
// ComputeNarrowPhaseDistance() and CalcDistanceFallback() in
// distance_to_shape_callback.

namespace drake {
namespace geometry {
namespace internal {
namespace shape_distance {
namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using fcl::Boxd;
using fcl::CollisionObjectd;
using fcl::Sphered;
using math::RigidTransform;
using math::RigidTransformd;
using std::make_shared;

// When a sphere is just touching a shape, confirms that the two witness
// points are at the same locations and nhat_BA_W is not NaN.  Other unit
// tests already checked the same code path as this test. Here we add this
// test to emphasize this special case.  We use Sphere-Box as a
// representative sample.
GTEST_TEST(ComputeNarrowPhaseDistance, sphere_touches_shape) {
  // Sphere
  const double radius = 1;
  CollisionObjectd sphere(make_shared<Sphered>(radius));
  const GeometryId sphere_id = GeometryId::get_new_id();
  EncodedData(sphere_id, true).write_to(&sphere);

  // Box [-1,1]x[-1,1]x[-1,1].
  const double side = 2;
  CollisionObjectd box(make_shared<Boxd>(side, side, side));
  const GeometryId box_id = GeometryId::get_new_id();
  EncodedData(box_id, true).write_to(&box);
  const RigidTransformd X_WB(RigidTransformd::Identity());
  const fcl::DistanceRequestd request{};

  // The sphere touches the box in the middle of a face of the box.
  const RigidTransformd X_WS(Vector3d{radius + side / 2., 0, 0});
  SignedDistancePair<double> result;
  ComputeNarrowPhaseDistance<double>(sphere, X_WS, box, X_WB, request, &result);
  const auto p_WCs = X_WS * result.p_ACa;
  const auto p_WCb = X_WB * result.p_BCb;
  EXPECT_EQ(p_WCs, p_WCb);
  EXPECT_FALSE((isnan(result.nhat_BA_W.array())).any());
  // The sphere A touches the box B on the right face (+x) of the box.
  EXPECT_EQ(Vector3d(1, 0, 0), result.nhat_BA_W);
}

const fcl::DistanceRequestd kFclRequest(
    /* enable_nearest_points_ */ true,
    /* enable_signed_distance */ true,
    /* rel_err_ */ 0.0,
    /* abs_err_ */ 0.0,
    /* distance_tolerance */ 1E-6,
    /* gjk_solver_type_ */ fcl::GJKSolverType::GST_LIBCCD);

// Box on box, face on face of the same size. The witness points could be
// anywhere on the common touching face.
//
//        Y
//        ↑
//        |
//        |
//     +----+----+
//     | A  | B  |    ---→ X
//     +----+----+
//
GTEST_TEST(CalcDistanceFallback, box_touches_box_1) {
  const RigidTransformd X_WA = RigidTransformd::Identity();
  CollisionObjectd box_A(make_shared<Boxd>(1.0, 1.0, 1.0),
                         X_WA.rotation().matrix(), X_WA.translation());
  const GeometryId box_A_id = GeometryId::get_new_id();
  EncodedData(box_A_id, true).write_to(&box_A);

  const RigidTransformd X_WB(Vector3d{1.0, 0, 0});
  CollisionObjectd box_B(make_shared<Boxd>(1.0, 1.0, 1.0),
                         X_WB.rotation().matrix(), X_WB.translation());
  const GeometryId box_B_id = GeometryId::get_new_id();
  EncodedData(box_B_id, true).write_to(&box_B);

  SignedDistancePair<double> result;
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);

  ASSERT_EQ(result.distance, 0);
  EXPECT_FALSE((isnan(result.nhat_BA_W.array())).any());

  // Temporary info
  // The witness points are at corner vertices.
  EXPECT_EQ(result.p_ACa, Vector3d(0.5, 0.5, 0.5));
  EXPECT_EQ(result.p_BCb, Vector3d(-0.5, 0.5, 0.5));

  //Sanity check the (sub)gradient
  box_A.setTranslation(X_WA.translation() + result.nhat_BA_W);
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);
  EXPECT_GT(result.distance, 0.577);
  EXPECT_LT(result.distance, 0.578);
}

// Box on box, small face on big face. The witness points could be anywhere
// on the small face.
//
//       Y
//       ↑
//       |
//       |
//          +----+
//          |    |
//     +----+    |
//     | A  | B  |    ---→ X
//     +----+    |
//          |    |
//          +----+
//
GTEST_TEST(CalcDistanceFallback, box_touches_box_2) {
  // Box A [-1,1]x[-1,1]x[-1,1] has size 2x2x2 meters.
  const RigidTransformd X_WA = RigidTransformd::Identity();
  CollisionObjectd box_A(make_shared<Boxd>(2.0, 2.0, 2.0),
      X_WA.rotation().matrix(), X_WA.translation());
  const GeometryId box_A_id = GeometryId::get_new_id();
  EncodedData(box_A_id, true).write_to(&box_A);

  // Box B [-1,1]x[-3,3]x[-1,1] has size 2x6x2 meters. Translating it by
  // (2,0,0) to [1,3]x[-3,3]x[-1,1]
  const RigidTransformd X_WB(Vector3d{2.0, 0, 0});
  CollisionObjectd box_B(make_shared<Boxd>(2.0, 6.0, 2.0),
      X_WB.rotation().matrix(), X_WB.translation());
  const GeometryId box_B_id = GeometryId::get_new_id();
  EncodedData(box_B_id, true).write_to(&box_B);

  SignedDistancePair<double> result;
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);

  ASSERT_EQ(result.distance, 0);
  EXPECT_FALSE((isnan(result.nhat_BA_W.array())).any());

  // Temporary information
  // The small box's witness point is at a corner vertex.
  EXPECT_EQ(result.p_ACa, Vector3d(1.0, 1.0, 1.0));
  // The large box's witness point is on its edge between
  // the vertex (min Bx, min By, max Bz) = (-1,-3,1) and
  // the vertex (min Bx, max By, max Bz) = (-1,3,1).
  EXPECT_EQ(result.p_BCb, Vector3d(-1.0, 1.0, 1.0));

  //Sanity check the (sub)gradient
  box_A.setTranslation(X_WA.translation() + result.nhat_BA_W);
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);
  EXPECT_GT(result.distance, 0.707);
  EXPECT_LT(result.distance, 0.708);
}

// Two boxes intersect at an edge.  The witness points could be anywhere on
// the edge.
//
//        Y
//        ↑
//        | +----+
//        | | B  |
//     +----+----+
//     | A  |  --------→ X
//     +----+
//
GTEST_TEST(CalcDistanceFallback, box_touches_box_3) {
  const RigidTransformd X_WA = RigidTransformd::Identity();
  CollisionObjectd box_A(make_shared<Boxd>(1.0, 1.0, 1.0),
                         X_WA.rotation().matrix(), X_WA.translation());
  const GeometryId box_A_id = GeometryId::get_new_id();
  EncodedData(box_A_id, true).write_to(&box_A);

  const RigidTransformd X_WB(Vector3d{1.0, 1.0, 0});
  CollisionObjectd box_B(make_shared<Boxd>(1.0, 1.0, 1.0),
                         X_WB.rotation().matrix(), X_WB.translation());
  const GeometryId box_B_id = GeometryId::get_new_id();
  EncodedData(box_B_id, true).write_to(&box_B);

  SignedDistancePair<double> result;
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);

  ASSERT_EQ(result.distance, 0);
  EXPECT_FALSE((isnan(result.nhat_BA_W.array())).any());

  // Temporary info
  // The witness points are at corner vertices.
  EXPECT_EQ(result.p_ACa, Vector3d(0.5, 0.5, 0.5));
  EXPECT_EQ(result.p_BCb, Vector3d(-0.5, -0.5, 0.5));

  //Sanity check the (sub)gradient
  box_A.setTranslation(X_WA.translation() + result.nhat_BA_W);
  CalcDistanceFallback<double>(box_A, box_B, kFclRequest, &result);
  EXPECT_GT(result.distance, 0.816);
  EXPECT_LT(result.distance, 0.817);
}

}  // namespace
}  // namespace shape_distance
}  // namespace internal
}  // namespace geometry
}  // namespace drake
