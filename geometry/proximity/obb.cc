#include "drake/geometry/proximity/obb.h"

#include <algorithm>
#include <limits>

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using math::RotationMatrixd;
using math::RigidTransformd;

bool Obb::HasOverlap(const Obb& box0_M, const Obb& box1_N,
                     const math::RigidTransformd& X_MN) {
  // Let A be the canonical frame of the first box.
  const RigidTransformd& X_MA = box0_M.pose();
  // Let B be the canonical frame of the second box.
  const RigidTransformd& X_NB = box1_N.pose();
  RigidTransformd X_AB = X_MA.inverse() * X_MN * X_NB;

  // We need to split the transform into the position and rotation components,
  // `p_AB` and `R_AB`. For the purposes of streamlining the math below, they
  // will henceforth be named `t` and `r` respectively. We also name the
  // half-width vectors of the two boxes `a` and `b` for convenience.
  const Vector3d& t = X_AB.translation();
  const Matrix3d& r = X_AB.rotation().matrix();
  const Vector3d& a = box0_M.half_width();
  const Vector3d& b = box1_N.half_width();

  // Compute some common subexpressions and add epsilon to counteract
  // arithmetic error, e.g. when two edges are parallel. We use the value as
  // specified from Gottschalk's OBB robustness tests.
  const double kEpsilon = 0.000001;
  Matrix3d abs_r = r.array().abs() + kEpsilon;

  // First category of cases separating along a's axes.
  for (int i = 0; i < 3; ++i) {
    if (abs(t[i]) > a[i] + b.dot(abs_r.block<1, 3>(i, 0))) {
      return false;
    }
  }

  // Second category of cases separating along b's axes.
  for (int i = 0; i < 3; ++i) {
    if (abs(t.dot(r.block<3, 1>(0, i))) >
        b[i] + a.dot(abs_r.block<3, 1>(0, i))) {
      return false;
    }
  }

  // Third category of cases separating along the axes formed from the cross
  // products of a's and b's axes.
  int i1 = 1;
  for (int i = 0; i < 3; ++i) {
    const int i2 = (i1 + 1) % 3;  // Calculate common sub expressions.
    int j1 = 1;
    for (int j = 0; j < 3; ++j) {
      const int j2 = (j1 + 1) % 3;
      if (abs(t[i2] * r(i1, j) - t[i1] * r(i2, j)) >
          a[i1] * abs_r(i2, j) + a[i2] * abs_r(i1, j) + b[j1] * abs_r(i, j2) +
              b[j2] * abs_r(i, j1)) {
        return false;
      }
      j1 = j2;
    }
    i1 = i2;
  }

  return true;
}

void Obb::PadBoundary() {
  const double max_position = pose_.translation().cwiseAbs().maxCoeff();
  const double max_half_width = half_width_.maxCoeff();
  const double scale = std::max(max_position, max_half_width);
  const double incr =
      std::max(scale * std::numeric_limits<double>::epsilon(), kTolerance);
  half_width_ += Vector3d::Constant(incr);
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
