#pragma once

#include <utility>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {
namespace internal {

/** Oriented bounding box used in a BVH (bounding volume hierarchy). The
 box is defined in a canonical frame B such that it is centered on Bo and its
 extents are aligned with B's axes. However, the box is posed in frame H of the
 BVH.

         |
         |
       H |
         +------
        /          /
       /        B ╱
                ／＼
             ／     ＼
 */
class Obb {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Obb)

  Obb(math::RigidTransformd X_HB, Vector3<double> half_width)
      : pose_(std::move(X_HB)), half_width_(std::move(half_width)) {
    DRAKE_DEMAND(half_width.x() >= 0.0);
    DRAKE_DEMAND(half_width.y() >= 0.0);
    DRAKE_DEMAND(half_width.z() >= 0.0);

    PadBoundary();
  }

  const math::RigidTransformd& pose() const { return pose_; }

  const Vector3<double>& half_width() const { return half_width_; }

  double CalcVolume() const {
    // Double the three half widths using * 8 instead of repeating * 2 three
    // times to help the compiler out.
    return half_width_[0] * half_width_[1] * half_width_[2] * 8;
  }

  /** Check whether two oriented bounding boxes overlap. The first box is
   posed in frame M of the first BVH, and the second box is posed in frame N of
   the second BVH.
   */
  static bool HasOverlap(const Obb& box0_M, const Obb& box1_N,
                         const math::RigidTransformd& X_MN);

 private:
  friend class ObbTester;

  void PadBoundary();

  static constexpr double kTolerance = 2e-14;

  math::RigidTransformd pose_;
  Vector3<double> half_width_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
