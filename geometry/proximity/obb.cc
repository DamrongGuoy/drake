#include "drake/geometry/proximity/obb.h"

#include <algorithm>
#include <limits>

#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;

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

/* Calculates the oriented bounding box from a given point set and a given
 orientation. The point set is expressed in frame M of a mesh, and the
 orientation of the box is specified by a rotation matrix R_MF, where F is
 a rotating frame of M around its origin Mo. The oriented bounding box has
 its canonical frame B. The box fits the point set, and frame B is aligned
 with frame F.

 This picture illustrates frame M, frame F, and frame B. The vectors Mz,
 Fx, and Bz are perpendicular to the page.

                                                 Bx
                              By               ⇗
                                ⇖       ⋰ ⋱ ⇗
                                  ⇖   ⋰   ⇗ ⋱
                                    ⇖   ⇗    ⋰
                                  ⋰  Bo   ⋰
                                ⋰       ⋰
                                ⋱     ⋰
                                  ⋱ ⋰

              My
              ↑
              ↑
              ↑
      Fy      ↑      Fx
        ⇖     ↑     ⇗
          ⇖   ↑   ⇗
            ⇖ ↑ ⇗
              Mo → → → → → → → Mx

  */
template <class MeshType>
Obb CalcOrientedBox(
    const MeshType& mesh_M,
    const std::set<typename MeshType::VertexIndex>& vertices,
    const RotationMatrixd& R_MF) {
  // L and H are the lowest corner and the highest corner of the oriented box.
  Vector3d p_FL, p_FH;
  p_FL.setConstant(std::numeric_limits<double>::max());
  p_FH.setConstant(std::numeric_limits<double>::lowest());
  const RotationMatrixd R_FM = R_MF.inverse();
  for (typename MeshType::VertexIndex v : vertices) {
    const Vector3d p_FV = R_FM * mesh_M.vertex(v).r_MV();
    for (int axis = 0; axis < 3; ++axis) {
      if (p_FV[axis] < p_FL[axis]) p_FL[axis] = p_FV[axis];
      if (p_FV[axis] > p_FH[axis]) p_FH[axis] = p_FV[axis];
    }
  }
  // An expression of a vector in frame F is the same as its expression
  // in frame B because frame F and frame B are aligned. Therefore,
  // the half-width vector of the oriented bounding box is:
  //
  //   p_BoH_B = p_BoH_F = p_LH_F / 2 = (p_FH - p_FL) / 2
  //
  // Notice that we calculate p_BoH_B without knowing Bo yet.
  const Vector3d p_BoH_B = (p_FH - p_FL) / 2.;

  const Vector3d p_ML = R_MF * p_FL;
  const Vector3d p_MH = R_MF * p_FH;
  const Vector3d p_MBo = (p_ML + p_MH) / 2.;
  return Obb(RigidTransformd(R_MF, p_MBo), p_BoH_B);
}

template <class MeshType>
Vector3d CalcVolumeGradient(
    const MeshType& mesh_M,
    const std::set<typename MeshType::VertexIndex>& vertices,
    const Obb& box) {
  const double volume_0 = box.CalcVolume();
  // B is the frame of the given box.
  const RotationMatrixd& R_MB = box.pose().rotation();

  // TODO(DamrongGuoy): Come up with a better guess of step size h.
  constexpr double h = 5. * M_PI / 180.;  // 5-degree step.
  // Br, Bp, By are frames of the rolled box, pitched box, and yawed box
  // respectively.
  static const RotationMatrixd R_BBr(RollPitchYawd(h, 0., 0.));
  static const RotationMatrixd R_BBp(RollPitchYawd(0., h, 0.));
  static const RotationMatrixd R_BBy(RollPitchYawd(0., 0., h));
  const RotationMatrixd R_MBr = R_MB * R_BBr;
  const RotationMatrixd R_MBp = R_MB * R_BBp;
  const RotationMatrixd R_MBy = R_MB * R_BBy;

  const double volume_roll =
      CalcOrientedBox(mesh_M, vertices, R_MBr).CalcVolume();
  const double volume_pitch =
      CalcOrientedBox(mesh_M, vertices, R_MBp).CalcVolume();
  const double volume_yaw =
      CalcOrientedBox(mesh_M, vertices, R_MBy).CalcVolume();

  const double diff_volume_roll = volume_roll - volume_0;
  const double diff_volume_pitch = volume_pitch - volume_0;
  const double diff_volume_yaw = volume_yaw - volume_0;

  return {diff_volume_roll / h, diff_volume_pitch / h, diff_volume_yaw / h};
}

template <class MeshType>
Obb OptimizeObbVolume(
    const MeshType& mesh_M,
    const std::set<typename MeshType::VertexIndex>& vertices,
    const Obb& box0) {

  const RotationMatrixd& R_MF0 = box0.pose().rotation();
  const double volume0 = box0.CalcVolume();

  // Locally minimize volume of the box as a function of the roll-pitch-yaw
  // angles (Euler angles).
  double volume = volume0;
  Obb box = box0;
  // Gradient has units of volume/radian.
  const Vector3d direction = CalcVolumeGradient(mesh_M, vertices, box0);
  // Set initial step to attempt a 10% volume reduction.
  double increment = volume0 / (10. * direction.norm());
  const double min_improvement = 0.001;  // 0.1% or give up.
  const double min_increment = increment / 1000000.;
  double step = 0.;
  for (int i = 0; i < 20; ++i) {
    step -= increment;
    const Obb try_box = CalcOrientedBox(
        mesh_M, vertices,
        R_MF0 * RotationMatrixd(RollPitchYawd(step * direction)));
    const double try_volume = try_box.CalcVolume();

    if (try_volume < volume) {
      const double improvement = (volume - try_volume) / volume;
      volume = try_volume;
      box = try_box;
      if (improvement < min_improvement) break;
      increment *= 1.5;  // grow slowly
      continue;
    }

    // Volume does not decrease.
    step += increment;  // back to previous best
    if (increment <= min_increment) break;
    increment /= 10.;  // shrink fast
  }

  // Make sure to return the box that has volume less than or equal to the
  // original.
  if (box.CalcVolume() < volume0)
    return box;
  else
    return box0;
}

template <class MeshType>
void CalcCovariance(
    const MeshType& mesh_M,
    const std::set<typename MeshType::VertexIndex>& vertices,
    Matrix3d* covariance_M) {
  DRAKE_DEMAND(vertices.size() > 0);

  // We divide by the number of vertices two times: one for centroid and
  // another for covariance matrix. That's why we save it here.
  const double one_over_n = 1.0 / vertices.size();

  Vector3d centroid_M(0., 0., 0.);
  for (typename MeshType::VertexIndex v : vertices) {
    centroid_M += mesh_M.vertex(v).r_MV();
  }
  centroid_M *= one_over_n;

  covariance_M->setZero();
  for (typename MeshType::VertexIndex v : vertices) {
    const Vector3d& p_MV = mesh_M.vertex(v).r_MV();
    // Displacement vector from the centroid to the vertex V expressed in
    // frame M.
    const Vector3d p_CV_M = p_MV - centroid_M;
    *covariance_M += p_CV_M * p_CV_M.transpose();
  }
  *covariance_M *= one_over_n;
}

template <class MeshType>
Obb ComputeObb(const MeshType& mesh_M,
               const std::set<typename MeshType::VertexIndex>& vertices,
               const bool optimize) {
  //
  // Ideas behind the algorithm.
  //
  // We use these three frames:
  // 1. M is the frame of the mesh.
  // 2. F is a temporary frame from rotation of M around its origin Mo.
  //    The rotation R_MF comes from the covariance method and local
  //    optimization by numerical gradient of volume of the oriented
  //    bounding box.
  // 3. B is the canonical frame of the oriented bounding box.
  //    Basis vectors of B are aligned with basis vectors of F.
  //
  // The following picture illustrates the three frames M, F, and B. The vectors
  // Mz, Fz, and Bz are perpendicular to the page.
  //
  //                                               Bx
  //                            By               ⇗
  //                              ⇖       ⋰ ⋱ ⇗
  //                                ⇖   ⋰   ⇗ ⋱
  //                                  ⇖   ⇗    ⋰
  //                                ⋰  Bo   ⋰
  //                              ⋰       ⋰
  //                              ⋱     ⋰
  //                                ⋱ ⋰
  //
  //            My
  //            ↑
  //            ↑
  //            ↑
  //    Fy      ↑      Fx
  //      ⇖     ↑     ⇗
  //        ⇖   ↑   ⇗
  //          ⇖ ↑ ⇗
  //            Mo → → → → → → → Mx
  //
  //
  //
  // Initially the algorithm computes the frame F as:
  // 0. Use covariance matrix of the point set to solve for the rotation
  //    R_MF of frame F expressed in M.
  //
  // The algorithm repeats these steps towards a local minimal volume of the
  // oriented bounding box.
  //
  // 1. Use R_MF to calculate the oriented bounding box B of the point set.
  //
  // 2. Use the half-width vector to calculate volume of the oriented bounding
  //    box.
  //              volume = 8 * p_BH.x() * p_BH.y() * p_BH.z()
  //
  // 4. Calculate numerical gradient of the volume as a function of
  //    additional roll-pitch-yaw angles [r,p,y],
  //              R_FB(r,p,y) = RotationMatrixd(RollPitchYawd(r,p,y)).
  //
  //    Set R_MF = R_MF * R_FB, calculate volume_new by step 1-3
  //    ∂volume/∂r = (volume_new - volume) / r
  //    ∂volume/∂p = (volume_new - volume) / p
  //    ∂volume/∂y = (volume_new - volume) / y
  //
  //    grad_volume(r,p,y) = (∂volume/∂r, ∂volume/∂p, ∂volume/∂y)
  //
  // 5. Update frame F and repeat 1-4.
  //
  // At the convergence, we perform the final computation:
  // 6. Calculate Bo from the two extremal points expressed in M.
  // 7. Define the pose X_MB from the rotation R_MF and Bo.
  // 8. Return the oriented bounding box with pose X_MB and the half-width
  //    vector expressed in B.
  //
  //
  // Why Bo is not the centroid of the point set?
  //
  // For a given point set, the center Bo of the oriented bounding box depends
  // on the orientation of the box. For a simple example, consider a set of
  // three points Q, R, S with p_MQ = (0,0,0), p_MR = (1,0,0), and
  // p_MS = (0,1,0). The oriented bounding box that is aligned with M has
  // its center Bo at p_MBo = (1/2, 1/2, 0) as shown in the first picture
  // below. The oriented bounding box of 45-degree rotation around Mz has the
  // center Bo at p_MBo = (1/4, 1/4, 0) as shown in the second picture below.
  // In this second case, Bo is half way between Q and the midpoint between R
  // and S, i.e., p_MBo = (p_MQ + (p_MR + p_MS)/2)/2.
  //
  //          My
  //          ↑
  //          ↑   By
  //          ↑   ⇑
  //  (0,1,0) S┄┄┄┄┄┄┄
  //          ┊  Bo   ┊
  //          ┊   +   ┊⇒ Bx
  //          ┊       ┊
  //  (0,0,0) Q┄┄┄┄┄┄┄R→ → → → → Mx
  //                 (1,0,0)
  //
  //
  //          My
  //          ↑
  //          ↑
  //     By   S(0,1,0)
  //       ⇖ ⋰↑⋱
  //       ⋰  ↑  ⋱ ⇗Bx
  //       ⋱  ↑ +Bo⋱
  //         ⋱Q→ → → ⋱R → → → → → Mx
  //    (0,0,0)⋱    ⋰ (1,0,0)
  //             ⋱⋰
  //
  // In both cases, Bo is not the centroid of {Q,R,S}. The position of
  // the centroid C is p_MC = (1/3, 1/3, 0).
  //

  // Covariance matrix is expressed in frame M of the mesh.
  Matrix3d covariance_M;
  CalcCovariance(mesh_M, vertices, &covariance_M);

  Eigen::SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(covariance_M);
  // Basis vectors of frame F are calculated from the eigenvectors of the
  // covariance matrix. Eigen library documentation says that the kᵗʰ
  // eigenvector corresponds to the kᵗʰ eigenvalue, which is sorted
  // in increasing order. Eigen library documentation also says that the
  // eigenvectors are normalized to have (Euclidean) norm equal to one. The
  // largest eigenvalue is the last one.
  const Vector3d& Fx_M = es.eigenvectors().col(2);
  const Vector3d& Fy_M = es.eigenvectors().col(1);
  const Vector3d Fz_M = Fx_M.cross(Fy_M);

  Obb box = CalcOrientedBox(
      mesh_M, vertices,
      RotationMatrixd::MakeFromOrthonormalColumns(Fx_M, Fy_M, Fz_M));

  if (optimize) {
    box = OptimizeObbVolume(mesh_M, vertices, box);
  }

  return box;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake

template drake::geometry::internal::Obb
drake::geometry::internal::ComputeObb(
    const drake::geometry::SurfaceMesh<double>& mesh_M,
    const std::set<drake::geometry::SurfaceMesh<double>::VertexIndex>& vertices,
    const bool optimize);

template drake::geometry::internal::Obb
drake::geometry::internal::ComputeObb(
    const drake::geometry::VolumeMesh<double>& mesh_M,
    const std::set<drake::geometry::VolumeMesh<double>::VertexIndex>& vertices,
    const bool optimize);
