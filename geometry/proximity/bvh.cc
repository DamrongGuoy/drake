#include "drake/geometry/proximity/bvh.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;
using Eigen::Matrix3d;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;

template <class MeshType>
BVH<MeshType>::BVH(const MeshType& mesh) {
  // Generate element indices and corresponding centroids. These are used
  // for calculating the split point of the volumes.
  const int num_elements = mesh.num_elements();
  std::vector<CentroidPair> element_centroids;
  for (IndexType i(0); i < num_elements; ++i) {
    element_centroids.emplace_back(i, ComputeCentroid(mesh, i));
  }

  root_node_ =
      BuildBVTree(mesh, element_centroids.begin(), element_centroids.end());
}

template <class MeshType>
std::unique_ptr<BVHNode<MeshType>>
BVH<MeshType>::BuildBVTree(
    const MeshType& mesh_M,
    const typename std::vector<CentroidPair>::iterator& start,
    const typename std::vector<CentroidPair>::iterator& end) {
  // Generate bounding volume.
  Obb obb = ComputeBoundingVolume(mesh_M, start, end);

  const int num_elements = end - start;
  if (num_elements == 1) {
    // Store element index in this leaf node.
    return std::make_unique<BVHNode<MeshType>>(obb, start->first);

  } else {
    // Sort the elements by centroid along the axis of greatest spread.
    // Note: We tried an alternative strategy for building the BVH using a
    // volume-based metric.
    // - Given a parent BV P, we would partition all of its contents into a left
    //   BV, L, and a right BV, R, such that we wanted to minimize (V(L) + V(R))
    //   / V(P). (V(X) is the volume measure of the bounding volume X).
    // - We didn't explore all possible partitions (there are an exponential
    //   number of such possible partitions).
    // - Instead, we ordered the mesh elements along an axis and then would
    //   consider partitions split between two adjacent elements in the sorted
    //   set. This was repeated for each of the axes and the partition with the
    //   minimum value was taken.
    // - We did explore several ordering options, including sorting by centroid,
    //   min, max, and a combination of them when the element overlapped the
    //   partition boundary.
    // This tentative partitioning strategy produced more BV-BV tests in some
    // simple examples than the simple bisection shown below and was abandoned.
    // Some possible reasons for this:
    // - Sorting by an alternate criteria might be a better way to order them.
    // - Only considering split points based on adjacent elements may be
    //   problematic.
    // - The elements individual extents are such they are not typically
    //   axis-aligned so, partitioning between two elements would often produce
    //   child BVs that have non-trivial overlap.
    // Finally, the primitive meshes we are producing are relatively regular and
    // are probably nicely compatible with the median split strategy. If we need
    // to do irregular distribution of elements, a more sophisticated strategy
    // may help. But proceed with caution -- there's no guarantee such a
    // strategy will yield performance benefits.
    int axis{};
    obb.half_width().maxCoeff(&axis);
    // Let B be the canonical frame of obb.
    const auto& Bw_M = obb.pose().rotation().col(axis);

    std::sort(start, end,
              [&Bw_M](const CentroidPair& a, const CentroidPair& b) {
                return Bw_M.dot(a.second) < Bw_M.dot(b.second);
              });

    // Continue with the next branches.
    const typename std::vector<CentroidPair>::iterator mid =
        start + num_elements / 2;
    return std::make_unique<BVHNode<MeshType>>(
        obb, BuildBVTree(mesh_M, start, mid), BuildBVTree(mesh_M, mid, end));
  }
}

template <class MeshType>
Obb BVH<MeshType>::ComputeBoundingVolume(
    const MeshType& mesh_M,
    const typename std::vector<CentroidPair>::iterator& start,
    const typename std::vector<CentroidPair>::iterator& end,
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

  // Prepare unique vertices (by vertex indices) for covariance method.
  // I use set<> instead of unordered_set<> because I think the order of
  // vertices can numerically influence the covariance matrix.
  std::set<typename MeshType::VertexIndex> vertices;
  // Check each mesh element in the given range.
  for (auto pair = start; pair < end; ++pair) {
    const auto& element = mesh_M.element(pair->first);
    // Check each vertex in the element.
    for (int v = 0; v < kElementVertexCount; ++v) {
      vertices.insert(element.vertex(v));
    }
  }
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

template <class MeshType>
Obb BVH<MeshType>::OptimizeObbVolume(
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
Vector3d BVH<MeshType>::CalcVolumeGradient(
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
Obb BVH<MeshType>::CalcOrientedBox(
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
void BVH<MeshType>::CalcCovariance(
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
Vector3d BVH<MeshType>::ComputeCentroid(
    const MeshType& mesh, const IndexType i) {
  Vector3d centroid{0, 0, 0};
  const auto& element = mesh.element(i);
  // Calculate average from all vertices.
  for (int v = 0; v < kElementVertexCount; ++v) {
    const auto& vertex = mesh.vertex(element.vertex(v)).r_MV();
    centroid += vertex;
  }
  centroid /= kElementVertexCount;
  return centroid;
}

template <class MeshType>
bool BVH<MeshType>::EqualTrees(const BVHNode<MeshType>& a,
                               const BVHNode<MeshType>& b) {
  if (&a == &b) return true;

  if (!a.obb().Equal(b.obb())) return false;

  if (a.is_leaf()) {
    if (!b.is_leaf()) {
      return false;
    }
    return a.element_index() == b.element_index();
  } else {
    if (b.is_leaf()) {
      return false;
    }
    return EqualTrees(a.left(), b.left()) && EqualTrees(a.right(), b.right());
  }
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake

template class drake::geometry::internal::BVH<
    drake::geometry::SurfaceMesh<double>>;
template class drake::geometry::internal::BVH<
    drake::geometry::VolumeMesh<double>>;
