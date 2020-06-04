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

// We use three frames.
// 1. M is the frame of the mesh.
// 2. F is a temporary frame with Fo = Mo, but Fx, Fy, Fz are from the
//    covariance method.
// 3. B is the canonical frame of the oriented bounding box. It has the same
//    basis vectors as F but its origin Bo is at the center of the box.
// Notice that the center of the oriented bounding box is not necessarily at
// the average position of the points. For example, the three points (0,0,0),
// (1,0,0), (0,1,0) have the average position (1/3, 1/3, 0), but the center
// of its oriented bounding box is (1/4, 1/4, 0).
template <class MeshType>
Obb BVH<MeshType>::ComputeBoundingVolume(
    const MeshType& mesh_M,
    const typename std::vector<CentroidPair>::iterator& start,
    const typename std::vector<CentroidPair>::iterator& end) {
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
  // TODO(DamrongGuoy): Study more about the order of eigenvectors from Eigen
  //  library. I do not know whether our covariance
  //  matrix can guarantee that the three eigenvectors in such order can form
  //  an appropriate rotation matrix (meaning that it preserves orientation).
  //  https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html

  // Let F be the frame with Fo = Mo but its basis vectors are calculated from
  // the eigenvectors of the covariance matrix. Eigen documentation says that
  // the kᵗʰ eigenvector corresponds to the kᵗʰ eigenvalue, which is sorted
  // in increasing order. Eigen documentation also says that the eigenvectors
  // are normalized to have (Euclidean) norm equal to one. The largest
  // eigenvalue is the last one.
  const Vector3d& Fx_M = es.eigenvectors().col(2);
  const Vector3d& Fy_M = es.eigenvectors().col(1);
  const Vector3d Fz_M = Fx_M.cross(Fy_M);
  const RotationMatrixd R_MF =
      RotationMatrixd::MakeFromOrthonormalColumns(Fx_M, Fy_M, Fz_M);
  const RigidTransformd X_MF(R_MF);

  // TODO(DamrongGuoy): Come up with a better notation than `low_F` and
  //  `high_F`.  Perhaps p_FLow and p_FHigh ? p_FL and p_FH? p_FC0 and p_FC1?
  Vector3d low_F, high_F;
  FindCorners(mesh_M, vertices, X_MF, &low_F, &high_F);

  const Vector3d low_M = X_MF * low_F;
  const Vector3d high_M = X_MF * high_F;

  const Vector3d center_M = (high_M + low_M) / 2;
  // The canonical frame B of the oriented bounding box has the same basis
  // vectors as frame F, but it is centered mid-way between the high point and
  // the low point.
  const RigidTransformd X_MB(R_MF, center_M);
  const Vector3d half_width_B = (high_F - low_F) / 2;
  return Obb(X_MB, half_width_B);
}

template <class MeshType>
void BVH<MeshType>::FindCorners(
    const MeshType& mesh_M,
    const std::set<typename MeshType::VertexIndex>& vertices,
    const RigidTransformd& X_MF, Vector3d* low_F, Vector3d* high_F) {
  low_F->setConstant(std::numeric_limits<double>::max());
  high_F->setConstant(std::numeric_limits<double>::lowest());
  RigidTransformd X_FM = X_MF.inverse();
  for (typename MeshType::VertexIndex v : vertices) {
    const Vector3d p_FV = X_FM * mesh_M.vertex(v).r_MV();
    for (int axis = 0; axis < 3; ++axis) {
      if (p_FV[axis] < (*low_F)[axis]) (*low_F)[axis] = p_FV[axis];
      if (p_FV[axis] > (*high_F)[axis]) (*high_F)[axis] = p_FV[axis];
    }
  }
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

}  // namespace internal
}  // namespace geometry
}  // namespace drake

template class drake::geometry::internal::BVH<
    drake::geometry::SurfaceMesh<double>>;
template class drake::geometry::internal::BVH<
    drake::geometry::VolumeMesh<double>>;
