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
  if (num_elements <= BVHNode<MeshType>::kMaxElementPerLeaf) {
    typename BVHNode<MeshType>::LeafData data{num_elements, {}};
    for (int i = 0; i < num_elements; ++i) {
      data.indices[i] = (start + i)->first;
    }
    // Store element indices in this leaf node.
    return std::make_unique<BVHNode<MeshType>>(obb, data);
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

  return ComputeObb(mesh_M, vertices, optimize);
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
    return a.EqualLeaf(b);
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
