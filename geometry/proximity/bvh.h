#pragma once

#include <memory>
#include <set>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

#include "drake/geometry/proximity/obb.h"
#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

template <class MeshType>
class BVHNode {
 public:
  BVHNode(Obb obb, typename MeshType::ElementIndex index)
      : obb_(std::move(obb)), child_(index) {}

  BVHNode(Obb obb, std::unique_ptr<BVHNode<MeshType>> left,
         std::unique_ptr<BVHNode<MeshType>> right)
      : obb_(std::move(obb)),
        child_(NodeChildren(std::move(left), std::move(right))) {}

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BVHNode)

  /** Returns the bounding volume.  */
  const Obb& obb() const { return obb_; }

  /** Returns the index into the mesh's elements.
   @pre Assumes that is_leaf() returns true.  */
  typename MeshType::ElementIndex element_index() const {
    return std::get<typename MeshType::ElementIndex>(child_);
  }

  /** Returns the left child branch.
   @pre Assumes that is_leaf() returns false.  */
  const BVHNode<MeshType>& left() const {
    return *(std::get<NodeChildren>(child_).left);
  }

  /** Returns the right child branch.
   @pre Assumes that is_leaf() returns false.  */
  const BVHNode<MeshType>& right() const {
    return *(std::get<NodeChildren>(child_).right);
  }

  /** Returns whether this is a leaf node as a opposed to a branch node.  */
  bool is_leaf() const {
    return std::holds_alternative<typename MeshType::ElementIndex>(child_);
  }

 private:
  struct NodeChildren {
    std::unique_ptr<BVHNode<MeshType>> left;
    std::unique_ptr<BVHNode<MeshType>> right;

    NodeChildren(std::unique_ptr<BVHNode<MeshType>> left_in,
                 std::unique_ptr<BVHNode<MeshType>> right_in)
        : left(std::move(left_in)), right(std::move(right_in)) {
      DRAKE_DEMAND(left != nullptr);
      DRAKE_DEMAND(right != nullptr);
      DRAKE_DEMAND(left != right);
    }

    NodeChildren(const NodeChildren& other)
        : NodeChildren{std::make_unique<BVHNode<MeshType>>(*other.left),
                       std::make_unique<BVHNode<MeshType>>(*other.right)} {}

    NodeChildren& operator=(NodeChildren other) {
      std::swap(left, other.left);
      std::swap(right, other.right);
      return *this;
    }
  };

  Obb obb_;

  // If this is a leaf node then the child refers to an index into the mesh's
  // elements (i.e., a tri or a tet) bounded by the node's bounding volume.
  // Otherwise, it refers to child nodes further down the tree.
  std::variant<typename MeshType::ElementIndex, NodeChildren> child_;
};

enum class BVHCallbackResult {
  Continue,
  Terminate
};

template <class MeshType, class OtherMeshType>
using BVHCallback = std::function<BVHCallbackResult(
    typename MeshType::ElementIndex, typename OtherMeshType::ElementIndex)>;

template <class MeshType>
class BVH {
 public:
  using IndexType = typename MeshType::ElementIndex;

  explicit BVH(const MeshType& mesh);

  BVH(const BVH& bvh) { *this = bvh; }

  BVH& operator=(const BVH& bvh) {
    if (&bvh == this) return *this;

    root_node_ = std::make_unique<BVHNode<MeshType>>(*bvh.root_node_);
    return *this;
  }

  BVH(BVH&&) = default;
  BVH& operator=(BVH&&) = default;

  const BVHNode<MeshType>& root_node() const { return *root_node_; }

  /** Perform a query of this bvh's mesh elements against the given bvh's
   mesh elements and runs the callback for each unculled pair.  */
  template <class OtherMeshType>
  void Collide(const BVH<OtherMeshType>& bvh,
               const math::RigidTransformd& X_AB,
               BVHCallback<MeshType, OtherMeshType> callback) const {
    using NodePair =
    std::pair<const BVHNode<MeshType>&, const BVHNode<OtherMeshType>&>;
    std::stack<NodePair, std::vector<NodePair>> node_pairs;
    node_pairs.emplace(root_node(), bvh.root_node());

    while (!node_pairs.empty()) {
      const auto& [node_a, node_b] = node_pairs.top();
      node_pairs.pop();

      // Check if the bounding volumes overlap.
      if (!Obb::HasOverlap(node_a.obb(), node_b.obb(), X_AB)) {
        continue;
      }

      // Run the callback on the pair if they are both leaf nodes, otherwise
      // check each branch.
      if (node_a.is_leaf() && node_b.is_leaf()) {
        BVHCallbackResult result =
            callback(node_a.element_index(), node_b.element_index());
        if (result == BVHCallbackResult::Terminate) return;  // Exit early.
      } else if (node_b.is_leaf()) {
        node_pairs.emplace(node_a.left(), node_b);
        node_pairs.emplace(node_a.right(), node_b);
      } else if (node_a.is_leaf()) {
        node_pairs.emplace(node_a, node_b.left());
        node_pairs.emplace(node_a, node_b.right());
      } else {
        node_pairs.emplace(node_a.left(), node_b.left());
        node_pairs.emplace(node_a.right(), node_b.left());
        node_pairs.emplace(node_a.left(), node_b.right());
        node_pairs.emplace(node_a.right(), node_b.right());
      }
    }
  }

  template <class OtherMeshType>
  std::vector<std::pair<IndexType, typename OtherMeshType::ElementIndex>>
  GetCollisionCandidates(const BVH<OtherMeshType>& bvh,
                         const math::RigidTransformd& X_AB) const {
    std::vector<std::pair<IndexType, typename OtherMeshType::ElementIndex>>
        result;
    auto callback =
        [&result](
            IndexType a,
            typename OtherMeshType::ElementIndex b) -> BVHCallbackResult {
          result.emplace_back(a, b);
          return BVHCallbackResult::Continue;
        };
    Collide(bvh, X_AB, callback);
    return result;
  }

 private:
  // Convenience class for testing.
  friend class BVHTester;

  using CentroidPair = std::pair<IndexType, Vector3<double>>;

  static std::unique_ptr<BVHNode<MeshType>> BuildBVTree(
  const MeshType& mesh,
  const typename std::vector<CentroidPair>::iterator& start,
  const typename std::vector<CentroidPair>::iterator& end);

  static Obb ComputeBoundingVolume(
      const MeshType& mesh_M,
      const typename std::vector<CentroidPair>::iterator& start,
      const typename std::vector<CentroidPair>::iterator& end);

  static void FindCorners(
      const MeshType& mesh_M,
      const std::set<typename MeshType::VertexIndex>& vertices,
      const math::RigidTransformd& X_MF, Vector3<double>* low_F,
      Vector3<double>* high_F);

  // M is the frame of the mesh. The return centroid and covariance matrix
  // are expressed in frame M.
  static void CalcCovariance(
      const MeshType& mesh_M,
      const std::set<typename MeshType::VertexIndex>& vertices,
      Matrix3<double>* covariance_M);

  // TODO(tehbelinda): Move this function into SurfaceMesh/VolumeMesh directly
  // and rename to CalcElementCentroid(ElementIndex).
  static Vector3<double> ComputeCentroid(const MeshType& mesh,
                                         IndexType i);

  static constexpr int kElementVertexCount = MeshType::kDim + 1;

  std::unique_ptr<BVHNode<MeshType>> root_node_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
