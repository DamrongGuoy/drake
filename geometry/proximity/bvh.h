#pragma once

#include <array>
#include <memory>
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
struct MeshTraits;

template <>
struct MeshTraits<SurfaceMesh<double>> {
  static constexpr int kMaxElementPerLeaf = 3;
};

template <>
struct MeshTraits<VolumeMesh<double>> {
  static constexpr int kMaxElementPerLeaf = 1;
};

template <class MeshType>
class BVHNode {
 public:
  BVHNode(Obb obb, typename MeshType::ElementIndex index)
      : obb_(std::move(obb)),
        child_(LeafData{1, {index}}) {}

  static constexpr int kMaxElementPerLeaf =
      MeshTraits<MeshType>::kMaxElementPerLeaf;

  // A leaf node can store as many element indices as kMaxElementPerLeaf.
  // The actual number of stored element indices is `num_index`.
  struct LeafData {
    int num_index;
    std::array<typename MeshType::ElementIndex, kMaxElementPerLeaf> indices;
  };

  BVHNode(Obb obb, LeafData data)
      : obb_(std::move(obb)),
        child_(std::move(data)) {}

  BVHNode(Obb obb, std::unique_ptr<BVHNode<MeshType>> left,
         std::unique_ptr<BVHNode<MeshType>> right)
      : obb_(std::move(obb)),
        child_(NodeChildren(std::move(left), std::move(right))) {}

  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BVHNode)

  /* Returns the bounding volume.  */
  const Obb& obb() const { return obb_; }

  /* Returns the number of element indices.
   @pre Assumes that is_leaf() returns true. */
  int num_element_index() const {
    return std::get<LeafData>(child_).num_index;
  }

  /* Returns the i-th entry of the leaf data, which is an
   index into the mesh's elements.
   @pre Assumes that is_leaf() returns true.
   @pre Assumes that `i` is less than LeafData::num_index */
  typename MeshType::ElementIndex element_index(int i) const {
    DRAKE_DEMAND(0 <= i && i < std::get<LeafData>(child_).num_index);
    return std::get<LeafData>(child_).indices[i];
  }

  /* Returns the left child branch.
   @pre Assumes that is_leaf() returns false.  */
  const BVHNode<MeshType>& left() const {
    return *(std::get<NodeChildren>(child_).left);
  }

  /* Returns the right child branch.
   @pre Assumes that is_leaf() returns false.  */
  const BVHNode<MeshType>& right() const {
    return *(std::get<NodeChildren>(child_).right);
  }

  /* Returns whether this is a leaf node as a opposed to a branch node.  */
  bool is_leaf() const {
    return std::holds_alternative<LeafData>(child_);
  }

  /* Compare this leaf node with the other leaf node.
   @pre Assumes that this->is_leaf() and other_leaf->is_leaf() are true.  */
  bool EqualLeaf(const BVHNode<MeshType>& other_leaf) const {
    if (this == &other_leaf) return true;
    if (this->num_element_index() != other_leaf.num_element_index()) {
      return false;
    }
    for (int i = 0; i <  this->num_element_index(); ++i) {
      if (this->element_index(i) != other_leaf.element_index(i)) {
        return false;
      }
    }
    return true;
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

  // If this is a leaf node then the child refers to indices into the mesh's
  // elements (i.e., triangles or tetrahedrons) bounded by the node's bounding
  // volume. Otherwise, it refers to child nodes further down the tree.
  // std::variant<typename MeshType::ElementIndex, NodeChildren> child_;
  std::variant<LeafData, NodeChildren> child_;
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

  /* Perform a query of this bvh's mesh elements against the given bvh's
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
        const int num_a_element = node_a.num_element_index();
        const int num_b_element = node_b.num_element_index();
        for (int a = 0; a < num_a_element; ++a) {
          for (int b = 0; b < num_b_element; ++b) {
            BVHCallbackResult result =
              callback(node_a.element_index(a), node_b.element_index(b));
            if (result == BVHCallbackResult::Terminate) {
              return;  // Exit early.
            }
          }
        }
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
        [&result](IndexType a,
                  typename OtherMeshType::ElementIndex b) -> BVHCallbackResult {
      result.emplace_back(a, b);
      return BVHCallbackResult::Continue;
    };
    Collide(bvh, X_AB, callback);
    return result;
  }

  /* Compares the two BVH instances for exact equality down to the last bit.
   Assumes that the quantities are measured and expressed in the same frame. */
  bool Equal(const BVH<MeshType>& other) const {
    if (this == &other) return true;
    return EqualTrees(this->root_node(), other.root_node());
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
      const typename std::vector<CentroidPair>::iterator& end,
      bool optimize = true);

  // TODO(tehbelinda): Move this function into SurfaceMesh/VolumeMesh directly
  // and rename to CalcElementCentroid(ElementIndex).
  static Vector3<double> ComputeCentroid(const MeshType& mesh,
                                         IndexType i);

  // Tests that the two hierarchy trees, rooted at nodes a and b, are equal in
  // the sense that they have identical structure and equal bounding volumes
  // (see Obb::Equal()).
  static bool EqualTrees(const BVHNode<MeshType>& a,
                         const BVHNode<MeshType>& b);

  static constexpr int kElementVertexCount = MeshType::kDim + 1;

  std::unique_ptr<BVHNode<MeshType>> root_node_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
