#include "drake/geometry/proximity/bvh.h"

#include <cmath>
#include <set>
#include <unordered_set>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/bvh_to_vtk.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obb.h"
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
using std::ceil;
using std::floor;
using std::function;
using std::log2;
using std::unordered_set;

/* Returns the height of a node in BVH tree. A leaf has height zero.
 Definition from graph theory:
 The **height** of a node in a rooted tree is the number of edges in a maximal
 path, going away from the root (i.e. its nodes have strictly increasing
 depth), that starts at that node and ends at a leaf.
 (https://en.m.wikipedia.org/wiki/Glossary_of_graph_theory_terms#height)  */
template <class MeshType>
int ComputeNodeHeight(const BVHNode<MeshType>& node) {
  if (node.is_leaf()) {
    return 0;
  }
  return 1 + std::max(ComputeNodeHeight(node.left()),
                      ComputeNodeHeight(node.right()));
}

/* Returns the height of BVH tree.
 Definition from graph theory:
 The **height** of a rooted tree is the height of its root.  That is, the height
 of a tree is the number of edges in a longest possible path, going away from
 the root, that starts at the root and ends at a leaf.
 (https://en.m.wikipedia.org/wiki/Glossary_of_graph_theory_terms#height)  */
template <class MeshType>
int ComputeTreeHeight(const BVH<MeshType>& bvh) {
  return ComputeNodeHeight(bvh.root_node());
}

template <class MeshType>
int CountNumNodes(const BVH<MeshType>& bvh) {
  const function<int(const BVHNode<MeshType>&)> traverse =
      [&traverse](const BVHNode<MeshType>& node) -> int {
        if (node.is_leaf()) {
          return 1;
        } else {
          int left_result = traverse(node.left());
          int right_result = traverse(node.right());
          return 1 + left_result + right_result;
        }
      };
  return traverse(bvh.root_node());
}

template <class MeshType>
int CountNumLeaves(const BVH<MeshType>& bvh) {
  const function<int(const BVHNode<MeshType>&)> traverse =
      [&traverse](const BVHNode<MeshType>& node) -> int {
        if (node.is_leaf()) {
          return 1;
        } else {
          int left_result = traverse(node.left());
          int right_result = traverse(node.right());
          return left_result + right_result;
        }
      };
  return traverse(bvh.root_node());
}

template <class MeshType>
bool TestEachLeaf(
    const BVH<MeshType>& bvh,
    const function<bool(const BVHNode<MeshType>&)>& test_function) {
  const function<bool(const BVHNode<MeshType>&)> traverse =
      [&traverse, &test_function](const BVHNode<MeshType>& node) -> bool {
        if (node.is_leaf()) {
          return test_function(node);
        } else {
          bool left_result = traverse(node.left());
          bool right_result = traverse(node.right());
          return left_result && right_result;
        }
      };
  return traverse(bvh.root_node());
}

template <class MeshType>
void VisitEachLeaf(const BVH<MeshType>& bvh,
                   const function<void(const BVHNode<MeshType>&)>& visitor) {
  const function<void(const BVHNode<MeshType>&)> traverse =
      [&traverse, &visitor](const BVHNode<MeshType>& node) {
        if (node.is_leaf()) {
          visitor(node);
        } else {
          traverse(node.left());
          traverse(node.right());
        }
      };
  return traverse(bvh.root_node());
}

template <class MeshTypeA, class MeshTypeB>
void VisitEachLeafPair(
    const BVH<MeshTypeA>& bvh_a, const BVH<MeshTypeB>& bvh_b,
    const function<void(const BVHNode<MeshTypeA>&, const BVHNode<MeshTypeB>&)>&
        visitor) {
  VisitEachLeaf<MeshTypeA>(
      bvh_a, [&bvh_b, &visitor](const BVHNode<MeshTypeA>& leaf_A) {
        VisitEachLeaf<MeshTypeB>(
            bvh_b, [&visitor, &leaf_A](const BVHNode<MeshTypeB>& leaf_B) {
              visitor(leaf_A, leaf_B);
            });
      });
}

class ObbTester : public ::testing::Test {
 public:
  static constexpr double kTolerance = Obb::kTolerance;
};

class BVHTester {
 public:
  BVHTester() = delete;

  template <class MeshType>
  static Vector3d ComputeCentroid(const MeshType& mesh,
                                  const typename MeshType::ElementIndex i) {
    return BVH<MeshType>::ComputeCentroid(mesh, i);
  }

  template <class MeshType>
  static Obb ComputeBoundingVolume(
      const MeshType& mesh,
      const typename std::vector<typename BVH<MeshType>::CentroidPair>::iterator
          start,
      const typename std::vector<typename BVH<MeshType>::CentroidPair>::iterator
          end,
      bool optimize) {
    return BVH<MeshType>::ComputeBoundingVolume(mesh, start, end, optimize);
  }
};

namespace {

GTEST_TEST(TestComputeOBB, SingleTri) {
  // The vertex positions are expressed in frame M of the mesh.
  std::vector<SurfaceVertex<double>> vertices_M;
  vertices_M.emplace_back(Vector3d(1., 0., 0.));
  vertices_M.emplace_back(Vector3d(0., 1., 0.));
  vertices_M.emplace_back(Vector3d(0., 0., 1.));
  std::vector<SurfaceFace> faces;
  faces.emplace_back(SurfaceVertexIndex(0), SurfaceVertexIndex(1),
                     SurfaceVertexIndex(2));
  SurfaceMesh<double> mesh_M(std::move(faces), std::move(vertices_M));

  // Only SurfaceIndex is relevant. The centroid of the triangle is not
  // relevant for this test.
  std::vector<std::pair<SurfaceFaceIndex, Vector3d>> tri_centroids;
  tri_centroids.emplace_back(0, Vector3d());

  const Obb obb = BVHTester::ComputeBoundingVolume<SurfaceMesh<double>>(
      mesh_M, tri_centroids.begin(), tri_centroids.end(), true);

  // Calculate the expected pose of the box expressed in frame M.
  // B is for the canonical frame of the box.
  const Vector3d Bx_M = Vector3d(1., -0.5, -0.5).normalized();
  const Vector3d By_M = Vector3d(0., 1., -1.).normalized();
  const Vector3d Bz_M = Vector3d(1., 1., 1.).normalized();
  const RotationMatrixd R_MB =
      RotationMatrixd::MakeFromOrthonormalColumns(Bx_M, By_M, Bz_M);
  // C is for the center of the box.
  const Vector3d p_MC(0.5, 0.25, 0.25);
  RigidTransformd expect_pose(R_MB, p_MC);

  EXPECT_TRUE(CompareMatrices(obb.pose().GetAsMatrix4(),
                              expect_pose.GetAsMatrix4(), 1e-15));

  // Calculate the expected half-width vector of the box.
  const Vector3d expect_half_width(std::sqrt(6.) / 4, 1. / std::sqrt(2.), 0.);
  EXPECT_TRUE(CompareMatrices(obb.half_width(), expect_half_width,
                              2.0 * ObbTester::kTolerance));

  const BVH<SurfaceMesh<double>> bvh(mesh_M);
}

GTEST_TEST(TestComputeOBB, Meshes) {
  // Test with a surface mesh.
  {
    // The sphere's mesh is so coarse that it is an octahedron.
    SurfaceMesh<double> mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);

    std::vector<std::pair<SurfaceFaceIndex, Vector3d>> tri_centroids;
    // Add all the elements from the octahedron to test multiple elements. The
    // centroid values are irrelevant for this test, but the bounding box should
    // encompass the whole sphere with a center of 0 and half width of 1.5.
    for (SurfaceFaceIndex i(0); i < mesh.num_elements(); ++i) {
      tri_centroids.emplace_back(i, Vector3d(0.5, 0.5, 0.5));
    }

    // Test without volume optimization.
    Obb obb = BVHTester::ComputeBoundingVolume<SurfaceMesh<double>>(
        mesh, tri_centroids.begin(), tri_centroids.end(), false);

    const Obb expect_obb(
        RigidTransformd(
            RotationMatrixd::MakeFromOrthonormalColumns(
                Vector3d::UnitZ(), Vector3d::UnitY(), -Vector3d::UnitX()),
            Vector3d::Zero()),
        Vector3d(1.5, 1.5, 1.5));

    EXPECT_TRUE(obb.Equal(expect_obb));

    const double expect_volume = 27.0;
    EXPECT_NEAR(obb.CalcVolume(), expect_volume, 1e-11);

    // Test with volume optimization.
    obb = BVHTester::ComputeBoundingVolume<SurfaceMesh<double>>(
        mesh, tri_centroids.begin(), tri_centroids.end(), true);

    const double expect_optimized_volume = 26.99614412891;
    EXPECT_NEAR(obb.CalcVolume(), expect_optimized_volume, 1e-11);
  }

  // Test with a volume mesh. As above, the centroids are still irrelevant. The
  // bounding box should encompass the whole ellipsoid with a center of 0 and
  // half width of 1, 2, 3.
  {
    VolumeMesh<double> volume_mesh = MakeEllipsoidVolumeMesh<double>(
        Ellipsoid(1., 2., 3.), 6, TessellationStrategy::kSingleInteriorVertex);
    std::vector<std::pair<VolumeElementIndex, Vector3d>> tet_centroids;
    for (VolumeElementIndex i(0); i < volume_mesh.num_elements(); ++i) {
      tet_centroids.emplace_back(i, Vector3d(0.5, 0.5, 0.5));
    }

    // Test without volume optimization.
    Obb obb = BVHTester::ComputeBoundingVolume<VolumeMesh<double>>(
        volume_mesh, tet_centroids.begin(), tet_centroids.end(), false);

    const Obb expect_obb(
        RigidTransformd(
            RotationMatrixd::MakeFromOrthonormalColumns(
                Vector3d::UnitZ(), -Vector3d::UnitY(), Vector3d::UnitX()),
            Vector3d::Zero()),
        Vector3d(3., 2., 1.));

    EXPECT_TRUE(obb.Equal(expect_obb));

    const double expect_volume = 48.0;
    EXPECT_NEAR(obb.CalcVolume(), expect_volume, 1e-11);

    // Test with volume optimization.
    obb = BVHTester::ComputeBoundingVolume<VolumeMesh<double>>(
        volume_mesh, tet_centroids.begin(), tet_centroids.end(), true);

    const double expect_optimized_volume = 41.31660113817;
    EXPECT_NEAR(obb.CalcVolume(), expect_optimized_volume, 1e-11);
  }
}

// Tests properties from building the bounding volume tree.
// Each node should have its OBB volume less than its parent's OBB volume.
// The structure of the binary tree should depend on the number of elements per
// leaf.
GTEST_TEST(BVH, TestBuildBVTree) {
  // This sphere's mesh is so coarse that it is actually an octahedron whose
  // eight faces are equilateral triangles.
  const SurfaceMesh<double> mesh =
      MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  ASSERT_EQ(mesh.num_elements(), 8);

  // The octahedron has 8 triangles, so we should end up with a perfect binary
  // tree of height 3 if each leaf has one triangle. (A perfect binary tree is
  // a binary tree in which all interior nodes have two children and all
  // leaves have the same depth or same level.
  // https://en.m.wikipedia.org/wiki/Binary_tree#perfect)
  // The following picture shows how the tree would look like. In the
  // picture, the number at each node is the number of elements belonging to
  // the subtree rooted at that node. (We do not store such numbers in the
  // tree; the numbers are for illustration only.)
  //
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  //          ／＼   ／＼   ／＼   ／＼
  //         1  1   1  1  1  1  1  1
  //
  // However, when we allow multiple elements per leaf, the tree becomes
  // shorter. These pictures show how the tree would look for hypothetical
  // values of the maximum number of elements per leaf.
  //
  // MeshTraits<SurfaceMesh<double>>::kMaxElementPerLeaf = 2 or 3
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  //
  //
  // MeshTraits<SurfaceMesh<double>>::kMaxElementPerLeaf = 4, 5, 6, or 7
  //                      8
  //                   ／   ＼
  //                 4        4
  //
  // If MeshTraits<SurfaceMesh<double>>::kMaxElementPerLeaf is changed, we
  // might have to change the expectations below. Right now it is assumed to
  // be 2 or 3, so the tree should look like the second picture above.
  //
  const BVH<SurfaceMesh<double>> bvh(mesh);
  EXPECT_EQ(CountNumNodes<SurfaceMesh<double>>(bvh), 7);
  EXPECT_EQ(CountNumLeaves<SurfaceMesh<double>>(bvh), 4);
  const int tree_height = ComputeTreeHeight<SurfaceMesh<double>>(bvh);
  EXPECT_EQ(tree_height, 2);

  // The depth of a node in a rooted tree is the number of edges in the path
  // from the root to the node.
  // https://en.m.wikipedia.org/wiki/Glossary_of_graph_theory_terms#depth
  // The maximum depth of a tree is the same as the height of the tree.
  const int max_depth = tree_height;
  const int num_elements = mesh.num_elements();
  std::set<SurfaceFaceIndex> element_indices;
  // Verify the structure of the perfect binary tree and check that:
  // 1. Each internal node's OBB has volume greater than that of each child.
  // 2. Leaf nodes have unique element indices in a valid range.
  std::function<void(const BVHNode<SurfaceMesh<double>>&, int)> check_tree =
      [&check_tree, &element_indices, num_elements, max_depth](
          const BVHNode<SurfaceMesh<double>>& node, int depth) {
        const int expect_height = max_depth - depth;
        EXPECT_EQ(ComputeNodeHeight(node), expect_height);
        if (depth < max_depth) {
          EXPECT_FALSE(node.is_leaf());
          const double node_volume = node.obb().CalcVolume();
          EXPECT_LT(node.left().obb().CalcVolume(), node_volume);
          EXPECT_LT(node.right().obb().CalcVolume(), node_volume);
          check_tree(node.left(), depth + 1);
          check_tree(node.right(), depth + 1);
        } else {
          EXPECT_TRUE(node.is_leaf());
          for (int i = 0; i < node.num_element_index(); ++i) {
            EXPECT_GE(node.element_index(i), 0);
            EXPECT_LT(node.element_index(i), num_elements);
            EXPECT_EQ(element_indices.count(node.element_index(i)), 0);
            element_indices.insert(node.element_index(i));
          }
        }
      };
  check_tree(bvh.root_node(), 0);
  // Check that we found a leaf node for all elements.
  EXPECT_EQ(element_indices.size(), num_elements);
}

class BVHTest : public ::testing::Test {
 public:
  BVHTest()
      : ::testing::Test(),
        mesh_(MakeSphereSurfaceMesh<double>(Sphere(1.5), 3)),
        bvh_(BVH<SurfaceMesh<double>>(mesh_)) {}

  void SetUp() override {
    // Verify that the tree is a perfect binary tree like this:
    //
    //                      8
    //                   ／   ＼
    //                 4        4
    //               ／＼       ／＼
    //            2     2     2     2
    //
    ASSERT_EQ(mesh_.num_elements(), 8);
    ASSERT_EQ(CountNumNodes(bvh_), 7);
    ASSERT_EQ(CountNumLeaves(bvh_), 4);
    ASSERT_EQ(ComputeTreeHeight(bvh_), 2);
    ASSERT_TRUE(TestEachLeaf<SurfaceMesh<double>>(
        bvh_, [](const BVHNode<SurfaceMesh<double>>& leaf) -> bool {
          return leaf.num_element_index() == 2;
        }));
  }

 protected:
  const SurfaceMesh<double> mesh_;
  const BVH<SurfaceMesh<double>> bvh_;
};

// Tests copy constructor.
TEST_F(BVHTest, TestCopy) {
  // Copy constructor.
  BVH<SurfaceMesh<double>> bvh_copy(bvh_);

  // Confirm that it's a deep copy.
  std::function<void(const BVHNode<SurfaceMesh<double>>&,
                     const BVHNode<SurfaceMesh<double>>&)>
      check_copy = [&check_copy](const BVHNode<SurfaceMesh<double>>& orig,
                                 const BVHNode<SurfaceMesh<double>>& copy) {
        EXPECT_NE(&orig, &copy);
        if (orig.is_leaf()) {
          ASSERT_TRUE(copy.is_leaf());
          EXPECT_TRUE(orig.EqualLeaf(copy));
        } else {
          check_copy(orig.left(), copy.left());
          check_copy(orig.right(), copy.right());
        }
      };
  check_copy(bvh_.root_node(), bvh_copy.root_node());
}

// Tests colliding while traversing through the bvh trees when there is no
// overlap.
TEST_F(BVHTest, TestCollideWhenNoOverlap) {
  // The two trees are completely separate so no bounding volumes overlap and
  // all pairs should be culled. The resulting vector should thus be empty.
  auto separate_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  RigidTransformd X_WV{Vector3d{4, 4, 4}};
  BVH<SurfaceMesh<double>> separate(separate_mesh);
  std::vector<std::pair<SurfaceFaceIndex, SurfaceFaceIndex>> pairs =
      bvh_.GetCollisionCandidates(separate, X_WV);
  EXPECT_EQ(pairs.size(), 0);
}

// Tests colliding while traversing through the bvh trees when they overlap.
// We want to ensure that we covered the 4 cases of branch and leaf
// comparisons, i.e:
//  1. branch : branch
//  2. branch : leaf
//  3. leaf : branch
//  4. leaf : leaf
GTEST_TEST(BVH, TestCollideWhenOverlap) {
  // We use two spheres of the same radius 1.5. One sphere has its frame
  // identical to World. Another sphere has its pose in World as 3 unit
  // translation in Wx direction. The two spheres will have one common point C
  // with p_WC = (3,0,0).
  const auto coarse_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  ASSERT_EQ(coarse_mesh.num_elements(), 8);
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  const BVH<SurfaceMesh<double>> coarse_bvh(coarse_mesh);
  ASSERT_EQ(CountNumNodes(coarse_bvh), 7);
  ASSERT_EQ(CountNumLeaves(coarse_bvh), 4);
  ASSERT_EQ(ComputeTreeHeight(coarse_bvh), 2);
  ASSERT_TRUE(TestEachLeaf<SurfaceMesh<double>>(
      coarse_bvh, [](const BVHNode<SurfaceMesh<double>>& leaf) -> bool {
        return leaf.num_element_index() == 2;
      }));

  // Create a higher resolution mesh so we have a different number of elements
  // across the bvh trees, i.e. 8 in our coarse octahedron and 32 here. We
  // place the meshes such that they are touching at one corner, so the
  // traversal will reach end leaf-leaf cases (4.) since there are potentially
  // colliding pairs. Since the trees have different heights, the traversal
  // will cover branch-leaf cases (2.) on its way. Swapping the order then
  // catches the opposing leaf-branch cases (3.). Since the trees have multiple
  // depths, then at higher levels, for example at the root, the branch-branch
  // cases (1.) will be covered.
  //
  //                                32
  //                             ／
  //                           16
  //                        ／
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  //
  const auto fine_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 2);
  ASSERT_EQ(fine_mesh.num_elements(), 32);
  const BVH<SurfaceMesh<double>> fine_bvh(fine_mesh);
  ASSERT_EQ(CountNumNodes(fine_bvh), 31);
  ASSERT_EQ(CountNumLeaves(fine_bvh), 16);
  ASSERT_EQ(ComputeTreeHeight(fine_bvh), 4);
  ASSERT_TRUE(TestEachLeaf<SurfaceMesh<double>>(
      fine_bvh, [](const BVHNode<SurfaceMesh<double>>& leaf) -> bool {
        return leaf.num_element_index() == 2;
      }));

  const RigidTransformd X_WV = RigidTransformd{Vector3d{3, 0, 0}};

  // Perform all-to-all checks between all leaves from the first tree and
  // all leaves from the second tree.
  int num_overlap_obbs = 0;
  unordered_set<const BVHNode<SurfaceMesh<double>>*> coarse_overlap;
  unordered_set<const BVHNode<SurfaceMesh<double>>*> fine_overlap;
  VisitEachLeafPair<SurfaceMesh<double>, SurfaceMesh<double>>(
      coarse_bvh, fine_bvh,
      [&X_WV, &num_overlap_obbs, &coarse_overlap, &fine_overlap](
          const BVHNode<SurfaceMesh<double>>& leaf_A,
          const BVHNode<SurfaceMesh<double>>& leaf_B) {
        if (Obb::HasOverlap(leaf_A.obb(), leaf_B.obb(), X_WV)) {
          ++num_overlap_obbs;
          coarse_overlap.insert(&leaf_A);
          fine_overlap.insert(&leaf_B);
        }
      });
  // TODO(DamrongGuoy): Investigate the reason for these three assertions.
  //  There are 25 pairs of overlapping OBBs from the leaves of the two trees.
  //  (Previously for kMaxElementPerLeaf = 1, there were 16 such pairs.) There
  //  are 4 leaves from the coarse tree participating, but there are 8 leaves
  //  from the fine tree participating.
  ASSERT_EQ(num_overlap_obbs, 25);
  ASSERT_EQ(coarse_overlap.size(), 4);
  ASSERT_EQ(fine_overlap.size(), 8);

  int num_triangle_pairs =
      coarse_bvh.GetCollisionCandidates(fine_bvh, X_WV).size();
  // There are 25 pairs of overlapping OBBs from the leaves of the two trees.
  // Each pair of leaves give 4 pairs of triangles. Totally there are
  // 25 x 4 = 100 pairs of triangles.
  EXPECT_EQ(num_triangle_pairs, 100);
  num_triangle_pairs = fine_bvh.GetCollisionCandidates(coarse_bvh, X_WV).size();
  EXPECT_EQ(num_triangle_pairs, 100);
  WriteBVHToVtk("debug_coarse_bvh.vtk", coarse_bvh, "debug coarse bvh");
  WriteSurfaceMeshToVtk("debug_coarse_mesh.vtk", coarse_mesh,
                        "debug coarse mesh");
  WriteBVHToVtk("debug_fine_bvh.vtk", fine_bvh, "debug fine bvh");
  WriteSurfaceMeshToVtk("debug_fine_mesh.vtk", fine_mesh,
                        "debug fine mesh");
}

// Tests colliding while traversing through the bvh trees but with early exit.
// We want to ensure that the trees are not fully traversed. One way to test
// this is to count towards a limit as the condition for the exit. If we
// specify a limit that is less than the number of potentially colliding pairs,
// we can verify that the traversal has exited since our result can be no more
// than this limit.
TEST_F(BVHTest, TestCollideEarlyExit) {
  int count{0};
  int limit{1};
  // This callback should only be run as many times as the specified limit
  // before the early exit kicks in.
  auto callback = [&count, &limit](SurfaceFaceIndex a,
                                   SurfaceFaceIndex b) -> BVHCallbackResult {
    ++count;
    return count < limit ? BVHCallbackResult::Continue
                         : BVHCallbackResult::Terminate;
  };
  // Since we're colliding bvh against itself there should be up to n^2
  // potentially colliding pairs, but we max out at our limit of 1.
  bvh_.Collide(bvh_, math::RigidTransformd::Identity(), callback);
  EXPECT_EQ(count, 1);

  count = 0;
  limit = 5;
  // Updating the limit to 5 should get further in the traversal with a result
  // of 5 pairs.
  bvh_.Collide(bvh_, math::RigidTransformd::Identity(), callback);
  EXPECT_EQ(count, 5);
}

// Tests colliding the bvh trees with different mesh types, i.e. mixing tris
// and tets by colliding surface and volume meshes.
GTEST_TEST(BVH, TestCollideSurfaceVolume) {

  auto volume_mesh = MakeEllipsoidVolumeMesh<double>(
      Ellipsoid(1.5, 2., 3.), 6, TessellationStrategy::kSingleInteriorVertex);
  ASSERT_EQ(volume_mesh.num_elements(), 8);
  //
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  //          ／＼   ／＼   ／＼   ／＼
  //         1  1   1  1  1  1  1  1
  //
  BVH<VolumeMesh<double>> tet_bvh(volume_mesh);
  ASSERT_EQ(CountNumNodes(tet_bvh), 15);
  ASSERT_EQ(CountNumLeaves(tet_bvh), 8);
  ASSERT_EQ(ComputeTreeHeight(tet_bvh), 3);
  ASSERT_TRUE(TestEachLeaf<VolumeMesh<double>>(
      tet_bvh, [](const BVHNode<VolumeMesh<double>>& leaf) -> bool {
        return leaf.num_element_index() == 1;
      }));

  auto surface_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  ASSERT_EQ(surface_mesh.num_elements(), 8);
  RigidTransformd X_WV{Vector3d{3, 0, 0}};
  //
  //                      8
  //                   ／   ＼
  //                 4        4
  //               ／＼       ／＼
  //            2     2     2     2
  //
  BVH<SurfaceMesh<double>> tri_bvh(surface_mesh);
  ASSERT_EQ(CountNumNodes(tri_bvh), 7);
  ASSERT_EQ(CountNumLeaves(tri_bvh), 4);
  ASSERT_EQ(ComputeTreeHeight(tri_bvh), 2);
  ASSERT_TRUE(TestEachLeaf<SurfaceMesh<double>>(
      tri_bvh, [](const BVHNode<SurfaceMesh<double>>& leaf) -> bool {
        return leaf.num_element_index() == 2;
      }));

  auto pairs = tet_bvh.GetCollisionCandidates(tri_bvh, X_WV);
  // The two octahedrons are tangentially touching along the X-axis. There
  // should be 4 leaves each that are colliding, resulting in 4x4 = 16 pairs
  // of leaves. Each pair of leaves give two tetrahedron-triangle pairs
  // because tet_bvh's leaf has one element and tri_bvh's leaf has two
  // elements as verified above. Totally there are 16x2 = 32 candidate pairs.
  EXPECT_EQ(pairs.size(), 32);
}

// Tests computing the centroid of an element.
GTEST_TEST(BVHTestNoFixture, TestComputeCentroid) {
  // Set resolution at double so that we get the coarsest mesh of 8 elements.
  auto surface_mesh =
      MakeEllipsoidSurfaceMesh<double>(Ellipsoid(1., 2., 3.), 6);
  Vector3d centroid = BVHTester::ComputeCentroid<SurfaceMesh<double>>(
      surface_mesh, SurfaceFaceIndex(0));
  // The first face of our octahedron is a triangle with vertices at 1, 2, and
  // 3 along each respective axis, so its centroid should average out to 1/3,
  // 2/3, and 3/3.
  EXPECT_TRUE(CompareMatrices(centroid, Vector3d(1. / 3., 2. / 3., 1.)));

  auto volume_mesh = MakeEllipsoidVolumeMesh<double>(
      Ellipsoid(1., 2., 3.), 6, TessellationStrategy::kSingleInteriorVertex);
  centroid = BVHTester::ComputeCentroid<VolumeMesh<double>>(
      volume_mesh, VolumeElementIndex(0));
  // The first face of our octahedron is a tet with vertices at 1, 2, and 3
  // along each respective axis and the origin 0, so its centroid should
  // average out to 1/4, 2/4, and 3/4.
  EXPECT_TRUE(CompareMatrices(centroid, Vector3d(0.25, 0.5, 0.75)));
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
