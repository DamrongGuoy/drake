#include "drake/geometry/proximity/bvh.h"

#include <memory>

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
  WriteBVHToVtk("single_triangle_obb_bvh.vtk", bvh,
                "OBB Tree of Single Triangle");
  WriteSurfaceMeshToVtk("single_triangle_surface_mesh.vtk", mesh_M,
                        "SurfaceMesh of Single Triangle");
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

    WriteBVHToVtk("octahedron_obb_bvh.vtk", BVH<SurfaceMesh<double>>(mesh),
                  "OBB Tree of SurfaceMesh of Octahedron");
    WriteSurfaceMeshToVtk("octahedron_surface_mesh.vtk", mesh,
                          "SurfaceMesh of Octahedron");
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

    WriteBVHToVtk("elliposid_tetrahedra_obb_bvh_optimize.vtk",
                  BVH<VolumeMesh<double>>(volume_mesh),
                  "OBB Tree of Tetrahedra of Ellipsoid");
    WriteVolumeMeshToVtk("elliposid_volume_mesh.vtk", volume_mesh,
                         "VolumeMesh of Ellipsoid");
  }
}

class BVHTest : public ::testing::Test {
 public:
  BVHTest()
      : ::testing::Test(),
        mesh_(MakeSphereSurfaceMesh<double>(Sphere(1.5), 3)),
        bvh_(BVH<SurfaceMesh<double>>(mesh_)) {}

 protected:
  SurfaceMesh<double> mesh_;
  BVH<SurfaceMesh<double>> bvh_;
};

// Tests properties from building the bounding volume tree.
TEST_F(BVHTest, TestBuildBVTree) {
  // Since it's a binary tree with a single element at each leaf node, the tree
  // depth can be found using 2^d = num_elements. The octahedron has 8 elements
  // so we should end up with a balanced tree of depth 3 where each level's
  // volume is less than its parent's volume.
  const BVHNode<SurfaceMesh<double>>& bv_tree = bvh_.root_node();
  const int num_elements = mesh_.num_elements();
  std::set<SurfaceFaceIndex> element_indices;
  std::function<void(const BVHNode<SurfaceMesh<double>>&, int)> check_node;
  check_node = [&check_node, &element_indices, num_elements](
                   const BVHNode<SurfaceMesh<double>>& node, int depth) {
    if (depth < 3) {
      const double node_volume = node.obb().CalcVolume();
      auto check_child = [&check_node, &node_volume,
                          &depth](const BVHNode<SurfaceMesh<double>>& child) {
        const double child_volume = child.obb().CalcVolume();
        EXPECT_LT(child_volume, node_volume);
        check_node(child, depth + 1);
      };
      check_child(node.left());
      check_child(node.right());
    } else {
      // At depth 3, we should reach the leaf node with a valid and unique
      // element instead of more branches.
      EXPECT_GE(node.element_index(), 0);
      EXPECT_LT(node.element_index(), num_elements);
      EXPECT_EQ(element_indices.count(node.element_index()), 0);
      element_indices.insert(node.element_index());
    }
  };
  check_node(bv_tree, 0);
  // Check that we found a leaf node for all elements.
  EXPECT_EQ(element_indices.size(), num_elements);
}

// Tests copy constructor.
TEST_F(BVHTest, TestCopy) {
  // Copy constructor.
  BVH<SurfaceMesh<double>> bvh_copy(bvh_);

  // Confirm that it's a deep copy.
  std::function<void(const BVHNode<SurfaceMesh<double>>&,
                     const BVHNode<SurfaceMesh<double>>&)>
      check_copy;
  check_copy = [&check_copy](const BVHNode<SurfaceMesh<double>>& orig,
                             const BVHNode<SurfaceMesh<double>>& copy) {
    EXPECT_NE(&orig, &copy);
    if (orig.is_leaf()) {
      EXPECT_EQ(orig.element_index(), copy.element_index());
    } else {
      check_copy(orig.left(), copy.left());
      check_copy(orig.right(), copy.right());
    }
  };
  check_copy(bvh_.root_node(), bvh_copy.root_node());
}

// Tests colliding while traversing through the bvh trees. We want to ensure
// that the case of no overlap is covered as well as the 4 cases of branch and
// leaf comparisons, i.e:
//  1. branch : branch
//  2. branch : leaf
//  3. leaf : branch
//  4. leaf : leaf
TEST_F(BVHTest, TestCollide) {
  // The two trees are completely separate so no bounding volumes overlap and
  // all pairs should be culled. The resulting vector should thus be empty.
  auto separate_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  RigidTransformd X_WV{Vector3d{4, 4, 4}};
  BVH<SurfaceMesh<double>> separate(separate_mesh);
  std::vector<std::pair<SurfaceFaceIndex, SurfaceFaceIndex>> pairs =
      bvh_.GetCollisionCandidates(separate, X_WV);
  EXPECT_EQ(pairs.size(), 0);

  // Create a higher resolution mesh so we have a different number of elements
  // across the bvh trees, i.e. 8 in our coarse octahedron and 32 here. We
  // place the meshes such that they are touching at one corner, so the
  // traversal will reach end leaf-leaf cases (4.) since there are potentially
  // colliding pairs. Since the trees have different depths, the traversal
  // will cover branch-leaf cases (2.) on its way. Swapping the order then
  // catches the opposing leaf-branch cases (3.). Since the trees have multiple
  // depths then at higher levels, for example at the root, the branch-branch
  // cases (1.) will be covered.
  auto tangent_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 2);
  X_WV = RigidTransformd{Vector3d{3, 0, 0}};
  BVH<SurfaceMesh<double>> tangent(tangent_mesh);
  pairs = bvh_.GetCollisionCandidates(tangent, X_WV);
  EXPECT_EQ(pairs.size(), 16);
  pairs = tangent.GetCollisionCandidates(bvh_, X_WV);
  EXPECT_EQ(pairs.size(), 16);
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
TEST_F(BVHTest, TestCollideSurfaceVolume) {
  // The two octahedrons are tangentially touching along the X-axis, so there
  // should be 4 elements each that are colliding, resulting in 4^2 = 16.
  auto volume_mesh = MakeEllipsoidVolumeMesh<double>(
      Ellipsoid(1.5, 2., 3.), 6, TessellationStrategy::kSingleInteriorVertex);
  BVH<VolumeMesh<double>> tet_bvh(volume_mesh);

  auto surface_mesh = MakeSphereSurfaceMesh<double>(Sphere(1.5), 3);
  RigidTransformd X_WV{Vector3d{3, 0, 0}};
  BVH<SurfaceMesh<double>> tri_bvh(surface_mesh);

  auto pairs = tet_bvh.GetCollisionCandidates(tri_bvh, X_WV);
  EXPECT_EQ(pairs.size(), 16);
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
