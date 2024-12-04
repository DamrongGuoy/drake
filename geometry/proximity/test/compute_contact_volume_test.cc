#include "drake/geometry/proximity/compute_contact_volume.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity/contact_surface_utility.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/make_sphere_field.h"
#include "drake/geometry/proximity/make_sphere_mesh.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::AngleAxisd;
using Eigen::Vector3d;
using math::RigidTransform;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrix;
using math::RotationMatrixd;
using std::make_unique;
using std::pair;
using std::unique_ptr;
using std::vector;

class ComputeContactVolumeTest : public ::testing::Test {
 public:
  ComputeContactVolumeTest()
      : box_mesh0_M_(MakeBoxVolumeMeshWithMa<double>(box_)),
        box_bvh0_M_(box_mesh0_M_),
        // Get a mesh of an octahedron from a sphere specification by
        // specifying very coarse resolution hint.
        octahedron_mesh1_N_(MakeSphereVolumeMesh<double>(
            sphere_, 10 * sphere_.radius(),
            TessellationStrategy::kSingleInteriorVertex)),
        octahedron_bvh1_N_(octahedron_mesh1_N_) {}

 protected:
  void SetUp() override {
    DRAKE_DEMAND(octahedron_mesh1_N_.num_elements() == 8);
  }

  // Geometry 0 and its field.
  const Box box_{0.06, 0.10, 0.14};  // 6cm-thick compliant pad.
  const VolumeMesh<double> box_mesh0_M_;
  const Bvh<Obb, VolumeMesh<double>> box_bvh0_M_;

  // Geometry 1 and its field.
  const Sphere sphere_{0.03};  // 3cm-radius (6cm-diameter) finger tip.
  const VolumeMesh<double> octahedron_mesh1_N_;
  const Bvh<Obb, VolumeMesh<double>> octahedron_bvh1_N_;
};

TEST_F(ComputeContactVolumeTest, ComputeContactVolume) {
  GeometryId first_id = GeometryId::get_new_id();
  GeometryId second_id = GeometryId::get_new_id();
  const RigidTransformd X_WM = RigidTransformd::Identity();
  const RigidTransformd X_WN(0.03 * Vector3d::UnitX());

  {
    SCOPED_TRACE("Request triangles.");
    WriteVolumeMeshToVtk("first_box_mesh0_M.vtk", box_mesh0_M_, "box_mesh0_M_");
    WriteVolumeMeshToVtk("second_octahedron_mesh1_N.vtk", octahedron_mesh1_N_,
                         "octahedron_mesh1_N_");

    std::pair<std::unique_ptr<ContactSurface<double>>,
              std::unique_ptr<ContactSurface<double>>>
        pair = ComputeContactVolume(
            first_id, box_mesh0_M_, box_bvh0_M_, X_WM, second_id,
            octahedron_mesh1_N_, octahedron_bvh1_N_, X_WN,
            HydroelasticContactRepresentation::kTriangle);

    {
      SCOPED_TRACE("First contact boundary.");
      ASSERT_NE(pair.first.get(), nullptr);
      const ContactSurface<double>& s1 = *pair.first.get();
      ASSERT_GT(s1.num_vertices(), 0);
      ASSERT_TRUE(s1.is_triangle());
      const std::vector<double>& distances = s1.tri_e_MN().values();
      EXPECT_NEAR(*std::min_element(distances.begin(), distances.end()), 0,
                  1e-14);
      EXPECT_NEAR(*std::max_element(distances.begin(), distances.end()), 0.03,
                  1e-14);
      WriteTriangleSurfaceMeshFieldLinearToVtk("first_boundary.vtk", "distance",
                                               s1.tri_e_MN(),
                                               "compute_contact_volume");
    }
    {
      SCOPED_TRACE("Second contact boundary.");
      ASSERT_NE(pair.second.get(), nullptr);
      const ContactSurface<double>& s2 = *pair.second.get();
      ASSERT_GT(s2.num_vertices(), 0);
      ASSERT_TRUE(s2.is_triangle());
      const std::vector<double>& distances = s2.tri_e_MN().values();
      EXPECT_NEAR(*std::min_element(distances.begin(), distances.end()), 0,
                  1e-14);
      EXPECT_NEAR(*std::max_element(distances.begin(), distances.end()),
                  0.01 * std::sqrt(3.0), 1e-14);
      WriteTriangleSurfaceMeshFieldLinearToVtk("second_boundary.vtk",
                                               "distance", s2.tri_e_MN(),
                                               "compute_contact_volume");
    }
  }
  {
    SCOPED_TRACE("Request polygons.");
    std::pair<std::unique_ptr<ContactSurface<double>>,
              std::unique_ptr<ContactSurface<double>>>
        pair = ComputeContactVolume(
            first_id, box_mesh0_M_, box_bvh0_M_, X_WM, second_id,
            octahedron_mesh1_N_, octahedron_bvh1_N_, X_WN,
            HydroelasticContactRepresentation::kPolygon);
    ASSERT_NE(pair.first.get(), nullptr);
    ASSERT_NE(pair.second.get(), nullptr);
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
