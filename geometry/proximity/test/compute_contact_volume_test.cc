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
      :  // Data for the box.
        box_mesh0_M_(MakeBoxVolumeMeshWithMa<double>(box_)),
        box_bvh0_M_(box_mesh0_M_),
        box_boundary_M_(box_mesh0_M_),
        box_hydro_compliant_geometry_(hydroelastic::SoftMesh(
            std::make_unique<VolumeMesh<double>>(box_mesh0_M_),
            std::make_unique<VolumeMeshFieldLinear<double, double>>(
                MakeBoxPressureField(box_, &box_mesh0_M_,
                                     box_hydro_modulus_)))),
        // Data for the octahedron.
        // Get a mesh of an octahedron from a sphere specification by
        // specifying very coarse resolution hint.
        octahedron_mesh1_N_(MakeSphereVolumeMesh<double>(
            sphere_, 10 * sphere_.radius(),
            TessellationStrategy::kSingleInteriorVertex)),
        octahedron_bvh1_N_(octahedron_mesh1_N_),
        octahedron_boundary_N_(octahedron_mesh1_N_),
        octahedron_hydro_compliant_geometry_(hydroelastic::SoftMesh(
            std::make_unique<VolumeMesh<double>>(octahedron_mesh1_N_),
            std::make_unique<VolumeMeshFieldLinear<double, double>>(
                MakeSpherePressureField(sphere_, &octahedron_mesh1_N_,
                                        octahedron_hydro_modulus_)))) {}

 protected:
  void SetUp() override {
    DRAKE_DEMAND(octahedron_mesh1_N_.num_elements() == 8);
  }

  // Geometry 0 and its field.
  const Box box_{0.06, 0.10, 0.14};  // 6cm-thick compliant pad.
  const double box_hydro_modulus_{1e7};
  const VolumeMesh<double> box_mesh0_M_;
  const Bvh<Obb, VolumeMesh<double>> box_bvh0_M_;
  const MeshDistanceBoundary box_boundary_M_;
  const hydroelastic::SoftGeometry box_hydro_compliant_geometry_;

  // Geometry 1 and its field.
  const Sphere sphere_{0.03};  // 3cm-radius (6cm-diameter) finger tip.
  const double octahedron_hydro_modulus_{1e7};
  const VolumeMesh<double> octahedron_mesh1_N_;
  const Bvh<Obb, VolumeMesh<double>> octahedron_bvh1_N_;
  const MeshDistanceBoundary octahedron_boundary_N_;
  const hydroelastic::SoftGeometry octahedron_hydro_compliant_geometry_;
};

TEST_F(ComputeContactVolumeTest, ComputeContactVolume) {
  GeometryId first_id = GeometryId::get_new_id();
  GeometryId second_id = GeometryId::get_new_id();
  const RigidTransformd X_WM = RigidTransformd::Identity();
  const RigidTransformd X_WN(0.03 * Vector3d::UnitX());

  WriteVolumeMeshToVtk("first_box_mesh0_M.vtk", box_mesh0_M_, "box_mesh0_M_");
  WriteVolumeMeshToVtk("second_octahedron_mesh1_N.vtk", octahedron_mesh1_N_,
                       "octahedron_mesh1_N_");

  std::pair<std::unique_ptr<ContactSurface<double>>,
            std::unique_ptr<ContactSurface<double>>>
      pair = ComputeContactVolume(first_id, box_boundary_M_, X_WM, second_id,
                                  octahedron_boundary_N_, X_WN,
                                  // Extra parameters because we do not have
                                  // triangle-triangle mesh clipping yet.
                                  box_hydro_compliant_geometry_,
                                  octahedron_hydro_compliant_geometry_);

  {
    SCOPED_TRACE("First contact boundary.");
    ASSERT_NE(pair.first.get(), nullptr);
    const ContactSurface<double>& s1 = *pair.first.get();
    ASSERT_GT(s1.num_vertices(), 0);
    ASSERT_FALSE(s1.is_triangle());
    const std::vector<double>& distances = s1.poly_e_MN().values();
    EXPECT_NEAR(*std::min_element(distances.begin(), distances.end()),
                -0.01 * std::sqrt(3.0), 1e-14);
    EXPECT_NEAR(*std::max_element(distances.begin(), distances.end()), 0,
                1e-14);
    // We can't write polygonal data to VTK yet. We only have it for
    // triangles.
    // WriteTriangleSurfaceMeshFieldLinearToVtk("first_boundary.vtk",
    //                                          "distance",
    //                                          s1.tri_e_MN(),
    //                                          "compute_contact_volume");
  }
  {
    SCOPED_TRACE("Second contact boundary.");
    ASSERT_NE(pair.second.get(), nullptr);
    const ContactSurface<double>& s2 = *pair.second.get();
    ASSERT_GT(s2.num_vertices(), 0);
    ASSERT_FALSE(s2.is_triangle());
    const std::vector<double>& distances = s2.poly_e_MN().values();
    EXPECT_NEAR(*std::min_element(distances.begin(), distances.end()), -0.03,
                1e-14);
    EXPECT_NEAR(*std::max_element(distances.begin(), distances.end()), 0,
                1e-14);
    // We can't write polygonal data to VTK yet. We only have it for
    // triangles.
    // WriteTriangleSurfaceMeshFieldLinearToVtk("second_boundary.vtk",
    //                                          "distance", s2.tri_e_MN(),
    //                                          "compute_contact_volume");
  }
}

TEST_F(ComputeContactVolumeTest, ComputeContactVolume_NoContact) {
  GeometryId first_id = GeometryId::get_new_id();
  GeometryId second_id = GeometryId::get_new_id();
  const RigidTransformd X_WM = RigidTransformd::Identity();
  // N is 100 meters far away from M.
  const RigidTransformd X_WN(100 * Vector3d::UnitX());

  std::pair<std::unique_ptr<ContactSurface<double>>,
            std::unique_ptr<ContactSurface<double>>>
      pair = ComputeContactVolume(first_id, box_boundary_M_, X_WM, second_id,
                                  octahedron_boundary_N_, X_WN,
                                  // Extra parameters because we do not have
                                  // triangle-triangle mesh clipping yet.
                                  box_hydro_compliant_geometry_,
                                  octahedron_hydro_compliant_geometry_);
  EXPECT_EQ(pair.first, nullptr);
  EXPECT_EQ(pair.second, nullptr);
}

TEST_F(ComputeContactVolumeTest, InPairOrderByGeometryIds) {
  GeometryId first_id = GeometryId::get_new_id();
  GeometryId second_id = GeometryId::get_new_id();
  const RigidTransformd X_WM = RigidTransformd::Identity();
  const RigidTransformd X_WN(0.03 * Vector3d::UnitX());

  {
    SCOPED_TRACE("ComputeContactVolume(first_id, second_id)");
    std::pair<std::unique_ptr<ContactSurface<double>>,
              std::unique_ptr<ContactSurface<double>>>
        pair = ComputeContactVolume(first_id, box_boundary_M_, X_WM, second_id,
                                    octahedron_boundary_N_, X_WN,
                                    // Extra parameters because we do not have
                                    // triangle-triangle mesh clipping yet.
                                    box_hydro_compliant_geometry_,
                                    octahedron_hydro_compliant_geometry_);
    // Each contact surface always has id_M(), id_N() in increasing order.
    EXPECT_EQ(pair.first->id_M(), first_id);
    EXPECT_EQ(pair.first->id_N(), second_id);
    EXPECT_EQ(pair.second->id_M(), first_id);
    EXPECT_EQ(pair.second->id_N(), second_id);
    // First contact surface is on the first_id's geometry, so the contact
    // surface has no grad_E_M.
    EXPECT_FALSE(pair.first->HasGradE_M());
    EXPECT_TRUE(pair.first->HasGradE_N());
    // Second contact surface is on the second_id's geometry, so the contact
    // surface has no grad_E_N.
    EXPECT_FALSE(pair.second->HasGradE_N());
    EXPECT_TRUE(pair.second->HasGradE_M());
  }

  {
    SCOPED_TRACE("Switch Id's.");
    std::pair<std::unique_ptr<ContactSurface<double>>,
              std::unique_ptr<ContactSurface<double>>>
        pair = ComputeContactVolume(second_id, box_boundary_M_, X_WM, first_id,
                                    octahedron_boundary_N_, X_WN,
                                    // Extra parameters because we do not have
                                    // triangle-triangle mesh clipping yet.
                                    box_hydro_compliant_geometry_,
                                    octahedron_hydro_compliant_geometry_);
    // Each contact surface always has id_M(), id_N() in increasing order.
    EXPECT_EQ(pair.first->id_M(), first_id);
    EXPECT_EQ(pair.first->id_N(), second_id);
    EXPECT_EQ(pair.second->id_M(), first_id);
    EXPECT_EQ(pair.second->id_N(), second_id);
    // First contact surface is on the first_id's geometry, so the contact
    // surface has no grad_E_M.
    EXPECT_FALSE(pair.first->HasGradE_M());
    EXPECT_TRUE(pair.first->HasGradE_N());
    // Second contact surface is on the second_id's geometry, so the contact
    // surface has no grad_E_N.
    EXPECT_FALSE(pair.second->HasGradE_N());
    EXPECT_TRUE(pair.second->HasGradE_M());
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
