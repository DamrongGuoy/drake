#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/calc_signed_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/optimize_sdfield.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;

GTEST_TEST(EmPressSignedDistanceField, GenerateFromInputSurface) {
  TriangleSurfaceMesh<double> input_mesh_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(input_mesh_M.num_vertices(), 486);
  EXPECT_EQ(input_mesh_M.num_triangles(), 968);
  const Aabb fitted_box_M = CalcBoundingBox(input_mesh_M);
  EXPECT_TRUE(CompareMatrices(fitted_box_M.center(),
                              Vector3d{-0.000021, -0.000189, 0.040183}, 1e-6));
  EXPECT_TRUE(CompareMatrices(fitted_box_M.half_width(),
                              Vector3d{0.040288, 0.040262, 0.040388}, 1e-6));

  const auto [mesh_EmPress_M, sdfield_EmPress_M] =
      MakeEmPressSDField(input_mesh_M, 0.02);  // grid_resolution,

  EXPECT_EQ(mesh_EmPress_M->num_vertices(), 167);
  EXPECT_EQ(mesh_EmPress_M->num_elements(), 568);
  WriteVolumeMeshFieldLinearToVtk("yellow_pepper_EmPress_sdfield.vtk",
                                  "SignedDistance(meters)", *sdfield_EmPress_M,
                                  "EmbeddedSignedDistanceField");
}

GTEST_TEST(CalcRMSErrorOfSDFieldTest, RootMeanSquaredError) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdfield_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};

  const TriangleSurfaceMesh<double> original_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(original_M.num_vertices(), 486);
  EXPECT_EQ(original_M.num_triangles(), 968);

  const double rms_error = CalcRMSErrorOfSDField(sdfield_M, original_M);

  // About 1.2mm RMS error.
  EXPECT_NEAR(rms_error, 0.001296, 1e-6);
}

GTEST_TEST(SDFieldOptimizerTest, Barebone) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdfield_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};

  const TriangleSurfaceMesh<double> original_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(original_M.num_vertices(), 486);
  EXPECT_EQ(original_M.num_triangles(), 968);

  SDFieldOptimizer optimizer(sdfield_M, original_M);
  struct SDFieldOptimizer::RelaxationParameters parameters{
      .alpha_exterior = 0.03,           // dimensionless
      .alpha = 0.3,                     // dimensionless
      .beta = 0.3,                      // dimensionless
      .target_boundary_distance = 1e-3  // meters
  };
  VolumeMesh<double> optimized_mesh = optimizer.OptimizeVertex(parameters);
  EXPECT_EQ(optimized_mesh.num_vertices(), 167);
  VolumeMeshFieldLinear<double, double> optimized_field =
      MakeEmPressSDField(optimized_mesh, original_M);

  WriteVolumeMeshFieldLinearToVtk("yellow_pepper_EmPress_optimized_sdfield.vtk",
                                  "SignedDistance(meters)", optimized_field,
                                  "Optimized EmbeddedSignedDistanceField");

  const double rms_error = CalcRMSErrorOfSDField(optimized_field, original_M);

  // About 0.6mm RMS error.
  EXPECT_NEAR(rms_error, 0.000608, 1e-6);
}

GTEST_TEST(MakeEmPressSDFieldAdapt, GenerateFromInputSurface) {
  TriangleSurfaceMesh<double> input_mesh_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(input_mesh_M.num_vertices(), 486);
  EXPECT_EQ(input_mesh_M.num_triangles(), 968);

  const auto [mesh_EmPress_M, sdfield_EmPress_M] =
      MakeEmPressSDFieldAdapt(input_mesh_M);  // future: min feature size

  ASSERT_NE(mesh_EmPress_M.get(), nullptr);
  ASSERT_NE(sdfield_EmPress_M.get(), nullptr);

  EXPECT_EQ(mesh_EmPress_M->num_vertices(), 7744);
  EXPECT_EQ(mesh_EmPress_M->num_elements(), 5808);
  WriteVolumeMeshFieldLinearToVtk("yellow_pepper_EmPress_sdfield_adapt.vtk",
                                  "SignedDistance(meters)", *sdfield_EmPress_M,
                                  "EmbeddedSignedDistanceField");

}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
