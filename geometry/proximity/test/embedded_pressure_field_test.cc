#include <gtest/gtest.h>

// TODO(DamrongGuoy): Remove these #include vtk.  Right now I'm checking
//  that bazel let me summon vtkUnstructuredGridQuadricDecimation from
//  the @vtk_internal.

// For files in "@vtk_internal//:vtkFooBar", you might see them in
// bazel-drake/external/+internal_repositories+vtk_internal/Foo/Bar/*.h

// To ease build system upkeep, we annotate VTK includes with their deps.
#include <vtkCellIterator.h>                       // vtkCommonDataModel
#include <vtkCleanPolyData.h>                      // vtkFiltersCore
#include <vtkDelaunay3D.h>                         // vtkFiltersCore
#include <vtkDoubleArray.h>                        // vtkCommonCore
#include <vtkPointData.h>                          // vtkCommonDataModel
#include <vtkPointSource.h>                        // vtkFiltersSources
#include <vtkPoints.h>                             // vtkCommonCore
#include <vtkPolyData.h>                           // vtkCommonDataModel
#include <vtkSmartPointer.h>                       // vtkCommonCore
#include <vtkUnstructuredGrid.h>                   // vtkCommonDataModel
#include <vtkUnstructuredGridQuadricDecimation.h>  // vtkFiltersCore
#include <vtkUnstructuredGridReader.h>             // vtkIOLegacy

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

GTEST_TEST(CoarsenSdField, FromMeshFieldLinear) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_optimized_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};

  VolumeMesh<double> coarsen_mesh_M = CoarsenSdField(sdf_M, 0.1);
  EXPECT_LT(coarsen_mesh_M.num_vertices(), 50);
  EXPECT_LT(coarsen_mesh_M.num_elements(), 200);

  TriangleSurfaceMesh<double> original_surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);

  EXPECT_LT(CalcRMSErrorOfSDField(coarsen_sdf_M, original_surface_M), 0.01);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk(
      "yellow_pepper_EmPress_decimated_optimized_sdfield.vtk",
      "SignedDistance(meters)", coarsen_sdf_M,
      "Decimated Optimized EmbeddedSignedDistanceField");
}

GTEST_TEST(CoarsenSdField, pepper_r0_005_sdf_optimize) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/pepper_r0.005_sdf_optimize.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 3575);
  EXPECT_EQ(support_mesh_M.num_elements(), 17413);
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};
  TriangleSurfaceMesh<double> original_surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_NEAR(CalcRMSErrorOfSDField(sdf_M, original_surface_M), 0.00016, 1e-5);

  VolumeMesh<double> coarsen_mesh_M = CoarsenSdField(sdf_M, 0.5);
  // TODO(DamrongGuoy):  We cannot check the exact number of vertices and
  //  tetrahedra due to the randomization in
  //  vtkUnstructuredGridQuadricDecimation. If we change the implementation
  //  of CoarsenSdField() to be deterministic, use EXPECT_EQ() instead.
  EXPECT_LT(coarsen_mesh_M.num_vertices(), 2100);
  EXPECT_LT(coarsen_mesh_M.num_elements(), 9000);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);

  EXPECT_NEAR(CalcRMSErrorOfSDField(coarsen_sdf_M, original_surface_M), 0.00018,
              1e-5);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk(
      "pepper_r0.005_sdf_optimize_coarsen.vtk", "SignedDistance(meters)",
      coarsen_sdf_M, "Decimated Optimized EmbeddedSignedDistanceField");
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
