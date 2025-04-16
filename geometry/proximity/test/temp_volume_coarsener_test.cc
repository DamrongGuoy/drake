#include <iostream>

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
#include "drake/geometry/proximity/make_ellipsoid_field.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/temp_volume_coarsener.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using math::RigidTransformd;
using math::RollPitchYawd;

GTEST_TEST(EigenSVDTest, FromSherm) {
  // Define a 4x4 matrix A and a 4x1 vector b
  Eigen::Matrix4d A;
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  Eigen::Vector4d b(1.0, 2.0, 3.0, 4.0);

  // Compute SVD of A
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Solve Ax = b using SVD
  Eigen::Vector4d x = svd.solve(b);

  // Output the solution
  std::cout << "Solution x:\n" << fmt::format("{}", fmt_eigen(x)) << std::endl;

  // Verify the solution (optional)
  std::cout << "Verification (A*x - b):\n"
            << fmt::format("{}", fmt_eigen(A * x - b)) << std::endl;
}

GTEST_TEST(EigenSVDTest, From_mesh_12vert_20tet_2IntV) {
  // clang-format off
  Matrix4d A1;
  A1 <<
     0.3298969072164947,                      0,                      0,                      0,
                      0,     0.1855670103092784,                      0,                      0,
                      0,                      0,     0.7422680412371133, -6.938893903907228e-17,
                      0,                      0, -6.938893903907228e-17,     0.7422680412371139;
  const Vector4d p1(0,     0,     0, -0.02);
  Matrix4d A2;
  A2 <<
     0.5510200929923924, -3.469446951953614e-18,  6.591949208711867e-17,      1.077181511882946,
 -3.469446951953614e-18,     0.1851037450380721, -2.081668171172169e-17,  2.775557561562891e-17,
  6.591949208711867e-17, -2.081668171172169e-17,      0.740414980152288, -2.775557561562891e-17,
      1.077181511882946,  2.775557561562891e-17, -2.775557561562891e-17,      2.523461181817247;
  // clang-format on
  const Vector4d p2(-0.00375, 0, 0, -0.0175);

  const Matrix4d A = A1 + A2;
  const Vector4d b = A1 * p1 + A2 * p2;

  drake::log()->warn("A = \n{}", fmt_eigen(A));
  drake::log()->warn("bᵀ = {}", fmt_eigen(b.transpose()));

  // Eigen::JacobiSVD<Matrix4d, Eigen::ComputeFullU | Eigen::ComputeFullV>
  //     svd(A);
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);

  drake::log()->warn("svd.singularValues()ᵀ = {}",
                     fmt_eigen(svd.singularValues().transpose()));

  Vector4d x = svd.solve(b);
  drake::log()->warn("xᵀ = svd.solve(b)ᵀ = {}", fmt_eigen(x.transpose()));

  Vector4d r = b - A * x;
  drake::log()->warn("(r = b - A*x)ᵀ = {}", fmt_eigen(r.transpose()));
}

GTEST_TEST(VolumeMeshCoarsenerTest, TwoInteriorVertices) {
  const Mesh mesh_spec{
      FindResourceOrThrow("drake/geometry/test/mesh_12vert_20tet_2IntV.vtk")};
  // It's the local mesh around two interior vertices in an ellipsoid.
  const VolumeMesh<double> local_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec);
  EXPECT_EQ(local_mesh_M.num_vertices(), 12);
  EXPECT_EQ(local_mesh_M.num_elements(), 20);

  // Confirm that there are two interior vertices by verifying that
  // the extracted surface mesh has 10 vertices.  Remove this extra
  // check later.
  {
    auto check_surface_mesh = ConvertVolumeToSurfaceMesh(local_mesh_M);
    EXPECT_EQ(check_surface_mesh.num_vertices(), 10);
  }

  // This is the original ellipsoid that we used for debugging and producing
  // the above local mesh.
  const Ellipsoid ellipsoid_M(0.03, 0.04, 0.02);
  // Hydroelastic modulus = 0.02 the length of the minimum principal semi-axis
  // will give the highest pressure = +0.02 at the center.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeEllipsoidPressureField<double>(ellipsoid_M, &local_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &local_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      MakeEllipsoidSurfaceMesh<double>(ellipsoid_M, 0.01);

  const int kTargetNumTetrahedra = 16;
  const double kFraction =
      static_cast<double>(kTargetNumTetrahedra) / local_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);
}

GTEST_TEST(EigenSVDTest, 2025_04_15_v105_v601) {
  // clang-format off
  Matrix4d A1;
  A1 <<
    6.018045408850664e-10, -2.584939414228211e-26,  1.033975765691285e-25, -5.169878828456423e-26,
   -2.584939414228211e-26,  3.385150542478499e-10, -1.421716677825516e-25,                      0,
    1.033975765691285e-25, -1.421716677825516e-25,  1.367562530432047e-08,  1.501019878597912e-08,
   -5.169878828456423e-26,                      0,  1.501019878597912e-08,  1.663405510054661e-08;
  const Vector4d p1(0, 0, -0.005, -0.015);
  Matrix4d A2;
  A2 <<
    2.376917340600975e-10, -5.331437541845686e-26,                      0,                      0,
   -5.331437541845686e-26,  1.337016004088048e-10,   9.04728794979874e-26,  2.584939414228211e-26,
                        0,   9.04728794979874e-26,  1.484743518050234e-08,  1.541662971671698e-08,
                        0,  2.584939414228211e-26,  1.541662971671698e-08,  1.603117148502875e-08;
  // clang-format on
  const Vector4d p2(0, 0, -0.0075, -0.0125);

  const Matrix4d A = A1 + A2;
  const Vector4d b = A1 * p1 + A2 * p2;

  Eigen::JacobiSVD<Matrix4d> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vector4d x = svd.solve(b);

  // x should be between the original p1(0, 0, -0.005, -0.015) and
  // p2(0, 0, -0.0075, -0.0125), but it didn't. The z and w coordinates of p1
  // and p2 are both sizable negative numbers (order of millimeters or
  // centimeters); however, perhaps underflow caused x to be zero in all
  // coordinates?
  EXPECT_TRUE(CompareMatrices(
      x, Vector4d(0, 0, -0.005044629395883285, -0.01491139552061834), 1e-16));
}

GTEST_TEST(VolumeMeshCoarsenerTest, Ellipsoid_1024) {
  const Ellipsoid ellipsoid_M(0.03, 0.04, 0.02);
  const VolumeMesh<double> ellipsoid_mesh_M = MakeEllipsoidVolumeMesh<double>(
      ellipsoid_M, 0.01, TessellationStrategy::kDenseInteriorVertices);
  // resolution:   0.005  0.01
  // num_vertices:  6017   833
  // num_element:  32768  4096
  EXPECT_EQ(ellipsoid_mesh_M.num_vertices(), 833);
  EXPECT_EQ(ellipsoid_mesh_M.num_elements(), 4096);

  // Hydroelastic modulus = 0.02 the length of the minimum principal semi-axis
  // will give the highest pressure = +0.02 at the center.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeEllipsoidPressureField<double>(ellipsoid_M, &ellipsoid_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &ellipsoid_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      ConvertVolumeToSurfaceMesh(ellipsoid_mesh_M);

  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_sdf.vtk", "SignedDistance(meter)",
                                  sdf_M, "VolumeMeshCoarsener coarsen");

  const int kTargetNumTetrahedra = 1024;
  const double kFraction = static_cast<double>(kTargetNumTetrahedra) /
                           ellipsoid_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(), 1.01 * kTargetNumTetrahedra);

  EXPECT_LT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
  int num_negative_volume_tetrahedra = 0;
  coarsen_mesh_M = VolumeMeshCoarsener::HackNegativeToPositiveVolume(
      coarsen_mesh_M, &num_negative_volume_tetrahedra);
  EXPECT_EQ(num_negative_volume_tetrahedra, 75);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_1024_coarsen_sdf.vtk",
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");
}

#if 0

GTEST_TEST(VolumeMeshCoarsenerTest, Ellipsoid_0256) {
  const Ellipsoid ellipsoid_M(0.03, 0.04, 0.02);
  const VolumeMesh<double> ellipsoid_mesh_M = MakeEllipsoidVolumeMesh<double>(
      ellipsoid_M, 0.01, TessellationStrategy::kDenseInteriorVertices);
  // resolution:   0.005  0.01
  // num_vertices:  6017   833
  // num_element:  32768  4096
  EXPECT_EQ(ellipsoid_mesh_M.num_vertices(), 833);
  EXPECT_EQ(ellipsoid_mesh_M.num_elements(), 4096);

  // Hydroelastic modulus = 0.02 the length of the minimum principal semi-axis
  // will give the highest pressure = +0.02 at the center.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeEllipsoidPressureField<double>(ellipsoid_M, &ellipsoid_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &ellipsoid_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      ConvertVolumeToSurfaceMesh(ellipsoid_mesh_M);

  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_sdf.vtk", "SignedDistance(meter)",
                                  sdf_M, "VolumeMeshCoarsener coarsen");

  const int kTargetNumTetrahedra = 256;
  const double kFraction = static_cast<double>(kTargetNumTetrahedra) /
                           ellipsoid_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(), 1.01 * kTargetNumTetrahedra);
  EXPECT_LT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
  int num_negative_volume_tetrahedra = 0;
  coarsen_mesh_M = VolumeMeshCoarsener::HackNegativeToPositiveVolume(
      coarsen_mesh_M, &num_negative_volume_tetrahedra);
  EXPECT_EQ(num_negative_volume_tetrahedra, 29);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_0256_coarsen_sdf.vtk",
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");
}

GTEST_TEST(VolumeMeshCoarsenerTest, Ellipsoid_0064) {
  const Ellipsoid ellipsoid_M(0.03, 0.04, 0.02);
  const VolumeMesh<double> ellipsoid_mesh_M = MakeEllipsoidVolumeMesh<double>(
      ellipsoid_M, 0.01, TessellationStrategy::kDenseInteriorVertices);
  // resolution:   0.005  0.01
  // num_vertices:  6017   833
  // num_element:  32768  4096
  EXPECT_EQ(ellipsoid_mesh_M.num_vertices(), 833);
  EXPECT_EQ(ellipsoid_mesh_M.num_elements(), 4096);

  // Hydroelastic modulus = 0.02 the length of the minimum principal semi-axis
  // will give the highest pressure = +0.02 at the center.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeEllipsoidPressureField<double>(ellipsoid_M, &ellipsoid_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &ellipsoid_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      ConvertVolumeToSurfaceMesh(ellipsoid_mesh_M);

  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_sdf.vtk", "SignedDistance(meter)",
                                  sdf_M, "VolumeMeshCoarsener coarsen");

  const int kTargetNumTetrahedra = 64;
  const double fraction = static_cast<double>(kTargetNumTetrahedra) /
                          ellipsoid_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(fraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(), 1.01 * kTargetNumTetrahedra);
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_0064_coarsen_sdf.vtk",
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");
}

GTEST_TEST(VolumeMeshCoarsenerTest, Box) {
  // About the size of a computer mouse.
  const Box box_M{0.07, 0.10, 0.04};
  const Box support_box_M{box_M.size() + Vector3d::Constant(0.01)};
  const VolumeMesh<double> box_mesh_M =
      MakeBoxVolumeMesh<double>(support_box_M, 0.01);
  const TriangleSurfaceMesh<double> original_surface_M =
      MakeBoxSurfaceMeshWithSymmetricTriangles<double>(box_M);
  EXPECT_EQ(box_mesh_M.num_vertices(), 648);
  EXPECT_EQ(box_mesh_M.num_elements(), 2640);

  const VolumeMeshFieldLinear<double, double> sdf_M =
      MakeEmPressSDField(box_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("box_sdf.vtk", "SignedDistance(meter)", sdf_M,
                                  "VolumeMeshCoarsener coarsen");

  const int kTargetNumTetrahedra = 264;
  const double fraction =
      static_cast<double>(kTargetNumTetrahedra) / box_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(fraction);
  EXPECT_LT(coarsen_mesh_M.num_elements(),
            static_cast<int>(1.01 * kTargetNumTetrahedra));

  EXPECT_LT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
  int num_negative_volume_tetrahedra = 0;
  coarsen_mesh_M = VolumeMeshCoarsener::HackNegativeToPositiveVolume(
      coarsen_mesh_M, &num_negative_volume_tetrahedra);
  EXPECT_EQ(num_negative_volume_tetrahedra, 28);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("box_coarsen_sdf.vtk",
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");

  EXPECT_LT(CalcRMSErrorOfSDField(coarsen_sdf_M, original_surface_M), 0.01);
}

GTEST_TEST(VolumeMeshCoarsenerTest, FromMeshFieldLinear) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_optimized_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};
  TriangleSurfaceMesh<double> original_surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));

  ASSERT_GT(support_mesh_M.CalcMinTetrahedralVolume(), 0);

  const double kFraction = 0.1;
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);
  EXPECT_LT(coarsen_mesh_M.num_elements(),
            static_cast<int>(1.01 * kFraction * 568));

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("test_coarsen_sdf.vtk",
                                  "SignedDistance(meters)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");

  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
  EXPECT_LT(CalcRMSErrorOfSDField(coarsen_sdf_M, original_surface_M), 0.01);
}

// Use VTK implementation (vtkUnstructuredGridQuadricDecimation)
GTEST_TEST(TempCoarsenVolumeMeshOfSdField, Ellipsoid) {
  const Ellipsoid ellipsoid_M(0.03, 0.04, 0.02);
  const VolumeMesh<double> support_mesh_M = MakeEllipsoidVolumeMesh<double>(
      // 0.004, 0.005
      ellipsoid_M, 0.01, TessellationStrategy::kDenseInteriorVertices);
  // resolution:   0.005  0.01
  // num_vertices:  6017   833
  // num_element:  32768  4096
  EXPECT_EQ(support_mesh_M.num_vertices(), 833);
  EXPECT_EQ(support_mesh_M.num_elements(), 4096);

  // Hydroelastic modulus = 0.02 the length of the minimum principal semi-axis
  // will give the highest pressure = +0.02 at the center.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeEllipsoidPressureField<double>(ellipsoid_M, &support_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &support_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      ConvertVolumeToSurfaceMesh(support_mesh_M);

  const int kTargetNumTetrahedra = 10;
  const double fraction =
      static_cast<double>(kTargetNumTetrahedra) / support_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      TempCoarsenVolumeMeshOfSdField(sdf_M, fraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(), 1.01 * kTargetNumTetrahedra);
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("ellipsoid_VtkCoarsen_sdf.vtk",
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "vtkUnstructuredGridQuadricDecimation");
}

GTEST_TEST(TempCoarsenVolumeMeshOfSdField, FromMeshFieldLinear) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_optimized_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};

  VolumeMesh<double> coarsen_mesh_M =
      TempCoarsenVolumeMeshOfSdField(sdf_M, 0.1);
  EXPECT_LT(coarsen_mesh_M.num_elements(), 60);

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

#endif

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
