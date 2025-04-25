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

GTEST_TEST(VolumeMeshCoarsenerTest, mesh_28vert_52tet) {
  const Mesh mesh_spec{
      FindResourceOrThrow("drake/geometry/test/mesh_28vert_52tet.vtk")};
  // It's the local mesh around two interior vertices in an ellipsoid.
  const VolumeMesh<double> local_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec);
  EXPECT_EQ(local_mesh_M.num_vertices(), 28);
  EXPECT_EQ(local_mesh_M.num_elements(), 52);

  // Confirm that there are two interior vertices by verifying that
  // the extracted surface mesh has 26 vertices.  Remove this extra
  // check later.
  {
    auto check_surface_mesh = ConvertVolumeToSurfaceMesh(local_mesh_M);
    EXPECT_EQ(check_surface_mesh.num_vertices(), 26);
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

  // for debugging
  WriteSurfaceMeshToVtk("ellipsoid_surface.vtk", original_surface_M,
                        "VolumeMeshCoarsenerTest mesh_28vert_52tet Ellipsoid");

  const int kTargetNumTetrahedra = 48;
  const double kFraction =
      static_cast<double>(kTargetNumTetrahedra) / local_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);

  EXPECT_LE(coarsen_mesh_M.num_elements(), kTargetNumTetrahedra);
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
}

#if 0

GTEST_TEST(VolumeMeshCoarsenerTest, mesh_12vert_20tet_2IntV) {
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

  EXPECT_LE(coarsen_mesh_M.num_elements(), kTargetNumTetrahedra);
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);
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
