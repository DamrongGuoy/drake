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

using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;

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
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  if (coarsen_mesh_M.CalcMinTetrahedralVolume() < 0) {
    coarsen_mesh_M =
        VolumeMeshCoarsener::HackNegativeToPositiveVolume(coarsen_mesh_M);
  }

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
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  if (coarsen_mesh_M.CalcMinTetrahedralVolume() < 0) {
    coarsen_mesh_M =
        VolumeMeshCoarsener::HackNegativeToPositiveVolume(coarsen_mesh_M);
  }

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
  const double kFraction = static_cast<double>(kTargetNumTetrahedra) /
                           ellipsoid_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(), 1.01 * kTargetNumTetrahedra);
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  if (coarsen_mesh_M.CalcMinTetrahedralVolume() < 0) {
    coarsen_mesh_M =
        VolumeMeshCoarsener::HackNegativeToPositiveVolume(coarsen_mesh_M);
  }

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
  const VolumeMesh<double> box_mesh_M = MakeBoxVolumeMesh<double>(box_M, 0.01);
  EXPECT_EQ(box_mesh_M.num_vertices(), 495);
  EXPECT_EQ(box_mesh_M.num_elements(), 1920);

  // Hydroelastic modulus = 0.02 (half thickness of the box) creates the
  // highest pressure = +0.02 at the center rectangle.
  const VolumeMeshFieldLinear<double, double> pressure_M =
      MakeBoxPressureField<double>(box_M, &box_mesh_M, 0.02);
  // Assign the negative of pressure values to signed distance values.
  // The pressure and the signed distance have opposite sign conventions.
  std::vector<double> signed_distances;
  for (const double p : pressure_M.values()) {
    signed_distances.push_back(-p);
  }
  const VolumeMeshFieldLinear<double, double> sdf_M{std::move(signed_distances),
                                                    &box_mesh_M};
  const TriangleSurfaceMesh<double> original_surface_M =
      ConvertVolumeToSurfaceMesh(box_mesh_M);

  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("box_sdf.vtk", "SignedDistance(meter)", sdf_M,
                                  "VolumeMeshCoarsener coarsen");

  const double kFraction = 0.1;
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);

  EXPECT_LT(coarsen_mesh_M.num_elements(),
            static_cast<int>(1.01 * kFraction * box_mesh_M.num_elements()));
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  if (coarsen_mesh_M.CalcMinTetrahedralVolume() < 0) {
    coarsen_mesh_M =
        VolumeMeshCoarsener::HackNegativeToPositiveVolume(coarsen_mesh_M);
  }

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
  const double kFraction =
      static_cast<double>(kTargetNumTetrahedra) / support_mesh_M.num_elements();
  VolumeMesh<double> coarsen_mesh_M =
      TempCoarsenVolumeMeshOfSdField(sdf_M, kFraction);

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

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
