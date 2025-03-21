#include <gtest/gtest.h>

// TODO(DamrongGuoy): Remove these #include vtk.  Right now I'm checking
//  that bazel let me summon vtkUnstructuredGridQuadricDecimation from
//  the @vtk_internal.

// For files in "@vtk_internal//:vtkFooBar", you might see them in
// bazel-drake/external/+internal_repositories+vtk_internal/Foo/Bar/*.h

// To ease build system upkeep, we annotate VTK includes with their deps.
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

// TODO(DamrongGuoy) Fix the following link errors and activate this test again.
//  I believe they are in "@vtk_internal//:vtkFiltersCore"
//  - vtkCleanPolyData.h
//  - vtkDelaunay3D.h
//  - vtkUnstructuredGridQuadricDecimation.h
//  bazel-out/k8-opt/...::TetrahedralFieldDecimator_SmokeTest_Test::TestBody():
//  error: undefined reference to 'drake_vendor::vtkCleanPolyData::New()'
//  bazel-out/k8-opt/...::TetrahedralFieldDecimator_SmokeTest_Test::TestBody():
//  error: undefined reference to 'drake_vendor::vtkDelaunay3D::New()'
//  bazel-out/k8-opt/...::TetrahedralFieldDecimator_SmokeTest_Test::TestBody():
//  error: undefined reference to
//  'drake_vendor::vtkUnstructuredGridQuadricDecimation::New()' collect2: error:
//  ld returned 1

GTEST_TEST(TetrahedralFieldDecimator, SmokeTest) {
  // From
  // https://gitlab.kitware.com/vtk/vtk/-/blob/v9.4.0/Filters/Core/Testing/Cxx/TestUnstructuredGridQuadricDecimation.cxx

  // SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill
  // Lorensen SPDX-License-Identifier: BSD-3-Clause

  // This test constructs a tetrahedrally meshed sphere by first generating
  // <numberOfOriginalPoints> points randomly placed within a unit sphere, then
  // removing points that overlap within a tolerance, and finally constructing a
  // delaunay 3d tetrahedralization from the points. Additionally, point data
  // corresponding to the points distance from the origin are added to this
  // data. The resulting tetrahedral mesh is then decimated <numberOfTests>
  // times, each time with a target reduction facter <targetReduction[test]>.
  // The number of remaining tetrahedra is then compared to the original number
  // of tetrahedra and compared against the target reduction factor. If the
  // difference is greater than <absTolerance>, the test fails. Otherwise, the
  // test passes.

  // # of points to generate the original tetrahedral mesh
  const vtkIdType numberOfOriginalPoints = 1.e4;

  // # of decimation tests to perform
  const vtkIdType numberOfTests = 4;

  // target reduction values for each test
  const double targetReduction[numberOfTests] = {.1, .3, .5, .7};

  // absolute tolerance between the expected and received tetrahedron reduction
  // to determine whether the decimation successfully executed
  const double absTolerance = 1.e-1;

  // Generate points within a unit sphere centered at the origin.
  vtkSmartPointer<vtkPointSource> source =
      vtkSmartPointer<vtkPointSource>::New();
  source->SetNumberOfPoints(numberOfOriginalPoints);
  source->SetCenter(0., 0., 0.);
  source->SetRadius(1.);
  source->SetDistributionToUniform();
  source->SetOutputPointsPrecision(vtkAlgorithm::DOUBLE_PRECISION);

  // Clean the polydata. This will remove overlapping points that may be
  // present in the input data.
  vtkSmartPointer<vtkCleanPolyData> cleaner =
      vtkSmartPointer<vtkCleanPolyData>::New();
  cleaner->SetInputConnection(source->GetOutputPort());
  cleaner->Update();

  // Create point data for use in decimation (the point data acts as a fourth
  // dimension in a Euclidean metric for determining the "nearness" of points).
  vtkPolyData* pd = cleaner->GetOutput();
  vtkPoints* points = pd->GetPoints();
  vtkSmartPointer<vtkDoubleArray> radius =
      vtkSmartPointer<vtkDoubleArray>::New();
  radius->SetName("radius");
  radius->SetNumberOfComponents(1);
  radius->SetNumberOfTuples(points->GetNumberOfPoints());
  double xyz[3];
  double r;
  for (vtkIdType i = 0; i < points->GetNumberOfPoints(); i++) {
    points->GetPoint(i, xyz);
    r = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
    radius->SetTypedTuple(i, &r);
  }
  pd->GetPointData()->SetScalars(radius);

  // Generate a tetrahedral mesh from the input points. By
  // default, the generated volume is the convex hull of the points.
  vtkSmartPointer<vtkDelaunay3D> delaunay3D =
      vtkSmartPointer<vtkDelaunay3D>::New();
  delaunay3D->SetInputData(pd);
  delaunay3D->Update();

  const vtkIdType numberOfOriginalTetras =
      delaunay3D->GetOutput()->GetNumberOfCells();

  for (vtkIdType test = 0; test < numberOfTests; test++) {
    // Decimate the tetrahedral mesh.
    vtkSmartPointer<vtkUnstructuredGridQuadricDecimation> decimate =
        vtkSmartPointer<vtkUnstructuredGridQuadricDecimation>::New();
    decimate->SetInputConnection(delaunay3D->GetOutputPort());
    decimate->SetScalarsName("radius");
    decimate->SetTargetReduction(targetReduction[test]);
    decimate->Update();

    // Compare the resultant decimation fraction with the expected fraction.
    double fraction =
        (1. - static_cast<double>(decimate->GetOutput()->GetNumberOfCells()) /
                  numberOfOriginalTetras);

    std::cout << "Test # " << test << std::endl;
    std::cout << "number of original tetras: " << numberOfOriginalTetras
              << std::endl;
    std::cout << "number of tetras after decimation: "
              << decimate->GetOutput()->GetNumberOfCells() << std::endl;
    std::cout << "fraction: " << fraction << std::endl;
    std::cout << "expected fraction: " << targetReduction[test] << std::endl;
    EXPECT_LE(std::fabs(fraction - targetReduction[test]), absTolerance);
  }
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
