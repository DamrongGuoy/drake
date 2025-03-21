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

GTEST_TEST(DecimateOptimizedSdFieldTest, FromVtkFile) {
  const std::string vtk_file = FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_optimized_sdfield.vtk");

  // Read the VTK file.
  vtkNew<vtkUnstructuredGridReader> reader;
  reader->SetFileName(vtk_file.c_str());
  reader->Update();
  const vtkIdType num_input_tetrahedra =
      reader->GetOutput()->GetNumberOfCells();
  EXPECT_EQ(num_input_tetrahedra, 568);

  // Decimate the tetrahedral mesh + field.
  vtkNew<vtkUnstructuredGridQuadricDecimation> decimate;
  decimate->SetInputConnection(reader->GetOutputPort());
  decimate->SetScalarsName("SignedDistance(meters)");
  // This will shrink the tetrahedral count to 1/10 original.
  const double kTargetReduction = 0.9;
  decimate->SetTargetReduction(kTargetReduction);
  decimate->Update();
  vtkUnstructuredGrid* vtk_mesh = decimate->GetOutput();
  const vtkIdType num_tetrahedra = vtk_mesh->GetNumberOfCells();
  EXPECT_EQ(num_tetrahedra, 56);
  const vtkIdType num_vertices = vtk_mesh->GetNumberOfPoints();
  EXPECT_EQ(num_vertices, 28);

  // Convert to drake::geometry::VolumeMesh.
  std::vector<Vector3d> vertices;
  vertices.reserve(num_vertices);
  vtkPoints* vtk_vertices = vtk_mesh->GetPoints();
  for (vtkIdType id = 0; id < num_vertices; id++) {
    double xyz[3];
    vtk_vertices->GetPoint(id, xyz);
    vertices.emplace_back(xyz);
  }
  std::vector<VolumeElement> tetrahedra;
  tetrahedra.reserve(vtk_mesh->GetNumberOfCells());
  auto iter =
      vtkSmartPointer<vtkCellIterator>::Take(vtk_mesh->NewCellIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal();
       iter->GoToNextCell()) {
    DRAKE_THROW_UNLESS(iter->GetCellType() == VTK_TETRA);
    vtkIdList* vtk_vertex_ids = iter->GetPointIds();
    // clang-format off
    tetrahedra.emplace_back(vtk_vertex_ids->GetId(0),
                            vtk_vertex_ids->GetId(1),
                            vtk_vertex_ids->GetId(2),
                            vtk_vertex_ids->GetId(3));
    // clang-format on
  }
  VolumeMesh<double> decimated_mesh{std::move(tetrahedra), std::move(vertices)};

  // Regenerate the signed-distance field with respect to the original input
  // surface again.
  TriangleSurfaceMesh<double> original_surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  VolumeMeshFieldLinear<double, double> decimated_field =
      MakeEmPressSDField(decimated_mesh, original_surface_M);

  // 7mm RMS Error.
  EXPECT_NEAR(CalcRMSErrorOfSDField(decimated_field, original_surface_M), 0.007,
              1e-3);
  // For debugging.
  // WriteVolumeMeshFieldLinearToVtk(
  //     "yellow_pepper_EmPress_decimated_optimized_sdfield.vtk",
  //     "SignedDistance(meters)", decimated_field,
  //     "Decimated Optimized EmbeddedSignedDistanceField");
}

GTEST_TEST(DecimateOptimizedSdFieldTest, FromMeshFieldLinear) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_optimized_sdfield.vtk")};
  const VolumeMesh<double> support_mesh_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(support_mesh_M.num_vertices(), 167);
  EXPECT_EQ(support_mesh_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &support_mesh_M};

  vtkNew<vtkPoints> vtk_points;
  for (const Vector3d& p_MV : support_mesh_M.vertices()) {
    vtk_points->InsertNextPoint(p_MV.x(), p_MV.y(), p_MV.z());
  }
  vtk_points->Modified();
  vtkNew<vtkUnstructuredGrid> vtk_mesh;
  vtk_mesh->SetPoints(vtk_points);
  for (const VolumeElement& tet : support_mesh_M.tetrahedra()) {
    const vtkIdType ptIds[] = {tet.vertex(0), tet.vertex(1), tet.vertex(2),
                               tet.vertex(3)};
    vtk_mesh->InsertNextCell(VTK_TETRA, 4, ptIds);
  }
  vtk_mesh->Modified();
  vtkNew<vtkDoubleArray> vtk_signed_distances;
  vtk_signed_distances->Allocate(support_mesh_M.num_vertices());
  vtk_signed_distances->SetName("SignedDistance(meters)");
  for (int v = 0; v < support_mesh_M.num_vertices(); ++v) {
    vtk_signed_distances->SetValue(v, sdf_M.EvaluateAtVertex(v));
  }
  vtk_signed_distances->Modified();
  vtk_mesh->GetPointData()->AddArray(vtk_signed_distances);
  vtk_mesh->Modified();

  // Decimate the tetrahedral mesh + field.
  vtkNew<vtkUnstructuredGridQuadricDecimation> decimate;
  decimate->SetInputData(vtk_mesh);
  decimate->SetScalarsName("SignedDistance(meters)");
  // This will shrink the tetrahedral count to 1/10 original.
  const double kTargetReduction = 0.9;
  decimate->SetTargetReduction(kTargetReduction);
  decimate->Update();

  vtkUnstructuredGrid* vtk_decimated_mesh = decimate->GetOutput();
  const vtkIdType num_tetrahedra = vtk_decimated_mesh->GetNumberOfCells();
  EXPECT_EQ(num_tetrahedra, 56);
  const vtkIdType num_vertices = vtk_decimated_mesh->GetNumberOfPoints();
  EXPECT_EQ(num_vertices, 31);

  // Convert to drake::geometry::VolumeMesh.
  std::vector<Vector3d> vertices;
  vertices.reserve(num_vertices);
  vtkPoints* vtk_vertices = vtk_decimated_mesh->GetPoints();
  for (vtkIdType id = 0; id < num_vertices; id++) {
    double xyz[3];
    vtk_vertices->GetPoint(id, xyz);
    vertices.emplace_back(xyz);
  }
  std::vector<VolumeElement> tetrahedra;
  tetrahedra.reserve(vtk_decimated_mesh->GetNumberOfCells());
  auto iter = vtkSmartPointer<vtkCellIterator>::Take(
      vtk_decimated_mesh->NewCellIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal();
       iter->GoToNextCell()) {
    DRAKE_THROW_UNLESS(iter->GetCellType() == VTK_TETRA);
    vtkIdList* vtk_vertex_ids = iter->GetPointIds();
    // clang-format off
    tetrahedra.emplace_back(vtk_vertex_ids->GetId(0),
                            vtk_vertex_ids->GetId(1),
                            vtk_vertex_ids->GetId(2),
                            vtk_vertex_ids->GetId(3));
    // clang-format on
  }
  VolumeMesh<double> decimated_mesh{std::move(tetrahedra), std::move(vertices)};

  // Regenerate the signed-distance field with respect to the original input
  // surface again.
  TriangleSurfaceMesh<double> original_surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  VolumeMeshFieldLinear<double, double> decimated_field =
      MakeEmPressSDField(decimated_mesh, original_surface_M);

  // 7mm RMS Error.
  EXPECT_NEAR(CalcRMSErrorOfSDField(decimated_field, original_surface_M), 0.003,
              1e-3);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk(
      "yellow_pepper_EmPress_decimated_optimized_sdfield.vtk",
      "SignedDistance(meters)", decimated_field,
      "Decimated Optimized EmbeddedSignedDistanceField");
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
