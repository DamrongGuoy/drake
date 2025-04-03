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

  const double kFraction = 0.7;
  VolumeMesh<double> coarsen_mesh_M =
      VolumeMeshCoarsener(sdf_M, original_surface_M).coarsen(kFraction);
  EXPECT_LT(coarsen_mesh_M.num_elements(),
            static_cast<int>(1.1 * kFraction * 568));
  EXPECT_GT(coarsen_mesh_M.CalcMinTetrahedralVolume(), 0);

  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, original_surface_M);
  // For debugging.
  WriteVolumeMeshFieldLinearToVtk("test_coarsen_sdf.vtk",
                                  "SignedDistance(meters)", coarsen_sdf_M,
                                  "VolumeMeshCoarsener coarsen");

  EXPECT_LT(CalcRMSErrorOfSDField(coarsen_sdf_M, original_surface_M), 0.01);
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
