#include <filesystem>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/calc_signed_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;

class SignedDistanceToInputYellowBellPepperSurfaceTest
    : public ::testing::Test {
 public:
  SignedDistanceToInputYellowBellPepperSurfaceTest()
      : mesh_M_(ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
            "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"))),
        bvh_M_(mesh_M_),
        mesh_normal_M_(std::get<FeatureNormalSet>(
            FeatureNormalSet::MaybeCreate(mesh_M_))) {}

 protected:
  const TriangleSurfaceMesh<double> mesh_M_;
  const Bvh<Obb, TriangleSurfaceMesh<double>> bvh_M_;
  const FeatureNormalSet mesh_normal_M_;
  const double kEps{1e-10};
};

// Positive signed distance with nearest point in a triangle.
TEST_F(SignedDistanceToInputYellowBellPepperSurfaceTest, SanityCheck) {
  const Vector3d p_MC = mesh_M_.element_centroid(0);
  const Vector3d n_M = mesh_M_.face_normal(0);
  // We don't know which one is face 0, but we know that a small-enough
  // translation from its centroid along its outward face normal keeps
  // the nearest point at its centroid. Here, we use 1 micron translation.
  const Vector3d p_MQ = p_MC + 1e-6 * n_M;
  const SignedDistanceToSurfaceMesh d =
      CalcSignedDistanceToSurfaceMesh(p_MQ, mesh_M_, bvh_M_, mesh_normal_M_);
  EXPECT_TRUE(CompareMatrices(d.nearest_point, p_MC, kEps));
  EXPECT_NEAR(d.signed_distance, 1e-6, kEps);
  EXPECT_TRUE(CompareMatrices(d.gradient, n_M, kEps));
}

TEST_F(SignedDistanceToInputYellowBellPepperSurfaceTest, FromCentroid) {
  const Vector3d p_MC = mesh_M_.centroid();
  EXPECT_TRUE(
      CompareMatrices(p_MC,
                      Vector3d{0.0003958794472057116, -0.002108731172542037,
                               0.03939643527278602},
                      kEps));
  const SignedDistanceToSurfaceMesh d =
      CalcSignedDistanceToSurfaceMesh(p_MC, mesh_M_, bvh_M_, mesh_normal_M_);
  EXPECT_TRUE(CompareMatrices(
      d.nearest_point,
      Vector3d{0.004840218708763107, 0.009154232436338356, 0.0554849597283423},
      kEps));
  // The nearest point of the centroid is around 1 or 2 centimeters above (+Z)
  // and behind (+Y) of the centroid.
  EXPECT_TRUE(CompareMatrices(
      d.nearest_point - p_MC,
      Vector3d{0.004444339261557395, 0.01126296360888039, 0.01608852445555629},
      kEps));
  // The centroid is about 2 centimeters deep.
  EXPECT_NEAR(d.signed_distance, -0.020135717515991757, kEps);
  EXPECT_TRUE(CompareMatrices(
      d.gradient,
      Vector3d{0.2207191900674862, 0.5593524839596783, 0.7990042789773349},
      kEps));
}

/* E N A B L E   W H E N   W E   N E E D   N E W   M E S H E S . */

// Recipe.
// 1. Input: a watertight, manifold, self-intersecting-free surface mesh.
// 2. Run MeshLab:
//      Filter/Remeshing_Simplification_Reconstruction
//        /Uniform Mesh Resampling (marching cube)
//    with coarse resolution and positive offset distance; for example,
//    1 centimeter (0.01 meter) for both the precision and the offset
//    parameters.
// 3. Save into _offset1cm.obj (no mtl).
// 4. Load the offset surface into this test.
// 5. Generate grid points at twice resolution of the offset surface; for
//    example, 2 centimeters.
// 6. Keep only the good grid points enclosed by the offset surface.
// 7. Output: the offset surface together with the good grid points in
//    a _surface.vtk file.
//
// After this test, externally run TetGen to create the background
// volumetric mesh and go to the next test. (Use venv_vtk2obj.py to convert the
// _surface.vtk file to .obj. Use venv_tetgen.py to create the embedding
// tetrahedral mesh.)

/******************* archive *
GTEST_TEST(GridInOffsetSurface, Offset1cmGrid2cm) {
  const TriangleSurfaceMesh<double> surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_pepper_offset1cm.obj"));
  EXPECT_EQ(surface_M.num_vertices(), 436);
  EXPECT_EQ(surface_M.num_triangles(), 868);

  const Bvh<Obb, TriangleSurfaceMesh<double>> surface_bvh_M{surface_M};
  const FeatureNormalSet surface_normal_M =
      std::get<FeatureNormalSet>(FeatureNormalSet::MaybeCreate(surface_M));

  // TODO(DamrongGuoy): Auto check for the bounding box and generate much
  //  less candidate points. Right now I happened to know the offset surface
  //  is bounded by [-0.050, 0.051]x[-0.051, 0.050]x[-0.01, 0.091], so
  //  a box of (±0.051)x(±0.051)x[±0.091] would be enough. Therefore, the box
  //  size is 0.102 x 0.102 x 0.182 meters (about 10x10x18 centimeters)
  Box bound(0.102, 0.102, 0.182);
  const double grid_resolution = 0.02;  // 2 centimeters.
  VolumeMesh<double> cover_grid =
      MakeBoxVolumeMesh<double>(bound, grid_resolution);

  std::vector<SurfaceTriangle> triangles{surface_M.triangles()};
  std::vector<Vector3d> vertices_M{surface_M.vertices()};
  // TODO(DamrongGuoy): Estimate the number of vertices in a better way.
  //  Right now I just guess.
  vertices_M.reserve(6000);

  int count_good_grid_points = 0;
  for (const Vector3d& grid : cover_grid.vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        grid, surface_M, surface_bvh_M, surface_normal_M);
    // TODO(DamrongGuoy): Better estimate of the distance threshold. Right now
    //  we pick any grid points inside the offset surface 1 millimeter and
    //  deeper.
    if (d.signed_distance <= -0.001) {
      ++count_good_grid_points;
      vertices_M.push_back(grid);
    }
  }
  EXPECT_EQ(count_good_grid_points, 92);

  TriangleSurfaceMesh<double> surface_with_grids{std::move(triangles),
                                                 std::move(vertices_M)};
  EXPECT_EQ(surface_with_grids.num_triangles(), 868);
  EXPECT_EQ(surface_with_grids.num_vertices(), 528);

  WriteSurfaceMeshToVtk("yellow_pepper_offset1cm_grid2cm_surface.vtk",
                        surface_with_grids, "YellowPepperOffsetEmbeddingGrid");
}

GTEST_TEST(BCCInsideOutOffsetSurface, GenerateOffset5mGrid1cm) {
  const TriangleSurfaceMesh<double> surface_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_pepper_MLabPrec5mmOffset5mm.obj"));
  EXPECT_EQ(surface_M.num_vertices(), 1522);
  EXPECT_EQ(surface_M.num_triangles(), 3040);

  const Bvh<Obb, TriangleSurfaceMesh<double>> surface_bvh_M{surface_M};
  const FeatureNormalSet surface_normal_M =
      std::get<FeatureNormalSet>(FeatureNormalSet::MaybeCreate(surface_M));

  // TODO(DamrongGuoy): Auto check for the bounding box and generate much
  //  less candidate points. Right now I happened to know the 18-centimeter
  //  cube centered at origin is more than 8-times enough for the pepper with
  //  bounding box [-0.05, 0.05]x[-0.05, 0,05]x[-0.006, 0.09].

  // 1-cm Cartesian grids covering the offset surface_M of 5-mm resolution.
  const double grid_resolution = 0.01;
  VolumeMesh<double> cover_grid =
      MakeBoxVolumeMesh<double>(Box::MakeCube(0.18), grid_resolution);
  WriteVolumeMeshToVtk("Cube18cm.vtk", cover_grid, "Cube18Centimeters");

  std::vector<SurfaceTriangle> triangles{surface_M.triangles()};
  std::vector<Vector3d> vertices_M{surface_M.vertices()};
  // TODO(DamrongGuoy): Estimate the number of vertices in a better way.
  //  Right now I just guess.
  vertices_M.reserve(6000);

  int count_good_grid_points = 0;
  for (const Vector3d& grid : cover_grid.vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        grid, surface_M, surface_bvh_M, surface_normal_M);
    // TODO(DamrongGuoy): Better estimate of the distance threshold. Right now
    //  we pick any grid points inside the offset surface 1 millimeter and
    //  deeper.
    if (d.signed_distance <= -0.001) {
      ++count_good_grid_points;
      vertices_M.push_back(grid);
    }
  }
  EXPECT_EQ(count_good_grid_points, 358);

  TriangleSurfaceMesh<double> surface_with_grids{std::move(triangles),
                                                 std::move(vertices_M)};
  EXPECT_EQ(surface_with_grids.num_triangles(), 3040);
  EXPECT_EQ(surface_with_grids.num_vertices(), 1522 + 358);

  WriteSurfaceMeshToVtk("yellow_pepper_offset_and_grids.vtk",
                        surface_with_grids, "YellowPepperOffsetEmbeddingGrid");
}
******************* archive */

// Create interpolated signed-distance field on the tetrahedral mesh
// (_tetgen.vtk) with respect to the input surface mesh (.obj).
GTEST_TEST(SDToSurfaceMeshFromVolumePoints, SignedDistanceField) {
  TriangleSurfaceMesh<double> input_surface_mesh =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(input_surface_mesh.num_vertices(), 486);
  EXPECT_EQ(input_surface_mesh.num_triangles(), 968);

  const Bvh<Obb, TriangleSurfaceMesh<double>> surface_bvh{input_surface_mesh};
  const FeatureNormalSet surface_normal = std::get<FeatureNormalSet>(
      FeatureNormalSet::MaybeCreate(input_surface_mesh));

  VolumeMesh<double> embedding_tetrahedral_mesh =
      ReadVtkToVolumeMesh(std::filesystem::path(FindResourceOrThrow(
          "drake/geometry/test/yellow_pepper_Offset1cmGrid2cm_tetgen.vtk")));
  EXPECT_EQ(embedding_tetrahedral_mesh.num_vertices(), 528);
  EXPECT_EQ(embedding_tetrahedral_mesh.num_elements(), 1808);

  std::vector<double> signed_distances;
  for (const Vector3d& tet_vertex : embedding_tetrahedral_mesh.vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        tet_vertex, input_surface_mesh, surface_bvh, surface_normal);
    signed_distances.push_back(d.signed_distance);
  }

  VolumeMeshFieldLinear<double, double> embedded_sdf{
      std::move(signed_distances), &embedding_tetrahedral_mesh};
  WriteVolumeMeshFieldLinearToVtk("yellow_pepper_Offset1cmGrid2cm_sdfield.vtk",
                                  "SignedDistance(meters)", embedded_sdf,
                                  "EmbeddedSignedDistanceField");
}

Aabb CalcBoundingBox(const VolumeMesh<double>& mesh_M) {
  Vector3d min_xyz{std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  Vector3d max_xyz{std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min()};
  for (const Vector3d& p_MV : mesh_M.vertices()) {
    for (int c = 0; c < 3; ++c) {
      if (p_MV(c) < min_xyz(c)) {
        min_xyz(c) = p_MV(c);
      }
      if (p_MV(c) > max_xyz(c)) {
        max_xyz(c) = p_MV(c);
      }
    }
  }
  return {(min_xyz + max_xyz) / 2, (max_xyz - min_xyz) / 2};
}

// Measure "distance" between the zero-level set implicit surface and the
// original mesh.
GTEST_TEST(MeasureDeviation, SurfaceVsSDField) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_Offset1cmGrid2cm_sdfield.vtk")};
  auto embedded_mesh_M = std::make_unique<VolumeMesh<double>>(
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield));
  EXPECT_EQ(embedded_mesh_M->num_vertices(), 528);
  EXPECT_EQ(embedded_mesh_M->num_elements(), 1808);
  auto embedded_sdfield_M =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
          embedded_mesh_M.get());
  const hydroelastic::SoftMesh compliant_embedded_mesh_M{
      std::move(embedded_mesh_M), std::move(embedded_sdfield_M)};

  const Aabb bounding_box_M = CalcBoundingBox(compliant_embedded_mesh_M.mesh());
  // Frame B and frame M are axis-aligned. Only their origins are different.
  const Box box_B(2.0 * bounding_box_M.half_width());
  const RigidTransformd X_MB(bounding_box_M.center());
  auto box_mesh_B = std::make_unique<VolumeMesh<double>>(
      MakeBoxVolumeMeshWithMa<double>(box_B));
  auto box_field_B = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeBoxPressureField<double>(box_B, box_mesh_B.get(),
                                   /* hydroelastic_modulus */ 1e-6));
  const hydroelastic::SoftMesh compliant_box_B{std::move(box_mesh_B),
                                               std::move(box_field_B)};

  // The kTriangle argument makes the level0 surface include centroids of
  // contact polygons for more checks.
  std::unique_ptr<ContactSurface<double>> level0_M =
      ComputeContactSurfaceFromCompliantVolumes(
          GeometryId::get_new_id(), compliant_box_B, X_MB,
          GeometryId::get_new_id(), compliant_embedded_mesh_M,
          RigidTransformd::Identity(),
          HydroelasticContactRepresentation::kTriangle);
  EXPECT_EQ(level0_M->tri_mesh_W().num_vertices(), 6498);
  // Sanity check visually that it makes sense to compute the level-0 surface
  // using ComputeContactSurfaceFromCompliantVolumes().
  //
  // WriteTriangleSurfaceMeshFieldLinearToVtk(
  //    "level0_surface.vtk", "SignedDistance(meters)", level0_M->tri_e_MN(),
  //    "SignedDistance(meters)"
  //);

  const TriangleSurfaceMesh<double> input_surface_mesh_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(input_surface_mesh_M.num_vertices(), 486);
  EXPECT_EQ(input_surface_mesh_M.num_triangles(), 968);
  const Bvh<Obb, TriangleSurfaceMesh<double>> input_surface_bvh_M{
      input_surface_mesh_M};
  const FeatureNormalSet input_surface_normal = std::get<FeatureNormalSet>(
      FeatureNormalSet::MaybeCreate(input_surface_mesh_M));

  double average_absolute_deviation = 0;
  double max_absolute_deviation = 0;
  double min_absolute_deviation = std::numeric_limits<double>::max();
  for (const Vector3d& level0_vertex : level0_M->tri_mesh_W().vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        level0_vertex, input_surface_mesh_M, input_surface_bvh_M,
        input_surface_normal);
    const double absolute_deviation = std::abs(d.signed_distance);
    average_absolute_deviation += absolute_deviation;
    if (absolute_deviation < min_absolute_deviation) {
      min_absolute_deviation = absolute_deviation;
    }
    if (absolute_deviation > max_absolute_deviation) {
      max_absolute_deviation = absolute_deviation;
    }
  }
  average_absolute_deviation /= level0_M->tri_mesh_W().num_vertices();

  // About 20 nanometers minimum deviation.
  EXPECT_NEAR(min_absolute_deviation, 2.026e-8, 1e-11);
  // About half a millimeter average deviation.
  EXPECT_NEAR(average_absolute_deviation, 0.0006822, 1e-6);
  // About 9 millimeters maximum deviation.
  EXPECT_NEAR(max_absolute_deviation, 0.008890, 1e-6);

  // TODO(DamrongGuoy): Measure distance from the input surface's vertices to
  //  the level0 surface too.
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
