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

GTEST_TEST(EmPressSignedDistanceField, MeasureDeviation) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_sdfield.vtk")};
  const VolumeMesh<double> mesh_EmPress_M =
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield);
  EXPECT_EQ(mesh_EmPress_M.num_vertices(), 167);
  EXPECT_EQ(mesh_EmPress_M.num_elements(), 568);
  VolumeMeshFieldLinear<double, double> sdfield_EmPress_M{
      MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
      &mesh_EmPress_M};

  const TriangleSurfaceMesh<double> original_M =
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj"));
  EXPECT_EQ(original_M.num_vertices(), 486);
  EXPECT_EQ(original_M.num_triangles(), 968);

  const auto [min_absolute_deviation, average_absolute_deviation,
              max_absolute_deviation] =
      MesaureDeviationOfZeroLevelSet(sdfield_EmPress_M, original_M);

  // About 0.1 micrometers minimum deviation, practically zero.
  EXPECT_NEAR(min_absolute_deviation, 1.189e-07, 1e-10);
  // About 1 millimeter average deviation.
  EXPECT_NEAR(average_absolute_deviation, 0.000911, 1e-6);
  // About 6 millimeters maximum deviation.
  EXPECT_NEAR(max_absolute_deviation, 0.006099, 1e-6);
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

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake

#if 0

////////////////////////////////////////////////////////////////////////
//////////////////////// OLDER VERSION pre 2025-03-16 //////////////////
////////////////////////////////////////////////////////////////////////
// I refactored most of the following code into ":make_empress_field" library.
// If there's a need to reuse these code, please move them back into the
// anonymous namespace above.

/******************* archive ********************

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

Aabb CalcBoundingBox(const TriangleSurfaceMesh<double>& mesh_M) {
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

bool IsPointInTheBand(const Vector3d& p_MV, const MeshDistanceBoundary& sdf_M,
                      const double inner_offset, const double outer_offset,
                      int* count_positive, int* count_negative,
                      int* count_zero) {
  SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
      p_MV, sdf_M.tri_mesh(), sdf_M.tri_bvh(),
      std::get<FeatureNormalSet>(sdf_M.feature_normal()));
  if (d.signed_distance < 0) {
    ++(*count_negative);
  } else if (d.signed_distance > 0) {
    ++(*count_positive);
  } else {
    ++(*count_zero);
  }
  if (-inner_offset <= d.signed_distance && d.signed_distance <= outer_offset) {
    return true;
  } else {
    return false;
  }
}

// Check whether a tetrahedron touches the band between the outer offset and
// the inner offset distances, i.e., the implicit region of points Q where
// -inner_offset <= signed_distance(Q) <= +outer_offset.
//
// @param inner_offset specifies the implicit surface of the level set
//                     at signed distance = -inner_offset (inside).
// @param outer_offset specifies to the implicit surface of the level set
//                     at signed distance = +outer_offset (outside).
//
// @pre Both inner_offset and outer_offset are positive numbers.
//
// It's not enough to use signed-distance query at the vertices of the
// tetrahedra, especially for coarser grid resolution. Ideally we should check
// whether a tetrahedron intersects the band (at a vertex, in an edge, or in
// a face, or even (pedantically) enclosing the entire band). For now, it's too
// laborious, so we will just check some sampling points inside the tetrahedron.
bool IsTetInTheBand(const int tet, const VolumeMesh<double>& tetrahedral_mesh_M,
                    const double inner_offset, const double outer_offset,
                    const MeshDistanceBoundary& sdf_M) {
  DRAKE_THROW_UNLESS(inner_offset > 0);
  DRAKE_THROW_UNLESS(outer_offset > 0);

  // Check whether some points in the tetrahedron are in the band.
  // For simplicity, for now, we check only the four vertices and the edge's
  // midpoints.
  //
  // We also check whether some vertices have different sign. That's an
  // indicator that the tetrahedron cross the implicit surface of the zero-th
  // level set, i.e., the input surface mesh.

  // Four vertices.
  int count_positive = 0;
  int count_negative = 0;
  int count_zero = 0;  // Unlikely, but we want to be sure.
  for (int i = 0; i < 4; ++i) {
    const Vector3d& p_MV = tetrahedral_mesh_M.vertex(
        tetrahedral_mesh_M.tetrahedra()[tet].vertex(i));
    if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                         &count_positive, &count_negative, &count_zero)) {
      return true;
    }
    if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
      return true;
    }
  }

  // Six edges (3 + 2 + 1 + 0) : {(a,b) : 0 <= a < b < 4}.
  //  a: 0     | 1   | 2 | 3
  //  b: 1 2 3 | 2 3 | 3 | {}
  for (int a = 0; a < 4; ++a) {
    const Vector3d& p_MA = tetrahedral_mesh_M.vertex(
        tetrahedral_mesh_M.tetrahedra()[tet].vertex(a));
    for (int b = a + 1; b < 4; ++b) {
      const Vector3d p_AB_M = tetrahedral_mesh_M.edge_vector(tet, a, b);
      // V is the midpoint from A to B.
      const Vector3d p_MV = p_MA + p_AB_M / 2;
      if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                           &count_positive, &count_negative, &count_zero)) {
        return true;
      }
      if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
        return true;
      }
    }
  }

  // TODO(DamrongGuoy): Check the four face centers and one cell center?

  // TODO(DamrongGuoy): Find a more efficient way than bruteforce subsampling.
  const int sampling_resolution = 3;
  const Vector3d& p_MA = tetrahedral_mesh_M.vertex(0);
  const Vector3d& p_MB = tetrahedral_mesh_M.vertex(1);
  const Vector3d& p_MC = tetrahedral_mesh_M.vertex(2);
  const Vector3d& p_MD = tetrahedral_mesh_M.vertex(3);
  // For example, sampling_resolution = 3 means 20 sub-samplings.
  // {(a,b,c,d) : a + b + c + d = 3, 0 <= a,b,c,d <= 3 }
  //            10          +     5     +   3   + 1 = 20
  // a: 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 | 2 2 2 | 3
  // b: 0 0 0 0 1 1 1 2 2 3 | 0 0 1 1 2 | 0 0 1 | 0
  // c: 0 1 2 3 0 1 2 0 1 0 | 0 1 0 1 0 | 0 1 0 | 0
  // d: 3 2 1 0 2 1 0 1 0 0 | 2 1 1 0 0 | 1 0 0 | 0
  for (int a = 0; a <= sampling_resolution; ++a) {
    for (int b = 0; b <= sampling_resolution - a; ++b) {
      for (int c = 0; c <= sampling_resolution - a - b; ++c) {
        int d = sampling_resolution - a - b - c;
        DRAKE_THROW_UNLESS(0 <= d && d <= sampling_resolution);
        const Vector3d p_MV =
            (a * p_MA + b * p_MB + c * p_MC + d * p_MD) / sampling_resolution;
        if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                             &count_positive, &count_negative, &count_zero)) {
          return true;
        }
        if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
          return true;
        }
      }
    }
  }

  return false;
}

// Proof-of-concept automatic generation of EmPress mesh from an input
// watertight, manifold, self-intersecting-free triangle surface mesh.
// In the future, other input representations, e.g. Neural Implicit Signed
// Distance Field, should work too.
GTEST_TEST(EmPress, Genesis) {
  const MeshDistanceBoundary input_M(
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj")));
  EXPECT_EQ(input_M.tri_mesh().num_vertices(), 486);
  EXPECT_EQ(input_M.tri_mesh().num_triangles(), 968);

  const Aabb fitted_box_M = CalcBoundingBox(input_M.tri_mesh());
  EXPECT_TRUE(CompareMatrices(fitted_box_M.center(),
                              Vector3d{-0.000021, -0.000189, 0.040183}, 1e-6));
  EXPECT_TRUE(CompareMatrices(fitted_box_M.half_width(),
                              Vector3d{0.040288, 0.040262, 0.040388}, 1e-6));

  // The 10% expanded box's frame B is axis-aligned with the input mesh's
  // frame M.  Only their origins are different.
  // The Aabb stores its half-width vector, but the Box stores its "full
  // width", and hence the extra 2.0 factor below.
  const Box expanded_box_B(1.1 * 2.0 * fitted_box_M.half_width());
  const RigidTransformd X_MB(fitted_box_M.center());

  // Adjustable parameters, case by case.
  // Resolution of the background Cartesian grid generated in the expanded box.
  const double grid_resolution = 0.02;  // 2 centimeters
  // EmPress will include the levelset implicit surface at the +out_offest
  // distance outside the input surface.
  const double out_offset = 0.001;  // 1 millimeter.
  // EmPress will include the levelset implicit surface at the -in_offset
  // distance inside the input surface.
  const double in_offset = 0.001;  // 1 millimeter.

  const VolumeMesh<double> background_B =
      MakeBoxVolumeMesh<double>(expanded_box_B, grid_resolution);
  EXPECT_EQ(background_B.num_vertices(), 216);
  EXPECT_EQ(background_B.num_elements(), 750);

  // Translate/change to the common frame of reference.
  VolumeMesh<double> temp = background_B;
  temp.TransformVertices(X_MB);
  const VolumeMesh<double> background_M = temp;

  // Collect tetrahedra in the band between the inner offset and
  // the outer offset.
  std::vector<int> non_unique_tetrahedra;
  for (int tet = 0; tet < background_M.num_elements(); ++tet) {
    if (IsTetInTheBand(tet, background_M, in_offset, out_offset, input_M)) {
      non_unique_tetrahedra.push_back(tet);
    }
  }
  std::sort(non_unique_tetrahedra.begin(), non_unique_tetrahedra.end());
  auto last =
      std::unique(non_unique_tetrahedra.begin(), non_unique_tetrahedra.end());
  non_unique_tetrahedra.erase(last, non_unique_tetrahedra.end());
  // The tetrahedra are unique now; alias to a better name.
  const std::vector<int>& qualified_tetrahedra = non_unique_tetrahedra;

  EXPECT_LE(qualified_tetrahedra.size(), background_M.num_elements());

  // Vertex index from the background_M to the ImPress.
  std::unordered_map<int, int> old_to_new;
  std::vector<Vector3d> new_vertices;
  int count = 0;
  for (const int tet : qualified_tetrahedra) {
    for (int i = 0; i < 4; ++i) {
      const int v = background_M.tetrahedra()[tet].vertex(i);
      if (!old_to_new.contains(v)) {
        new_vertices.push_back(background_M.vertex(v));
        old_to_new[v] = count;
        ++count;
      }
    }
  }
  EXPECT_LE(count, background_M.num_vertices());
  ASSERT_EQ(count, new_vertices.size());

  std::vector<VolumeElement> new_tetrahedra;
  for (const int tet : qualified_tetrahedra) {
    new_tetrahedra.emplace_back(
        old_to_new.at(background_M.tetrahedra()[tet].vertex(0)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(1)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(2)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(3)));
  }

  VolumeMesh<double> mesh_EmPress_M{std::move(new_tetrahedra),
                                    std::move(new_vertices)};
  EXPECT_EQ(mesh_EmPress_M.num_vertices(), 167);
  EXPECT_EQ(mesh_EmPress_M.tetrahedra().size(), 568);

  WriteVolumeMeshToVtk("yellow_pepper_EmPress_mesh.vtk", mesh_EmPress_M,
                       "Tetrahedral Mesh for EmPress Embedded Pressure Field");

  // Does this optional step help or not?  Collect more tetrahedra that share
  // vertices with previous qualified tetrahedra.  It might mimic "snowflake"
  // condition in [Stuart2013]; they claimed "snowflake" help.
  //
  // [Stuart2013] D.A. Stuart, J.A. Levine, B. Jones, and A.W. Bargteil.
  // Automatic construction of coarse, high-quality tetrahedralizations that
  // enclose and approximate surfaces for animation.
  // Proceedings - Motion in Games 2013, MIG 2013, pages 191-199.
  //
}

// Create signed-distance field on the tetrahedral mesh (.vtk) with respect to
// the input surface mesh (.obj).
GTEST_TEST(EmPressMeshToEmPressSignedDistances, Generate) {
  const MeshDistanceBoundary input_M(
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj")));
  EXPECT_EQ(input_M.tri_mesh().num_vertices(), 486);
  EXPECT_EQ(input_M.tri_mesh().num_triangles(), 968);

  VolumeMesh<double> mesh_EmPress_M =
      ReadVtkToVolumeMesh(std::filesystem::path(FindResourceOrThrow(
          "drake/geometry/test/yellow_pepper_EmPress_mesh.vtk")));
  EXPECT_EQ(mesh_EmPress_M.num_vertices(), 167);
  EXPECT_EQ(mesh_EmPress_M.num_elements(), 568);

  std::vector<double> signed_distances;
  for (const Vector3d& tet_vertex : mesh_EmPress_M.vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        tet_vertex, input_M.tri_mesh(), input_M.tri_bvh(),
        std::get<FeatureNormalSet>(input_M.feature_normal()));
    signed_distances.push_back(d.signed_distance);
  }

  VolumeMeshFieldLinear<double, double> signed_distance_field_EmPress_M{
      std::move(signed_distances), &mesh_EmPress_M};
  WriteVolumeMeshFieldLinearToVtk(
      "yellow_pepper_EmPress_sdfield.vtk", "SignedDistance(meters)",
      signed_distance_field_EmPress_M, "EmbeddedSignedDistanceField");
}

// Measure "distance" between the zero-level set implicit surface and the
// original mesh. We use the machinery of contact surfaces to extract the
// zero-level set as a polygonal surface.
GTEST_TEST(MeasureDeviationOfEmPressSignedDistances, Verify) {
  const Mesh mesh_spec_with_sdfield{FindResourceOrThrow(
      "drake/geometry/test/yellow_pepper_EmPress_sdfield.vtk")};
  auto mesh_EmPress_M = std::make_unique<VolumeMesh<double>>(
      MakeVolumeMeshFromVtk<double>(mesh_spec_with_sdfield));
  EXPECT_EQ(mesh_EmPress_M->num_vertices(), 167);
  EXPECT_EQ(mesh_EmPress_M->num_elements(), 568);
  auto sdfield_EmPress_M =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeScalarValuesFromVtkMesh<double>(mesh_spec_with_sdfield),
          mesh_EmPress_M.get());
  const hydroelastic::SoftMesh compliant_hydro_EmPress_M{
      std::move(mesh_EmPress_M), std::move(sdfield_EmPress_M)};

  const Aabb bounding_box_M = CalcBoundingBox(compliant_hydro_EmPress_M.mesh());
  // Frame B and frame M are axis-aligned. Only their origins are different.
  const Box box_B(2.0 * bounding_box_M.half_width());
  const RigidTransformd X_MB(bounding_box_M.center());
  auto box_mesh_B = std::make_unique<VolumeMesh<double>>(
      MakeBoxVolumeMeshWithMa<double>(box_B));
  auto box_field_B = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeBoxPressureField<double>(box_B, box_mesh_B.get(),
          // Using hydroelastic_modulus = 1e-6 will try to track the
          // implicit surface of the zero level set of the pepper.
                                   1e-6));
  const hydroelastic::SoftMesh compliant_box_B{std::move(box_mesh_B),
                                               std::move(box_field_B)};

  // The kTriangle argument makes the level0 surface include centroids of
  // contact polygons for more checks.
  std::unique_ptr<ContactSurface<double>> level0_M =
      ComputeContactSurfaceFromCompliantVolumes(
          GeometryId::get_new_id(), compliant_box_B, X_MB,
          GeometryId::get_new_id(), compliant_hydro_EmPress_M,
          RigidTransformd::Identity(),
          HydroelasticContactRepresentation::kTriangle);
  EXPECT_EQ(level0_M->tri_mesh_W().num_vertices(), 3404);

  const MeshDistanceBoundary input_M(
      ReadObjToTriangleSurfaceMesh(FindResourceOrThrow(
          "drake/geometry/test/yellow_bell_pepper_no_stem_low.obj")));
  EXPECT_EQ(input_M.tri_mesh().num_vertices(), 486);
  EXPECT_EQ(input_M.tri_mesh().num_triangles(), 968);

  double average_absolute_deviation = 0;
  double max_absolute_deviation = 0;
  double min_absolute_deviation = std::numeric_limits<double>::max();
  std::vector<double> deviation_values;
  for (const Vector3d& level0_vertex : level0_M->tri_mesh_W().vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        level0_vertex, input_M.tri_mesh(), input_M.tri_bvh(),
        std::get<FeatureNormalSet>(input_M.feature_normal()));
    deviation_values.push_back(d.signed_distance);
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

  // About 0.1 micrometers minimum deviation, practically zero.
  EXPECT_NEAR(min_absolute_deviation, 1.189e-07, 1e-10);
  // About 1 millimeter average deviation.
  EXPECT_NEAR(average_absolute_deviation, 0.000911, 1e-6);
  // About 6 millimeters maximum deviation.
  EXPECT_NEAR(max_absolute_deviation, 0.006099, 1e-6);

  TriangleSurfaceMeshFieldLinear<double, double> deviation_field{
      std::move(deviation_values), &level0_M->tri_mesh_W()};
  WriteTriangleSurfaceMeshFieldLinearToVtk(
      "level0_deviation_field.vtk", "Deviation(meters)", deviation_field,
      "MeasureDeviationOfEmPressSignedDistances");

  // TODO(DamrongGuoy): Measure distance from the input surface's vertices to
  //  the level0 surface too.
}


******************* archive ********************/

//////////////////////////////////////////////////////////////////////
// O B S O L E T E   C O D E   B E L O W
//////////////////////////////////////////////////////////////////////
// From here down are obsolete code and recipe.  If you need it, move it back
// into the anonymous namespace above before compiling.
//
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

/******************* archive ********************
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
          // Using hydroelastic_modulus = 1e-6 will try to track the
          // implicit surface of the zero level set of the pepper.
          1e-6));
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

******************* archive ********************/

#endif
