#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/geometry/proximity/detect_null_simplex.h"
#include "drake/geometry/proximity/make_mesh_field.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/volume_mesh_refiner.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Vector3d;

GTEST_TEST(BunnyMeshTest, VolumeMeshRefiner) {
  const std::string test_file =
      FindResourceOrThrow("drake/geometry/proximity/test/MeshImprovement"
                          "/bunny.vtk");
  VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);
  internal::WriteVolumeMeshFieldLinearToVtk(
      "bunny0_pressure.vtk", "pressure1",
      internal::MakeVolumeMeshPressureField(&test_mesh, 1e0),
      "Test pressure on coarse bunny");

  ASSERT_EQ(test_mesh.num_vertices(), 478);
  ASSERT_EQ(test_mesh.num_elements(), 1247);
  ASSERT_EQ(internal::DetectNullTetrahedron(test_mesh).size(), 337);
  ASSERT_EQ(internal::DetectNullInteriorTriangle(test_mesh).size(), 551);
  ASSERT_EQ(internal::DetectNullInteriorEdge(test_mesh).size(), 214);
  {
    internal::WriteVolumeMeshToVtk(
        "bunny_null_tetrahedron.vtk",
        internal::CreateSubMesh(test_mesh,
                                internal::DetectNullTetrahedron(test_mesh)),
        "Null tetrahedron in bunny mesh");
  }

  internal::VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.refine();
  internal::WriteVolumeMeshFieldLinearToVtk(
      "bunny2_pressure.vtk", "pressure1",
      internal::MakeVolumeMeshPressureField(&refined_mesh, 1e0),
      "Test pressure on refined bunny");
  internal::WriteVolumeMeshToVtk("bunny_fixed.vtk", refined_mesh,
                                 "bunny_fixed_by_VolumeMeshRefiner");

  EXPECT_EQ(internal::DetectNullTetrahedron(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorEdge(refined_mesh).size(), 0);
  EXPECT_EQ(refined_mesh.num_vertices(), 692);
  EXPECT_EQ(refined_mesh.num_elements(), 2302);

  // First algorithm:
  //   1. RefineTetrahedron
  //   2. RefineTriangle
  //   3. RefineEdge
  // refined_mesh.num_vertices() 1580
  // refined_mesh.num_elements() 6246

  // Second algorithm:
  //   1. RefineEdge
  //   2. RefineTriangle
  //   3. RefineTetrahedron
  // refined_mesh.num_vertices() 692
  // refined_mesh.num_elements() 2302
}

GTEST_TEST(BubbleMeshTest, VolumeMeshRefiner) {
  const std::string test_file =
      FindResourceOrThrow("drake/geometry/proximity/test/MeshImprovement"
                          "/bubble.vtk");
  VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);

  ASSERT_EQ(test_mesh.num_vertices(), 89);
  EXPECT_EQ(test_mesh.num_elements(), 215);
  EXPECT_EQ(internal::DetectNullTetrahedron(test_mesh).size(), 69);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(test_mesh).size(), 115);
  EXPECT_EQ(internal::DetectNullInteriorEdge(test_mesh).size(), 46);
  {
    internal::WriteVolumeMeshToVtk(
        "bubble_null_tetrahedron.vtk",
        internal::CreateSubMesh(test_mesh,
                                internal::DetectNullTetrahedron(test_mesh)),
        "Null tetrahedron in bubble mesh");
  }

  internal::VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.refine();
  EXPECT_EQ(refined_mesh.num_vertices(), 135);
  EXPECT_EQ(refined_mesh.num_elements(), 449);
  EXPECT_EQ(internal::DetectNullTetrahedron(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorEdge(refined_mesh).size(), 0);
  internal::WriteVolumeMeshToVtk("bubble_fixed.vtk", refined_mesh,
                                 "Fixed bubble mesh");
  internal::WriteVolumeMeshFieldLinearToVtk(
      "bubble_pressure.vtk", "pressure",
      internal::MakeVolumeMeshPressureField(&refined_mesh, 1e6),
      "Test pressure on fixed bubble with hydroelastic modulus 1e6 Pascals");
}

GTEST_TEST(TeddyMeshTest, VolumeMeshRefiner) {
  const std::string test_file =
      FindResourceOrThrow("drake/geometry/proximity/test/MeshImprovement"
                          "/teddy.vtk");
  VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);

  ASSERT_EQ(test_mesh.num_vertices(), 335);
  EXPECT_EQ(test_mesh.num_elements(), 859);
  EXPECT_EQ(internal::DetectNullTetrahedron(test_mesh).size(), 110);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(test_mesh).size(), 185);
  EXPECT_EQ(internal::DetectNullInteriorEdge(test_mesh).size(), 75);
  {
    internal::WriteVolumeMeshToVtk(
        "teddy_null_tetrahedron.vtk",
        internal::CreateSubMesh(test_mesh,
                                internal::DetectNullTetrahedron(test_mesh)),
        "Null tetrahedron in teddy mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "teddy_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&test_mesh, 1e6),
        "Test pressure on teddy with hydroelastic modulus 1e6 Pascals");
  }

  internal::VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.refine();
  EXPECT_EQ(refined_mesh.num_vertices(), 410);
  EXPECT_EQ(refined_mesh.num_elements(), 1207);
  EXPECT_EQ(internal::DetectNullTetrahedron(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorEdge(refined_mesh).size(), 0);
  {
    internal::WriteVolumeMeshToVtk("teddy_fixed.vtk", refined_mesh,
                                   "Fixed teddy mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "teddy_fixed_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&refined_mesh, 1e6),
        "Test pressure on fixed teddy with hydroelastic modulus 1e6 Pascals");
  }
}

GTEST_TEST(BoxMeshTest, VolumeMeshRefiner) {
  const std::string test_file =
      FindResourceOrThrow("drake/geometry/proximity/test/MeshImprovement"
                          "/box_diamondcubic.vtk");
  VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);

  EXPECT_EQ(test_mesh.num_vertices(), 60);
  EXPECT_EQ(test_mesh.num_elements(), 120);
  EXPECT_EQ(internal::DetectNullTetrahedron(test_mesh).size(), 32);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(test_mesh).size(), 44);
  EXPECT_EQ(internal::DetectNullInteriorEdge(test_mesh).size(), 12);
  {
    internal::WriteVolumeMeshToVtk(
        "box_diamondcubic_null_tetrahedron.vtk",
        internal::CreateSubMesh(test_mesh,
                                internal::DetectNullTetrahedron(test_mesh)),
        "Null tetrahedron in box_diamondcubic mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "box_diamondcubic_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&test_mesh, 1e6),
        "Test pressure on box_diamondcubic with "
        "hydroelastic modulus 1e6 Pascals");
  }

  internal::VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.refine();
  EXPECT_EQ(refined_mesh.num_vertices(), 76);
  EXPECT_EQ(refined_mesh.num_elements(), 208);
  EXPECT_EQ(internal::DetectNullTetrahedron(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorEdge(refined_mesh).size(), 0);
  {
    internal::WriteVolumeMeshToVtk("box_diamondcubic_fixed.vtk", refined_mesh,
                                   "Fixed box_diamondcubic mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "box_diamondcubic_fixed_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&refined_mesh, 1e6),
        "Test pressure on fixed box_diamondcubic with "
        "hydroelastic modulus 1e6 Pascals");
  }
}

GTEST_TEST(PepperMeshTest, VolumeMeshRefiner) {
  const std::string test_file =
      FindResourceOrThrow("drake/geometry/proximity/test/MeshImprovement"
                          "/pepper.vtk");
  VolumeMesh<double> test_mesh = internal::ReadVtkToVolumeMesh(test_file);

  EXPECT_EQ(test_mesh.num_vertices(), 667);
  EXPECT_EQ(test_mesh.num_elements(), 2519);
  EXPECT_EQ(internal::DetectNullTetrahedron(test_mesh).size(), 171);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(test_mesh).size(), 309);
  EXPECT_EQ(internal::DetectNullInteriorEdge(test_mesh).size(), 138);
  {
    internal::WriteVolumeMeshToVtk(
        "pepper_null_tetrahedron.vtk",
        internal::CreateSubMesh(test_mesh,
                                internal::DetectNullTetrahedron(test_mesh)),
        "Null tetrahedron in pepper mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "pepper_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&test_mesh, 1e6),
        "Test pressure on pepper with "
        "hydroelastic modulus 1e6 Pascals");
  }

  internal::VolumeMeshRefiner refiner(test_mesh);
  VolumeMesh<double> refined_mesh = refiner.refine();
  EXPECT_EQ(refined_mesh.num_vertices(), 805);
  EXPECT_EQ(refined_mesh.num_elements(), 3163);
  EXPECT_EQ(internal::DetectNullTetrahedron(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorTriangle(refined_mesh).size(), 0);
  EXPECT_EQ(internal::DetectNullInteriorEdge(refined_mesh).size(), 0);
  {
    internal::WriteVolumeMeshToVtk("pepper_fixed.vtk", refined_mesh,
                                   "Fixed pepper mesh");
    internal::WriteVolumeMeshFieldLinearToVtk(
        "pepper_fixed_pressure.vtk", "pressure",
        internal::MakeVolumeMeshPressureField(&refined_mesh, 1e6),
        "Test pressure on fixed pepper with "
        "hydroelastic modulus 1e6 Pascals");
  }
}

}  // namespace
}  // namespace geometry
}  // namespace drake

