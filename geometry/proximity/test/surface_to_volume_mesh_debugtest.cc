#include "drake/geometry/proximity/surface_to_volume_mesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/find_runfiles.h"
#include "drake/common/test_utilities/expect_no_throw.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

namespace fs = std::filesystem;

using Eigen::Vector3d;

GTEST_TEST(hot3d_117658302265452_RoLoPoly, Throw_ObjMeshOrientable_Init) {
  const fs::path filename =
      FindResourceOrThrow("drake/geometry/test/"
                          "hot3d_117658302265452_RoLoPoly.obj");
  const TriangleSurfaceMesh<double> surface =
      ReadObjToTriangleSurfaceMesh(filename);

  EXPECT_EQ(surface.num_vertices(), 294);
  EXPECT_EQ(surface.num_triangles(), 588);

  // Stack trace at the throw.
  // ObjMeshOrientable::Init()
  // ObjMeshOrientable::ObjMeshOrientable()
  // TetMesher::fillHole()
  // TetMesher::formTwoCavities()
  // TetMesher::faceRecovery()
  // TetMesher::initializeCDT()
  // TetMesher::compute()
  EXPECT_ANY_THROW(
      /*VolumeMesh<double> volume = */ ConvertSurfaceToVolumeMesh(surface));

  // EXPECT_EQ(volume.vertices().size(), 0);
  // EXPECT_EQ(volume.tetrahedra().size(), 03d);
}


}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake
