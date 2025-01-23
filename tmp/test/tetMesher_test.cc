#include "../tetMesher.h"
#include "../objMesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"

namespace drake {
namespace geometry {
namespace {

namespace fs = std::filesystem;

GTEST_TEST(TetMesherTest, InputObjFile) {
  const fs::path obj_file =
      FindResourceOrThrow("drake/geometry/test/non_convex_mesh.obj");

  vegafem::ObjMesh obj_mesh(obj_file.native());
  vegafem::TetMesher mesher;
  vegafem::TetMesh *tet_mesh;
  tet_mesh = mesher.compute(&obj_mesh);
  //   TetMesh * compute(
  //       ObjMesh * surfaceMesh, double refinementQuality = 1.1,
  //       double alpha = 2.0, double minDihedralAngle = 0.0,
  //       int maxSteinerVertices = -1, double maxTimeSeconds = -1.0);
}

}  // namespace
}  // namespace geometry
}  // namespace drake