#include "../tetMesher.h"
#include "../objMesh.h"
#include "../vega_mesh_to_drake_mesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"

namespace drake {
namespace geometry {
namespace {

namespace fs = std::filesystem;

GTEST_TEST(TetMesherTest, SimpleNonConvex) {
  const fs::path obj_file =
      FindResourceOrThrow("drake/geometry/test/non_convex_mesh.obj");

  vegafem::ObjMesh obj_mesh(obj_file.native());
  EXPECT_EQ(obj_mesh.getNumVertices(), 5);
  EXPECT_EQ(obj_mesh.getNumFaces(), 6);

  vegafem::TetMesher mesher;
  const double kUseThisForCoarsestMesh = std::numeric_limits<double>::max();
  vegafem::TetMesh* tet_mesh =
      mesher.compute(&obj_mesh, /*refinementQuality*/ kUseThisForCoarsestMesh);

  EXPECT_EQ(tet_mesh->getNumVertices(), 5);
  EXPECT_EQ(tet_mesh->getNumElements(), 3);
}


}  // namespace
}  // namespace geometry
}  // namespace drake