#include "../vega_mesh_to_drake_mesh.h"
#include "../tetMesher.h"
#include "../objMesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {
namespace {

namespace fs = std::filesystem;
using Eigen::Vector3d;

vegafem::TetMesh MakeVegaFemTetMesh() {
  const fs::path obj_file =
      FindResourceOrThrow("drake/geometry/test/non_convex_mesh.obj");

  vegafem::ObjMesh obj_mesh(obj_file.native());
  vegafem::TetMesher mesher;
  // TetMesh * compute(
  //     ObjMesh * surfaceMesh, double refinementQuality = 1.1,
  //     double alpha = 2.0, double minDihedralAngle = 0.0,
  //     int maxSteinerVertices = -1, double maxTimeSeconds = -1.0);

  vegafem::TetMesh result(*mesher.compute(&obj_mesh));
  return result;
}

GTEST_TEST(VegaFemTetMeshToDrakeVolumeMeshTest, Simple) {
  // A mesh of two tetrahedra with five vertices.
  const vegafem::TetMesh vega_mesh(
      std::vector<vegafem::Vec3d>{
          {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}},
      std::vector<vegafem::Vec4i>{{0, 1, 2, 3}, {1, 2, 3, 4}});
  const int vega_num_vertices = vega_mesh.getNumVertices();
  const int vega_num_elements = vega_mesh.getNumElements();
  // Sanity check that we set up the input correctly.
  ASSERT_EQ(vega_num_vertices, 5);
  ASSERT_EQ(vega_num_elements, 2);

  VolumeMesh<double> drake_mesh = VegaFemTetMeshToDrakeVolumeMesh(vega_mesh);

  ASSERT_EQ(drake_mesh.num_vertices(), vega_num_vertices);
  ASSERT_EQ(drake_mesh.num_elements(), vega_num_elements);
  // These checks are somewhat redundant; better be safe than sorry.
  for (int v = 0; v < vega_num_vertices; ++v) {
    const Vector3d& drake_V = drake_mesh.vertex(v);
    EXPECT_EQ(drake_V.x(), vega_mesh.getVertex(v)[0]);
    EXPECT_EQ(drake_V.y(), vega_mesh.getVertex(v)[1]);
    EXPECT_EQ(drake_V.z(), vega_mesh.getVertex(v)[2]);
  }
  for (int element = 0; element < vega_num_elements; ++element) {
    const VolumeElement& drake_element = drake_mesh.element(element);
    EXPECT_EQ(drake_element.vertex(0), vega_mesh.getVertexIndices(element)[0]);
    EXPECT_EQ(drake_element.vertex(1), vega_mesh.getVertexIndices(element)[1]);
    EXPECT_EQ(drake_element.vertex(2), vega_mesh.getVertexIndices(element)[2]);
    EXPECT_EQ(drake_element.vertex(3), vega_mesh.getVertexIndices(element)[3]);
  }
}

}  // namespace
}  // namespace geometry
}  // namespace drake