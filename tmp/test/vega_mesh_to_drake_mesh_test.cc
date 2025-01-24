#include "../vega_mesh_to_drake_mesh.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {
namespace {

using Eigen::Vector3d;

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

GTEST_TEST(DrakeTriangleSurfaceMeshToVegaObjMeshTest, Tetrahedron) {
  // A four-triangle mesh of a standard tetrahedron.
  const TriangleSurfaceMesh<double> drake_surface_mesh{
      {// The triangle windings give outward normals.
       SurfaceTriangle{0, 2, 1}, SurfaceTriangle{0, 1, 3},
       SurfaceTriangle{0, 3, 2}, SurfaceTriangle{1, 2, 3}},
      {Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
       Vector3d::UnitZ()}};

  vegafem::ObjMesh obj_mesh =
      DrakeTriangleSurfaceMeshToVegaObjMesh(drake_surface_mesh);

  EXPECT_EQ(obj_mesh.getNumVertices(), 4);
  EXPECT_EQ(obj_mesh.getNumFaces(), 4);
}

GTEST_TEST(VegaCdtTest, Tetrahedron) {
  // A four-triangle mesh of a standard tetrahedron.
  const TriangleSurfaceMesh<double> drake_surface_mesh{
      {// The triangle windings give outward normals.
       SurfaceTriangle{0, 2, 1}, SurfaceTriangle{0, 1, 3},
       SurfaceTriangle{0, 3, 2}, SurfaceTriangle{1, 2, 3}},
      {Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY(),
       Vector3d::UnitZ()}};

  VolumeMesh<double> drake_tetrahedral_mesh = VegaCdt(drake_surface_mesh);

  // Expect a one-tetrahedron mesh with four vertices.
  EXPECT_EQ(drake_tetrahedral_mesh.num_vertices(), 4);
  EXPECT_EQ(drake_tetrahedral_mesh.num_elements(), 1);
}

}  // namespace
}  // namespace geometry
}  // namespace drake