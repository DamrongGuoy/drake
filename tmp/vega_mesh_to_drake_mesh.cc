#include "vega_mesh_to_drake_mesh.h"

#include <limits>

#include "tetMesher.h"

namespace drake {
namespace geometry {

using Eigen::Vector3d;

VolumeMesh<double> VegaFemTetMeshToDrakeVolumeMesh(
    const vegafem::TetMesh& vega_mesh) {
  const int num_vertices = vega_mesh.getNumVertices();
  const int num_elements = vega_mesh.getNumElements();

  std::vector<Vector3d> drake_vertices;
  drake_vertices.reserve(num_vertices);
  for (int v = 0; v < num_vertices; ++v) {
    const vegafem::Vec3d& v3d = vega_mesh.getVertex(v);
    drake_vertices.emplace_back(v3d[0], v3d[1], v3d[2]);
  }

  std::vector<VolumeElement> drake_elements;
  drake_elements.reserve(num_elements);
  for (int element = 0; element < num_elements; ++element) {
    const int * v4i = vega_mesh.getVertexIndices(element);
    drake_elements.emplace_back(v4i[0], v4i[1], v4i[2], v4i[3]);
  }

  return {std::move(drake_elements), std::move(drake_vertices)};
}

vegafem::ObjMesh DrakeTriangleSurfaceMeshToVegaObjMesh(
    const TriangleSurfaceMesh<double>& drake_mesh) {
  const int num_vertices = drake_mesh.num_vertices();
  const int num_triangles = drake_mesh.num_triangles();

  std::vector<vegafem::Vec3d> vertexPositions;
  vertexPositions.reserve(num_vertices);
  for (int v = 0; v < num_vertices; ++v) {
    const Vector3d& p_MV = drake_mesh.vertex(v);
    vertexPositions.emplace_back(p_MV.x(), p_MV.y(), p_MV.z());
  }

  std::vector<vegafem::Vec3i> triangles;
  triangles.reserve(num_triangles);
  for (int t = 0; t < num_triangles; ++t) {
    const SurfaceTriangle& tri = drake_mesh.element(t);
    triangles.emplace_back(tri.vertex(0), tri.vertex(1), tri.vertex(2));
  }

  vegafem::ObjMesh result(vertexPositions, triangles);
  return result;
}

}  // namespace geometry
}  // namespace drake