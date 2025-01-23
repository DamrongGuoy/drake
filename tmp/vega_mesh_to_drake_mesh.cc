#include "vega_mesh_to_drake_mesh.h"

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

}  // namespace geometry
}  // namespace drake