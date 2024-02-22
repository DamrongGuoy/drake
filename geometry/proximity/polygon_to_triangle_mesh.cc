#include "drake/geometry/proximity/polygon_to_triangle_mesh.h"

#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Vector3d;

TriangleSurfaceMesh<double> MakeTriangleFromPolygonMesh(
    const PolygonSurfaceMesh<double>& poly_mesh) {
  std::vector<Vector3<double>> vertices;
  vertices.reserve(poly_mesh.num_vertices());
  for (int v = 0; v < poly_mesh.num_vertices(); ++v) {
    vertices.push_back(poly_mesh.vertex(v));
  }

  std::vector<SurfaceTriangle> tris;
  // According to Euler's formula, the maximum number of possible triangles.
  tris.reserve(2 * poly_mesh.num_vertices() - 4);
  for (int p = 0; p < poly_mesh.num_faces(); ++p) {
    // Create a triangle fan around vertex 0.
    const SurfacePolygon& poly = poly_mesh.element(p);
    const int v0 = poly.vertex(0);
    int v1 = poly.vertex(1);
    for (int v = 2; v < poly.num_vertices(); ++v) {
      const int v2 = poly.vertex(v);
      tris.emplace_back(v0, v1, v2);
      v1 = v2;
    }
  }

  return TriangleSurfaceMesh<double>(std::move(tris), std::move(vertices));
}

TriangleSurfaceMesh<double> MakeTriangleFromPolygonMeshWithCentroids(
    const PolygonSurfaceMesh<double>& poly_mesh) {
  std::vector<Vector3<double>> vertices;
  const int num_poly_mesh_vertices = poly_mesh.num_vertices();
  const int num_polygons = poly_mesh.num_faces();
  vertices.reserve(num_poly_mesh_vertices + num_polygons);
  for (int v = 0; v < num_poly_mesh_vertices; ++v) {
    vertices.push_back(poly_mesh.vertex(v));
  }
  int num_triangles = 0;
  for (int p = 0; p < num_polygons; ++p) {
    vertices.push_back(poly_mesh.element_centroid(p));
    num_triangles += poly_mesh.element(p).num_vertices();
  }

  std::vector<SurfaceTriangle> tris;
  tris.reserve(num_triangles);
  for (int p = 0; p < num_polygons; ++p) {
    // Create a triangle fan around the centroid.
    const SurfacePolygon& poly = poly_mesh.element(p);
    // Say the polygon has 6 vertices (vertex v0 to v5).
    // i 5 0 1 2 3 4
    // j 0 1 2 3 4 5
    int i = poly.num_vertices() - 1;
    for (int j = 0; j < poly.num_vertices(); ++j) {
      const int vi = poly.vertex(i);
      const int vj = poly.vertex(j);
      const int v_centroid = num_poly_mesh_vertices + p;
      tris.emplace_back(v_centroid, vi, vj);
      i = j;
    }
  }

  return TriangleSurfaceMesh<double>(std::move(tris), std::move(vertices));
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
