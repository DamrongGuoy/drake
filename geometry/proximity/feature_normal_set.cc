#include "drake/geometry/proximity/feature_normal_set.h"

#include "drake/common/eigen_types.h"
#include "drake/math/rigid_transform.h"

namespace drake {
namespace geometry {

using Eigen::Vector3d;
using math::RigidTransformd;

std::variant<FeatureNormalSet, std::string> FeatureNormalSet::MaybeCreate(
    const TriangleSurfaceMesh<double>& mesh_M) {
  std::vector<Vector3<double>> vertex_normals(mesh_M.num_vertices(),
                                              Vector3d::Zero());
  std::unordered_map<SortedPair<int>, Vector3<double>> edge_normals;
  const std::vector<Vector3d>& vertices = mesh_M.vertices();
  // We can compute a tolerance for a knife edge based on a minimum dihedral
  // angle θ between adjacent faces using the following formula:
  // ‖nᵢ + nⱼ‖² = 2(1-cos(θ)), where nᵢ and nⱼ are the face normals of the
  // two adjacent faces.
  //     The case for "pointy" vertices is more complex and requires more
  // assumptions and guesswork. Empirically, the pointy vertices have the
  // same approximate magnitude as for the edge tolerance, but we won't get
  // into deriving the tolerance for simplicity's sake.
  constexpr double kToleranceSquaredNorm = 3e-6;

  // Accumulate data from the mesh. They are not normal vectors yet. We will
  // normalize them afterward.
  for (int f = 0; f < mesh_M.num_triangles(); ++f) {
    const Vector3d& face_normal = mesh_M.face_normal(f);
    const SurfaceTriangle& tri = mesh_M.triangles()[f];
    const int v[3] = {tri.vertex(0), tri.vertex(1), tri.vertex(2)};
    const Vector3d unit_edge_vector[3] = {
        (vertices[v[1]] - vertices[v[0]]).normalized(),
        (vertices[v[2]] - vertices[v[1]]).normalized(),
        (vertices[v[0]] - vertices[v[2]]).normalized()};

    // Accumulate angle*normal for each vertex of the triangle.
    for (int i = 0; i < 3; ++i) {
      const double angle =
          std::acos(unit_edge_vector[i].dot(-unit_edge_vector[(i + 2) % 3]));
      vertex_normals[v[i]] += angle * face_normal;
    }
    // Accumulate normal for each edge of the triangle.
    for (int i = 0; i < 3; ++i) {
      const auto edge = MakeSortedPair(v[i], v[(i + 1) % 3]);
      auto it = edge_normals.find(edge);
      if (it == edge_normals.end()) {
        edge_normals[edge] = face_normal;
      } else {
        it->second += face_normal;
        if (it->second.squaredNorm() < kToleranceSquaredNorm) {
          return "FeatureNormalSet: Cannot compute an edge normal because "
                 "the two triangles sharing the edge make a very sharp edge.";
        }
        it->second.normalize();
      }
    }
  }
  for (auto& v_normal : vertex_normals) {
    if (v_normal.squaredNorm() < kToleranceSquaredNorm) {
      return "FeatureNormalSet: Cannot compute a vertex normal because "
             "the triangles sharing the vertex form a very pointy needle.";
    }
    v_normal.normalize();
  }

  return FeatureNormalSet(std::move(vertex_normals), std::move(edge_normals));
}

}  // namespace geometry
}  // namespace drake
