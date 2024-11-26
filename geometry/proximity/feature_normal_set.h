#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/drake_throw.h"
#include "drake/common/sorted_pair.h"
#include "drake/common/ssize.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace geometry {

/* %FeatureNormalSet provides outward normal vectors at vertices and edges of a
 triangle surface mesh. The normal at a vertex is the angle-weighted average of
 the face normals of the triangles sharing the vertex. The normal at an edge is
 the equal-weight average of the face normals of the two triangles sharing the
 edge.

 The following paper shows that normals computed in this way are most suitable
 for the inside-outside test of a point closest to a vertex or an edge.

 J.A. Baerentzen; H. Aanaes. Signed distance computation using the angle
 weighted pseudonormal. IEEE Transactions on Visualization and Computer
 Graphics (Volume: 11, Issue: 3, May-June 2005).  */
class FeatureNormalSet {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(FeatureNormalSet);

  // Computes the normals at the vertices and the edges of the given surface
  // mesh expressed in frame M.
  //
  // @pre  The mesh is watertight and a closed manifold. Otherwise, the
  //       computed normals are incorrect.
  //
  // @returns the computed feature normals or an error message if the mesh has
  // extremely sharp features. They make calculation of the vertex normals and
  // edge normals too sensitive to numerical rounding.
  //     Think of the simplest possible knife model of two adjacent triangles
  // sharing a single edge. The two triangles would have a small angle between
  // them, creating a "sharp edge".  If the edge is too sharp, the mesh will be
  // rejected.  We have a generous tolerance -- the angle can be less than a
  // degree -- but it would be better to stay well away from impractically sharp
  // edges.
  //     There is an analogous situation with vertices, where a vertex
  // creates a pointy, needle-like region on the mesh. The skinny "needles"
  // have the same problem as the thin knife edges. Presence of these
  // "needles" may lead to the mesh being rejected.
  static std::variant<FeatureNormalSet, std::string> MaybeCreate(
      const TriangleSurfaceMesh<double>& mesh_M);

  FeatureNormalSet(const TriangleSurfaceMesh<double>& mesh_M) {
    FeatureNormalSet f = std::get<FeatureNormalSet>(MaybeCreate(mesh_M));
    vertex_normals_ = std::move(f.vertex_normals_);
    edge_normals_ = std::move(f.edge_normals_);
  }

  // Returns the normal at a vertex `v` as the angle-weighted average of face
  // normals of triangles sharing the vertex. The weight of a triangle is the
  // angle at vertex `v` in that triangle.
  //
  // The returned normal vector is expressed in the mesh's frame.
  //
  // @param v  the vertex index into the mesh
  //
  // @throws std::exception if v < 0 or v >= number of vertices in the mesh
  const Vector3<double>& vertex_normal(int v) const {
    DRAKE_THROW_UNLESS(0 <= v && v < ssize(vertex_normals_));
    return vertex_normals_[v];
  }

  // Returns the normal at an edge `uv` as the average normal from the two
  // triangles sharing the edge. Both triangles have equal weight.
  //
  // The returned normal vector is expressed in the mesh's frame.
  //
  // @param uv   the edge between vertices u and v, as represented by the
  //             sorted indices of vertices.
  //
  // @throws std::exception if the edge `uv` is not in the mesh.
  const Vector3<double>& edge_normal(const SortedPair<int>& uv) const {
    auto it = edge_normals_.find(uv);
    DRAKE_THROW_UNLESS(it != edge_normals_.end());
    return it->second;
  }

  const Vector3<double> edge_normal_of_vertex_pair(int u, int v) const {
    auto it = edge_normals_.find(SortedPair<int>(u, v));
    if (it != edge_normals_.end())
      return it->second;
    else
      return Vector3<double>::Zero();
  }

 private:
  FeatureNormalSet(
      std::vector<Vector3<double>>&& vertex_normals,
      std::unordered_map<SortedPair<int>, Vector3<double>>&& edge_normals)
      : vertex_normals_(std::move(vertex_normals)),
        edge_normals_(std::move(edge_normals)) {}

  std::vector<Vector3<double>> vertex_normals_{};
  std::unordered_map<SortedPair<int>, Vector3<double>> edge_normals_{};
};

}  // namespace geometry
}  // namespace drake
