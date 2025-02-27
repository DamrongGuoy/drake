#include "drake/geometry/proximity/surface_to_volume_mesh.h"

#include <limits>
#include <random>

#include "drake/tmp/vega_cdt.h"

namespace drake {
namespace geometry {

using Eigen::Vector3d;

VolumeMesh<double> ConvertSurfaceToVolumeMesh(
    const TriangleSurfaceMesh<double>& surface_mesh) {
  // In several unit tests (with OK_ThrowNullptr, OK_UndecidableCase, and
  // OK_InfiniteLoop), we observed geometric degeneracy, e.g., co-planar
  // points or co-spherical points. For example, before this fix, with
  // GTEST_TEST(cube_corners_Tet2Tri2Tet, OK_UndecidableCase), we got this
  // diagnostic message from delauanyMesher.cpp:
  //
  // vegafem::DelaunayMesher::DelaunayBall::contains(int newVtx):
  // (4,18,20,32)33
  // [1 -1 -1][-0.333333 -1 -1][0.333333 -1 1][0.333333 -1 0.333333]
  // [0.333333 -1 -0.333333]
  // 0 0 0
  //
  // The five points are co-planar with all Y==-1.  We are dealing with a
  // flat zero-volume tetrahedron here.  All coordinates are ±1 or
  // ±0.333333.  They look like this picture:
  //
  //                          ^ Z
  //                          |
  //                          +1      V20
  //                          |      (v[2])
  //                          |
  //                          +
  //                          |
  //                          |
  //                          +0.333  V32
  //                          |      (v[3])
  //                          |
  //  +-------+-------+-------+-------+-------+-------+---> X
  // -1              -0.333   | 0     0.333           1
  //                          |
  //                          +-0.333 V33
  //                          |      (newVtx)
  //                          |
  //                          +
  //                          |
  //                          |
  //                  V18     +-1                     V4
  //                 (v[1])   |                     (v[0])
  //                          |
  //
  //
  // Since the problem is geometric degeneracy (co-planar, co-spherical points).
  // We will hack it by randomly perturbing each vertex slightly to get out
  // of what Boris Delaunay called "special" system, as described in
  // Proposition 1 of his 1934 paper, translated from French to English:
  //
  // Proposition 1: If the system, E is special we can always make such an
  // infinitely small affine transformation of the space after which the
  // system E becomes non-special.
  //
  // B. Delaunay, Sur la sph`ere vide. A la m´emoire de Georges Vorono¨ı,
  // Bulletin de l’Acad´emie des Sciences de l’URSS. Classe des sciences
  // math´ematiques et na, 1934, Issue 6, 793–800.
  //
  // B. Delaunay, On the empty sphere. In memory of Georges Voronoı, Bulletin
  // of the Academy of Sciences of the USSR. Class of mathematical and natural
  // sciences, 1934, Issue 6, 793–800.
  //
  // https://www.mathnet.ru/links/fa0a27b42b442e6005f751a78e65d057/im4937.pdf
  //
  const int random_seed = 20250227;  // Instead of std::random_device rd;
  std::mt19937 gen(random_seed);     // Use const random_seed instead of rd()
  // We will use 10 micrometer random perturbation.
  std::uniform_real_distribution<> dis(-1e-5, 1e-5);
  std::vector<Vector3d> perturbed_vertices(surface_mesh.vertices());
  for (Vector3d& v : perturbed_vertices) {
    v = v + Vector3d(dis(gen), dis(gen), dis(gen));
  }
  TriangleSurfaceMesh<double> perturbed_surface(
      std::vector<SurfaceTriangle>(surface_mesh.triangles()),
      std::move(perturbed_vertices));

  VolumeMesh<double> volume = VegaCdt(perturbed_surface);

  // TODO(DamrongGuoy): Extract connectivity from `volume` and use the
  //  original vertex coordinates as much as we can.  Since the vertices in
  //  `volume` consist of both the perturbed input vertices and also a number
  //  of Stiener points, necessary for CDT existence, we might want to
  //  maintain an unordered map `std::unordered_map<Vector3d, int>`
  //  from (X,Y,Z) to the input vertex index. Then, we can scan
  //  `volume.vertices()` and replace each perturbed vertex with the original
  //  vertex.

  return volume;
}

}  // namespace geometry
}  // namespace drake
