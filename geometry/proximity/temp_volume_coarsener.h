#pragma once

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "drake/common/eigen_types.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_mesh_refiner.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

/* Return a new coarsen tetrahedral mesh supporting the field. This version
 uses vtkUnstructuredGridQuadricDecimation.  It's the baseline implementation.
 The rest of this file is our own implementation.

 @param[in] sdf_M      signed distance field on a support mesh.
 @param[in] fraction   a number between 0 and 1 that control how aggressive
                       the coarsening process is. The target number of
                       tetrahedra equals the `fraction` times the number of
                       the given tetrahedra.  */
VolumeMesh<double> TempCoarsenVolumeMeshOfSdField(
    const VolumeMeshFieldLinear<double, double>& sdf_M, double fraction);

//-------------------------------------------------------------------------
// Quadric Error Metric Functions
//-------------------------------------------------------------------------

// This class is similar to vtkUnstructuredGridQuadricDecimationSymMat4.
//
// In this implementation, SymMat4 has a private member M_ of type
// Eigen::Matrix4d. This class is an adapter of Matrix4d to make sure
// M_ always remains a symmetric matrix, from constructors to all algebraic
// operations.
//
// Eigen doesn't have a direct representation of a symmetric matrix.
// For a quick prototype, we store and operate the entire 16 coefficients.
// In the future, for efficiency, we should consider Eigen's views
// for Triangular and Self-adjoint(symmetric) matrices. See
// https://eigen.tuxfamily.org/dox/group__QuickRefPage.html#QuickRef_DiagTriSymm
//
// N.B.: Do not expose M_ since we might change to a quicker implementation
// in the future.
//
class SymMat4 {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SymMat4);

  SymMat4() : M_(Eigen::Matrix4d::Zero()) {}

  static SymMat4 FromOuterProductOfVector4d(const Eigen::Vector4d& n) {
    const Eigen::Matrix4d M = n * n.transpose();
    return SymMat4(M);
  }
  static SymMat4 Zero() { return SymMat4(Eigen::Matrix4d::Zero()); }
  static SymMat4 Identity() { return SymMat4(Eigen::Matrix4d::Identity()); }
  SymMat4 operator+(const SymMat4& A1) const { return SymMat4(M_ + A1.M_); }
  SymMat4 operator-(const SymMat4& A1) const { return SymMat4(M_ - A1.M_); }
  Eigen::Vector4d operator*(const Eigen::Vector4d& v) const { return M_ * v; }
  SymMat4& operator*=(const double& f) {
    M_ *= f;
    return *this;
  }
  SymMat4& operator/=(const double& f) {
    M_ /= f;
    return *this;
  }
  SymMat4& operator+=(const SymMat4& A1) {
    M_ += A1.M_;
    return *this;
  }
  double trace() const { return M_(0, 0) + M_(1, 1) + M_(2, 2) + M_(3, 3); }
  // Constant access.
  const Eigen::Matrix4d& Mat4d() const { return M_; }

 protected:
  explicit SymMat4(Eigen::Matrix4d M_in) : M_(std::move(M_in)) {}
  Eigen::Matrix4d M_;
};

// This representation of Quadric Error Metric function is similar to
// vtkUnstructuredGridQuadricDecimationQEF.  Instead of the standard
// representation as Q[A, b, c] in [Garland & Zhou]:
//
//                Q(x) = xᵀAx + 2bᵀx + c,
//
// we will use the alternative representation Q[A, p, e] with the
// minimum quadric error e and the minimizer p:
//
//                Q(x) = (x-p)ᵀA(x-p) + e,
//
// which is more numerically stable near the minimizer (x → p) and
// the minimum error e is readily available.  See Section 3.4 "Numerical
// Issues" in [Huy2007].
//
// The downside of this representation is that combining two QEF Q1 and Q2
// requires solving a linear system to determine the combined minimizer and
// then recalculating the new minimum error. (See CalcCombinedMinimizer() and
// CalcCombinedMinError().)  Combining two standard-representation
// Q1[A1, b1, c1] and Q2[A2, b2, c2] is simply (Q1+Q2)[A1+A2, b1+b2, c1+c2];
// however, we will also need to solve a linear system to determine the new
// location to which the edge contraction goes. Depending on the final
// scheduling algorithm (still under construction), we might change this
// design choice.
//
// [Huy2007] Huy, Vo; Callahan, Steven; Lindstrom, Peter; Pascucci, Valerio;
// and Silva, Claudio. (2007). Streaming Simplification of Tetrahedral Meshes.
// IEEE transactions on visualization and computer graphics.
// 13. 145-55. 10.1109/TVCG.2007.21.
//
// [Garland & Zhou] M. Garland and Y. Zhou. (2005). Quadric-Based
// Simplification in Any Dimension. ACM Trans. Graphics, vol. 24, no. 2,
// Apr. 2005.
//
struct QEF {
  QEF() : A(SymMat4::Zero()), p(0, 0, 0, 0), e(0) {}

  static QEF Zero() { return {SymMat4::Zero(), Eigen::Vector4d::Zero(), 0}; }

  static QEF Combine(const QEF& Q1, const QEF& Q2) {
    QEF Q;
    Q.A = Q1.A + Q2.A;
    Q.p = CalcCombinedMinimizer(Q1, Q2);
    Q.e = CalcCombinedMinError(Q1, Q2, Q.p);
    return Q;
  }

  double Evaluate(const Eigen::Vector4d& x) const {
    return (x - p).dot(A * (x - p)) + e;
  }

  // Experimental feature to move the minimizer. For example, when we
  // contract a boundary edge between two vertices, the combined minimizer is
  // not on the boundary surface. In that case, we might want to move the
  // minimizer to the boundary surface.
  QEF WithMinimizerMoveTo(const Eigen::Vector4d& new_p) const {
    double new_e = (new_p - p).dot(A * (new_p - p)) + e;
    QEF Q;
    Q.A = A;
    Q.p = new_p;
    Q.e = new_e;
    return Q;
  }

  SymMat4 A;
  // The minimizer.
  Eigen::Vector4d p;
  // The minium quadric error ε.
  double e;

 private:
  QEF(SymMat4 A_in, Eigen::Vector4d p_in, const double& e_in)
      : A(std::move(A_in)), p(std::move(p_in)), e(e_in) {}

  // Find the minimizer p for the combined QEF of Q1 and Q2 (used in edge
  // contraction).
  //
  // Combining the standard forms Q₁(A₁,b₁,c₁) and Q₂(A₂,b₂,c₂) into
  // Q(A,b,c) is straightforward:
  //      A = A₁+A₂, b = b₁+b₂, c = c₁+c₂.
  // However, our representations of Q₁(A₁,p₁,e₁) and Q₂(A₂,p₂,e₂)
  // needs the combined minimizer p as a solution to the 4x4 linear
  // system:
  //      Solve for p in (A₁+A₂)p = (A₁p₁ + A₂p₂).
  //
  // Notice that A₁, A₂ are positive semi-definite symmetric matrices (eigen
  // values are non-negative real numbers). (A₁+A₂) are not always invertible.
  //
  // @return the minimizer of the combined QEF of Q1 and Q2.
  static Eigen::Vector4d CalcCombinedMinimizer(const QEF& Q1, const QEF& Q2);

  // Given Q1(A1,p1,e1), Q2(A2,p2,e2), and the minimizer p of their
  // combined QEF, calculate their combined error e:
  //
  //       e = e₁ + e₂ + (p-p₁)ᵀA₁(p-p₁) + (p-p₂)ᵀA₂(p-p₂)
  //
  static double CalcCombinedMinError(const QEF& Q1, const QEF& Q2,
                                     const Eigen::Vector4d& p);
};

//-------------------------------------------------------------------------
// Treatment of Boundary
//-------------------------------------------------------------------------

// Calculate the projection of the given point `p` to a small-offset surface
// from the boundary.
//
// @return the Vector4d V of the new position V.(x,y,z) and the signed
// distance V.w = offset distance.
//
// @note For simplicity, in this implementation, we simply displace the
// nearest point on the boundary surface for the specified offset distance
// along the gradient of p.  It only works if `p` is already near the
// boundary surface.   In the adversarial case, for example, `p` is on the
// medial axis, the projection becomes unstable.
//
// @pre The point p is very near the boundary surface. The offset_distance
//      is near zero (it could be positive or negative).
//
Eigen::Vector4d CalcProjectionToOffsetSurface(
    const Eigen::Vector3d& p, const MeshDistanceBoundary& boundary,
    const double offset_distance);

//-------------------------------------------------------------------------
// Main VolumeMeshCoarsener
//-------------------------------------------------------------------------

class VolumeMeshCoarsener : VolumeMeshRefiner {
 public:
  VolumeMeshCoarsener(const VolumeMeshFieldLinear<double, double>& sdfield_M,
                      const TriangleSurfaceMesh<double>& original_M);

  VolumeMesh<double> coarsen(double fraction);

 protected:
  // Contract the edge between the v0-th vertex and the v1-th vertex to the
  // `new_position` with `new_scalar` value.
  //
  // The v1-th vertex will lose all its incident tetrahedra.
  // The v0-th vertex will gain all incident tetrahedra from v1-th vertex,
  // except the "deleted" tetrahedra sharing both vertices.
  //
  // @throw  if IsEdgeContractible() is false.
  //
  void ContractEdge(int v0, int v1, const Eigen::Vector3d& new_position,
                    double new_scalar);

  // Is the edge(v0,v1) contractible to the new position with new scalar
  // field value.  This is a precondition before calling ContractEdge().
  bool IsEdgeContractible(int v0, int v1, const Eigen::Vector3d& new_position,
                          double new_scalar);

  // Is the vertex movable to the new position without a negative-volume
  // tetrahedron?  We use this function when we perturb a boundary vertex to
  // a small-offset surface to check that it will not create a bad
  // tetrahedron.
  bool IsVertexMovable(int vertex, const Eigen::Vector3d& new_position);

  static double CalcTetrahedronVolume(
      int tetrahedron_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices);

  double CalcMinIncidentTetrahedronVolume(int vertex) const;

  // Return true if all incident tetrahedra of the given vertex,
  // excluding the ones incident to the excluded vertex, have
  // positive volumes.
  //
  // Before the edge contraction of vertices v0 and v1, the caller is supposed
  // to temporarily change the entries in `vertices` for v0 (`vertex_index`)
  // and v1 (`exclude_vertex_index`) to the common new position (and edge
  // v0v1 becomes a zero-length edge). Then, this function can check whether
  // all the morphed tetrahedra incident to v0, excluding the zero-volume
  // tetrahedra on the zero-length edge v0v1, will continue to have positive
  // volumes.
  static bool AreAllMorphedTetrahedraPositive(
      int vertex_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices,
      const std::vector<std::vector<int>>& vertex_to_tetrahedra,
      int exclude_vertex_index, double kTinyVolume);

  // Create a new mesh without unused vertices. Renumber vertex indices used
  // by the tetrahedra.
  VolumeMesh<double> CompactMesh(
      const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3d>& vertices) const;

  //--------------------------------------------------------
  // Data related to combinatorics and discrete meshes and
  // interpolated signed distances.
  //--------------------------------------------------------

  // const double kTinyVolume_ = 1e-12;  // 0.1-millimeter cube
  const double kTinyVolume_ = std::numeric_limits<double>::lowest();

  // signed_distances[i] := the signed distance value of the i-th vertex.
  // As we perform edge contraction, the value of `signed_distances[i]` can
  // change since we might move the i-th vertex during edge contraction.
  // Or `signed_distances[i]` becomes irrelevant because the i-th vertex
  // was deleted by the edge contraction.
  std::vector<double> signed_distances_;

  // "is_vertex_deleted_[i]==true" means vertices_[i] was marked for deletion
  //                               from an edge contraction.
  std::vector<bool> is_vertex_deleted_;

  // "is_tet_deleted[i]==true" means tetrahedra_[i] was marked for deletion
  //                           from an edge contraction.
  std::vector<bool> is_tet_deleted_;
  int num_tet_deleted_{0};
  // "is_tet_morphed[i]==true"  means tetrahedra_[i] was changed by edge
  //                            contraction. One of its vertex has changed to
  //                            a new vertex or new position.
  std::vector<bool> is_tet_morphed_;

  // "boundary_to_volume_[u]==v" means support_boundary_mesh_.vertex(u)
  // corresponds to vertices_[v].
  std::vector<int> boundary_to_volume_;
  // "volume_to_boundary_.at(u)==v" means vertices_[u] corresponds to
  // support_boundary_mesh.vertex(v).  Since it's a map, some vertices_[i]
  // may not have a corresponding support_boundary_mesh.vertex(j), i.e.,
  // volume_to_boundary_.contains(i) == false.
  std::unordered_map<int, int> volume_to_boundary_;
  TriangleSurfaceMesh<double> support_boundary_mesh_{
      {{0, 1, 2}},
      {Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
       Eigen::Vector3d::UnitY()}};

  // For signed-distance query to the original mesh during optimization.
  const MeshDistanceBoundary original_boundary_;

  //--------------------------------------------------------
  // For priority queue
  //--------------------------------------------------------

  // Light-weight representation of an edge as a sorted pair of vertex indices.
  // An Edge edge_uv with [u,v] = [edge.first(), edge.second()] means the
  // edge_uv is between the u-th vertex and the v-th vertex of vertices_[]
  // (inherited from VolumeMeshRefiner).
  using Edge = SortedPair<int>;

  // We cache the QEF of the potential edge contraction here, so we don't have
  // to recalculate it again when we inspect an edge shared by multiple
  // tetrahedra.
  std::map<Edge, QEF> edge_QEFs_;

  // Type of the stored elements in our priority queue. We want to rank
  // the tetrahedra by the cost of their edge contraction.
  //
  // It is the current design choice to store tetrahedra, not edges, in
  // the priority queue. The reason is that our main data structures are
  // based on tetrahedra not edges.  We update tetrahedra_[] and associated
  // data when we change the mesh.  We don't want to maintain another
  // data structure for edges yet.
  struct TetrahedronCost {
    // Index of the tetrahedron. For example, is_tet_deleted_.at(tet) informs
    // whether the tetrahedron was deleted already, and tetrahedra_[tet] is
    // the VolumeElement representation of the tetrahedron (tetrahedra_ is
    // an inherited member from VolumeMeshRefiner).
    int tet;

    // The minimum cost of contracting an edge of the tetrahedron. It is the
    // minimum of the edge-contraction errors of the six edges of the
    // tetrahedron.
    //
    //  cost = min {
    //           Quadric Error Measure of contracting Edge(vi, vj);
    //           where vi = tetrahedra_[tet].vertex(i), and
    //           vj = tetrahedra_[tet].vertex(j)
    //           : 0 <= i < j < 4
    //         }
    double cost;

    bool operator>(const TetrahedronCost& other) const {
      return cost > other.cost;
    }
  };

  // It has a side effect that the map edge_QEFs_ is updated if the QEFs of
  // some edges of the tetrahedron wasn't calculated before.
  //
  // @param[out] min_edge an edge with the minimum error for contraction.
  double CalcOrFetchTetrahedronMinEdgeCost(int tet, Edge* min_edge);

  // By default, the priority_queue puts the highest-priority element at
  // the top.  However, we want the lowest cost at the top, so we use the
  // std::greater<>.
  std::priority_queue<TetrahedronCost, std::vector<TetrahedronCost>,
                      std::greater<TetrahedronCost>>
      tetrahedron_queue_;

  // Assume the Quadric Error Metric is already initialized at the vertices.
  void InitializeTetrahedronQueue();

  //--------------------------------------------------------
  // Visual debugging facilities
  //--------------------------------------------------------
 protected:
  VolumeMesh<double> DebugTetrahedraOfVertex(int v0) const;
  VolumeMesh<double> DebugTetrahedraOfBothVertex(int v0, int v1) const;

  void WriteTetrahedraOfVertex(int v0, const std::string& file_name);
  void WriteTetrahedraOfFirstExcludeSecond(int first_vertex, int second_vertex,
                                           const std::string& file_name);
  void WriteTetrahedraOfBothVertices(int first_vertex, int second_vertex,
                                     const std::string& file_name);

  void WriteTetrahedraBeforeEdgeContraction(
      int v0, int v1, const std::string& prefix_file_name);
  // This version accepts the tetrahedra on the edge v0v1 before the edge
  // contraction happens.  The parameter v0 and v1 only go into the output
  // file name without bearing on the output mesh.
  void WriteSavedTetrahedraBeforeEdgeContraction(
      int v0, int v1, const VolumeMesh<double>& tetrahedra_on_edge_v0_v1,
      const std::string& prefix_file_name);
  // This version accepts the tetrahedra sharing the vertex v before the edge
  // contraction happens. The parameter v only go into the output file name
  // without bearing on the mesh.
  void WriteSavedTetrahedraOfVertexBeforeEdgeContration(
      int v, const VolumeMesh<double>& tetrahedra_on_vertex,
      const std::string& prefix_file_name);

  // After edge contraction, v0 gains all tetrahedra from v1.
  // The parameter v1 only goes into the output file name without bearing
  // on the incident tetrahedra.
  void WriteTetrahedraAfterEdgeContraction(int v0, int v1,
                                           const std::string& prefix_file_name);

 public:
  // Callers can use this function for debugging.
  // - Skip near-zero volume tetrahedra, and
  // - Flip each non-near-zero negative-volume tetrahedron
  //   to positive; however, it will create overlapping tetrahedra.
  static VolumeMesh<double> HackNegativeToPositiveVolume(
      const VolumeMesh<double>& mesh,
      int* num_negative_volume_tetrahedra = nullptr);

  //--------------------------------------------------------
  // Functions and data related to Quadric Error Metrics
  //--------------------------------------------------------

 protected:
  // Update QEF of the four vertices of the tet-th tetrahedron.
  //
  // The fundamental quadric matrix A of the tetrahedron is the
  // outer product of the (column) Vector4d n divided by the volume of
  // the tetrahedron.
  //
  //         A = n nᵀ / volume
  //
  // The vector n is the generalized cross product of the three edge vectors
  // V01, V02, V03 (V0i = Vi - V0) of the tetrahedron + scalar field in
  // four dimensions. See the determinant formula in Section 3.4
  // "Numerical Issues" of [Huy2007].
  //
  //            | ∂x ∂y ∂z ∂w |
  //   n =  det | <-- V01 --> | ; ∂x,∂y,∂z,∂w are the basis vectors in ℝ⁴.
  //            | <-- V02 --> |
  //            | <-- V03 --> |
  //
  // Each coefficient of n has its unit in cubic meters.
  // Each coefficient of A has its unit in cubic meters.
  //
  // Finally, the contribution of A into each of the four vertices is A/4.
  //
  void UpdateVerticesQuadricsFromTet(int tet);

  // Reference implementation is in
  // vtkUnstructuredGridQuadricDecimationFace::UpdateQuadric().
  //
  // N.B. Do this after calling UpdateVerticesQuadricsFromTet() for all
  // tetrahedra.
  void UpdateVerticesQuadricsFromBoundaryFace(int boundary_tri);

  void InitializeVertexQEFs();

  // vertex_Qs[i] = Quadric error metric at the i-th vertex.
  std::vector<QEF> vertex_Qs_;
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
