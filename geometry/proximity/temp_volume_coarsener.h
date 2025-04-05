#pragma once

#include <memory>
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

/* Return a new coarsen tetrahedral mesh supporting the field.

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
// It represents the A matrix in the class document of QEF below.
//
// In this implementation, SymMat4 has a private member M of type
// Eigen::Matrix4d and provide the conjugate gradient solver for the
// new minimizer (see ConjugateR() below).
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

 protected:
  explicit SymMat4(Eigen::Matrix4d M_in) : M_(std::move(M_in)) {}
  Eigen::Matrix4d M_;
};

// Representation of Quadric Error Metric function (similar to
// vtkUnstructuredGridQuadricDecimationQEF).  Instead of the standard
// representation as (A, b, c) in [Garland & Zhou]:
//
//                Q(x) = xᵀAx + 2bᵀx + c,
//
// we will use the alternative representation (A, p, e) with the
// minimum quadric error e and the minimizer p:
//
//                Q(x) = (x-p)ᵀA(x-p) + e,
//
// which is more numerically stable near the minimizer (x → p) and
// the minimum error e is readily available.  See Section 3.4 "Numerical
// Issues" in [Huy2007].
//
// @pre A is a 4x4 symmetric positive semi-definite matrix.
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
  // values are non-negative real numbers).
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
// Treatment of Triangulated Boundary Surface
//-------------------------------------------------------------------------

// Calculate the projection of the given point `p` to the 1-millimeter
// outer offset surface from the boundary.
//
// @return the Vector4d V of the new position V.(x,y,z) and the signed
// distance V.w = 0.001.
//
// @note For simplicity, in this implementation, we simply displace the
// nearest point on the boundary surface for one millimeter along the
// gradient of p.  It only works if `p` is already near the boundary
// surface.   In the adversarial case, for example, `p` is on the
// medial axis, the projection becomes unstable.
//
// @pre The point p is very near the boundary surface.
//
Eigen::Vector4d CalcProjectionTo1MmSurface(
    const Eigen::Vector3d& p, const MeshDistanceBoundary& boundary);

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

  static double CalcTetrahedronVolume(
      int tetrahedron_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices);

  // Return true if all incident tetrahedra of the given vertex,
  // excluding the ones also incident to the excluded vertex, have
  // positive volumes.
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

  const double kTinyVolume = 1e-9;  // 1-millimeter cube

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
  // Functions and data related to Quadric Error Metrics
  //--------------------------------------------------------

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
