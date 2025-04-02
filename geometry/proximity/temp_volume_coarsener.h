#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Representation of Quadric Error Metric function (similar to
// vtkUnstructuredGridQuadricDecimationQEF).  Instead of the standard
// representation as (A, b, c) in [Garland & Zhou]:
//
//                Q(x) = xᵀAx - 2bᵀx + c,
//
// we will use the alternative representation (A, p, e) with the
// minimum quadric error e and the minimizer p:
//
//                Q(x) = (x-p)ᵀA(x-p) + e,
//
// which is more numerically stable near the minimizer (x → p) and
// the minimum error e is readily available.  See Section 3.4 "Numerical
// Issues" in:
//
// [Huy2007] Huy, Vo; Callahan, Steven; Lindstrom, Peter; Pascucci, Valerio; and
// Silva, Claudio. (2007). Streaming Simplification of Tetrahedral Meshes.
// IEEE transactions on visualization and computer graphics.
// 13. 145-55. 10.1109/TVCG.2007.21.
//
// [Garland & Zhou] M. Garland and Y. Zhou. (2005). Quadric-Based
// Simplification in Any Dimension. ACM Trans. Graphics, vol. 24, no. 2,
// Apr. 2005.
//
struct QEF {
  QEF() : p(0, 0, 0, 0), e(0) {}

  // This class-internal type `QEF::SymMat4` is similar to
  // vtkUnstructuredGridQuadricDecimationSymMat4.
  // It represents the A matrix in the class document of QEF above.
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
    SymMat4() : M_(Eigen::Matrix4d::Zero()) {}

    static SymMat4 FromOuterProductOfVector4d(const Eigen::Vector4d& n) {
      const Eigen::Matrix4d M = n * n.transpose();
      return SymMat4(M);
    }

    static SymMat4 Zero() { return SymMat4(Eigen::Matrix4d::Zero()); }

    SymMat4 operator+(const SymMat4& A1) const { return SymMat4(M_ + A1.M_); }
    SymMat4 operator-(const SymMat4& A1) const { return SymMat4(M_ - A1.M_); }
    SymMat4 operator*=(const double& f) {
      M_ *= f;
      return *this;
    }
    SymMat4 operator+=(const SymMat4& A1) {
      M_ += A1.M_;
      return *this;
    }
    Eigen::Vector4d operator*(const Eigen::Vector4d& v) const { return M_ * v; }

    // Conjugate gradient solver to update the symmetric matrix.
    // See the algorithm in Fig. 5 of [Huy2007].
    //
    // For an example usage, see the implementation of the constructor
    // QEF(const QEF&, const QEF&, const Vector4d&).
    //
    // @param[in,out] x is the new minimizer after following the gradient from
    //                its initial position.
    //
    void ConjugateR(const SymMat4& A1, const SymMat4& A2,
                    const Eigen::Vector4d& p1, Eigen::Vector4d* x) const;

   protected:
    explicit SymMat4(Eigen::Matrix4d M_in) : M_(std::move(M_in)) {}
    Eigen::Matrix4d M_;
  };

  QEF(SymMat4 A_in, Eigen::Vector4d p_in, const double& e_in)
      : A(std::move(A_in)), p(std::move(p_in)), e(e_in) {}

  QEF(const QEF& Q1, const QEF& Q2, Eigen::Vector4d x)
      : A(Q1.A + Q2.A), p(std::move(x)) {
    UpdatePE(Q1, Q2);
  }

  void Sum(const QEF& Q1, const QEF& Q2, const Eigen::Vector4d& x) {
    A = Q1.A + Q2.A;
    p = x;
    UpdatePE(Q1, Q2);
  }

  void Sum(const QEF& Q1, const QEF& Q2) {
    A = Q1.A + Q2.A;
    p = (Q1.p + Q2.p) / 2;
    UpdatePE(Q1, Q2);
  }

  // Note: for p=origin and e=0, Q(x) = xᵀAx.
  SymMat4 A;
  // The minimizer.
  Eigen::Vector4d p;
  // The minium quadric error ε.
  double e;

 protected:
  void UpdatePE(const QEF& Q1, const QEF& Q2) {
    A.ConjugateR(Q1.A, Q2.A, Q1.p, &p);
    const Eigen::Vector4d p1_to_p = this->p - Q1.p;
    const Eigen::Vector4d p2_to_p = this->p - Q2.p;
    e = Q1.e + Q2.e + p1_to_p.dot(Q1.A * p1_to_p) + p2_to_p.dot(Q2.A * p2_to_p);
  }
};

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
  bool ContractEdge(int v0, int v1, const Eigen::Vector3d& new_position,
                    double new_scalar);

  static double CalcTetrahedronVolume(
      int tetrahedron_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices);

  static bool AreAllIncidentTetrahedraPositive(
      int vertex_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices,
      const std::vector<std::vector<int>>& vertex_to_tetrahedra,
      const double kTinyVolume);

  // Return true if all incident tetrahedra of the given vertex,
  // excluding the ones also incident to the excluded vertex, have
  // positive volumes.
  static bool AreAllMorphedTetrahedraPositive(
      int vertex_index, const std::vector<VolumeElement>& tetrahedra,
      const std::vector<Eigen::Vector3<double>>& vertices,
      const std::vector<std::vector<int>>& vertex_to_tetrahedra,
      int exclude_vertex_index, double kTinyVolume);

  //--------------------------------------------------------
  // Related to combinatorics and discrete meshes and
  // interpolated signed distances.
  //--------------------------------------------------------

  const double kTinyVolume = 1e-12;  // cubic meters, 0.1x0.1x0.1-mm cube.

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
  // Related to Quadric Error Metrics
  //--------------------------------------------------------

  // vertex_Qs[i] = Quadric error metric at the i-th vertex.
  std::vector<QEF> vertex_Qs_;

  // Update QEF of the four vertices of the tet-th tetrahedron.
  void UpdateVertexQuadrics(int tet);
};

}  // namespace internal
}  // namespace geometry
}  // namespace drake
