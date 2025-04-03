#include "drake/geometry/proximity/temp_volume_coarsener.h"

#include <algorithm>

// You might see these files in:
// bazel-drake/external/+internal_repositories+vtk_internal/...
//
// To ease build system upkeep, we annotate VTK includes with their deps.
#include <vtkCellIterator.h>                       // vtkCommonDataModel
#include <vtkCleanPolyData.h>                      // vtkFiltersCore
#include <vtkDelaunay3D.h>                         // vtkFiltersCore
#include <vtkDoubleArray.h>                        // vtkCommonCore
#include <vtkPointData.h>                          // vtkCommonDataModel
#include <vtkPointSource.h>                        // vtkFiltersSources
#include <vtkPoints.h>                             // vtkCommonCore
#include <vtkPolyData.h>                           // vtkCommonDataModel
#include <vtkSmartPointer.h>                       // vtkCommonCore
#include <vtkUnstructuredGrid.h>                   // vtkCommonDataModel
#include <vtkUnstructuredGridQuadricDecimation.h>  // vtkFiltersCore
#include <vtkUnstructuredGridReader.h>             // vtkIOLegacy

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/calc_signed_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;

VolumeMesh<double> TempCoarsenVolumeMeshOfSdField(
    const VolumeMeshFieldLinear<double, double>& sdf_M, const double fraction) {
  // Convert drake's VolumeMeshFieldLinear to vtk's mesh data.
  vtkNew<vtkPoints> vtk_points;
  for (const Vector3d& p_MV : sdf_M.mesh().vertices()) {
    vtk_points->InsertNextPoint(p_MV.x(), p_MV.y(), p_MV.z());
  }
  vtkNew<vtkUnstructuredGrid> vtk_mesh;
  vtk_mesh->SetPoints(vtk_points);
  for (const VolumeElement& tet : sdf_M.mesh().tetrahedra()) {
    const vtkIdType ptIds[] = {tet.vertex(0), tet.vertex(1), tet.vertex(2),
                               tet.vertex(3)};
    vtk_mesh->InsertNextCell(VTK_TETRA, 4, ptIds);
  }
  vtkNew<vtkDoubleArray> vtk_signed_distances;
  vtk_signed_distances->Allocate(sdf_M.mesh().num_vertices());
  const std::string kFieldName("SignedDistance(meter)");
  vtk_signed_distances->SetName(kFieldName.c_str());
  for (int v = 0; v < sdf_M.mesh().num_vertices(); ++v) {
    vtk_signed_distances->SetValue(v, sdf_M.EvaluateAtVertex(v));
  }
  vtk_mesh->GetPointData()->AddArray(vtk_signed_distances);

  // Decimate the tetrahedral mesh + field.
  vtkNew<vtkUnstructuredGridQuadricDecimation> decimate;
  decimate->SetInputData(vtk_mesh);
  decimate->SetScalarsName(kFieldName.c_str());
  const double target_reduction = 1.0 - fraction;
  decimate->SetTargetReduction(target_reduction);
  // Default BoundaryWeight is 100.
  decimate->SetBoundaryWeight(1);
  decimate->Update();

  vtkNew<vtkUnstructuredGrid> vtk_decimated_mesh;
  vtk_decimated_mesh->ShallowCopy(decimate->GetOutput());

  // Convert vtk's mesh data back to drake's VolumeMesh.
  const vtkIdType num_vertices = vtk_decimated_mesh->GetNumberOfPoints();
  std::vector<Vector3d> vertices;
  vertices.reserve(num_vertices);
  vtkPoints* vtk_vertices = vtk_decimated_mesh->GetPoints();
  for (vtkIdType id = 0; id < num_vertices; id++) {
    double xyz[3];
    vtk_vertices->GetPoint(id, xyz);
    vertices.emplace_back(xyz);
  }
  std::vector<VolumeElement> tetrahedra;
  tetrahedra.reserve(vtk_decimated_mesh->GetNumberOfCells());
  auto iter = vtkSmartPointer<vtkCellIterator>::Take(
      vtk_decimated_mesh->NewCellIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal();
       iter->GoToNextCell()) {
    DRAKE_THROW_UNLESS(iter->GetCellType() == VTK_TETRA);
    vtkIdList* vtk_vertex_ids = iter->GetPointIds();
    tetrahedra.emplace_back(vtk_vertex_ids->GetId(0), vtk_vertex_ids->GetId(1),
                            vtk_vertex_ids->GetId(2), vtk_vertex_ids->GetId(3));
  }
  return {std::move(tetrahedra), std::move(vertices)};
}

double VolumeMeshCoarsener::CalcTetrahedronVolume(
    const int tetrahedron_index, const std::vector<VolumeElement>& tetrahedra,
    const std::vector<Vector3<double>>& vertices) {
  const int v0 = tetrahedra.at(tetrahedron_index).vertex(0);
  const int v1 = tetrahedra.at(tetrahedron_index).vertex(1);
  const int v2 = tetrahedra.at(tetrahedron_index).vertex(2);
  const int v3 = tetrahedra.at(tetrahedron_index).vertex(3);
  const Vector3d edge_01 = vertices.at(v1) - vertices.at(v0);
  const Vector3d edge_02 = vertices.at(v2) - vertices.at(v0);
  const Vector3d edge_03 = vertices.at(v3) - vertices.at(v0);
  return edge_01.cross(edge_02).dot(edge_03) / 6.0;
}

bool VolumeMeshCoarsener::AreAllIncidentTetrahedraPositive(
    int vertex_index, const std::vector<VolumeElement>& tetrahedra,
    const std::vector<Eigen::Vector3<double>>& vertices,
    const std::vector<std::vector<int>>& vertex_to_tetrahedra,
    const double kTinyVolume) {
  return std::ranges::all_of(
      vertex_to_tetrahedra.at(vertex_index).cbegin(),
      vertex_to_tetrahedra.at(vertex_index).cend(), [&](int tet) {
        return CalcTetrahedronVolume(tet, tetrahedra, vertices) >= kTinyVolume;
      });
}

bool VolumeMeshCoarsener::AreAllMorphedTetrahedraPositive(
    int vertex_index, const std::vector<VolumeElement>& tetrahedra,
    const std::vector<Eigen::Vector3<double>>& vertices,
    const std::vector<std::vector<int>>& vertex_to_tetrahedra,
    int exclude_vertex_index, double kTinyVolume) {
  const std::vector<int>& exclude_tetrahedra =
      vertex_to_tetrahedra.at(exclude_vertex_index);
  return std::ranges::all_of(
      vertex_to_tetrahedra.at(vertex_index).cbegin(),
      vertex_to_tetrahedra.at(vertex_index).cend(), [&](int tet) {
        if (std::find(exclude_tetrahedra.begin(), exclude_tetrahedra.end(),
                      tet) != exclude_tetrahedra.end()) {
          // Tetrahedron with both vertices 'vertex_index' and
          // 'exclude_vertex_index' do not count.
          return true;
        }
        return CalcTetrahedronVolume(tet, tetrahedra, vertices) >= kTinyVolume;
      });
}

bool VolumeMeshCoarsener::IsEdgeContractible(const int v0, const int v1,
                                             const Vector3d& new_position,
                                             double new_scalar) {
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(signed_distances_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(signed_distances_));

  // Prohibit edge contraction to the new position with different sign of the
  // new scalar value.
  if ((new_scalar < 0 && signed_distances_[v0] > 0) ||
      (new_scalar > 0 && signed_distances_[v0] < 0) ||
      (new_scalar < 0 && signed_distances_[v1] > 0) ||
      (new_scalar > 0 && signed_distances_[v1] < 0)) {
    return false;
  }

  // Pretend to change both v0 and v1 to the new position and then check
  // whether the incident tetrahedra still have at least a tiny positive
  // volumes.
  const Vector3d saved_v0_position(vertices_[v0]);
  const Vector3d saved_v1_position(vertices_[v1]);
  const double saved_value0 = signed_distances_[v0];
  const double saved_value1 = signed_distances_[v1];
  vertices_[v0] = new_position;
  vertices_[v1] = new_position;
  signed_distances_[v0] = new_scalar;
  signed_distances_[v1] = new_scalar;

  const bool is_contractible0 = AreAllMorphedTetrahedraPositive(
      v0, tetrahedra_, vertices_, vertex_to_tetrahedra_, v1, kTinyVolume);
  const bool is_contractible1 = AreAllMorphedTetrahedraPositive(
      v1, tetrahedra_, vertices_, vertex_to_tetrahedra_, v0, kTinyVolume);

  // Roll-back. Later the actual edge contraction will happen, if it's
  // contractible.
  vertices_[v0] = saved_v0_position;
  vertices_[v1] = saved_v1_position;
  signed_distances_[v0] = saved_value0;
  signed_distances_[v1] = saved_value1;

  return is_contractible0 && is_contractible1;
}

void VolumeMeshCoarsener::ContractEdge(const int v0, const int v1,
                                       const Vector3d& new_position,
                                       double new_scalar) {
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(signed_distances_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(signed_distances_));

  // The caller is supposed to check for contractibility before calling
  // this function.
  DRAKE_THROW_UNLESS(IsEdgeContractible(v0, v1, new_position, new_scalar));

  vertices_[v0] = new_position;
  vertices_[v1] = new_position;
  signed_distances_[v0] = new_scalar;
  signed_distances_[v1] = new_scalar;

  // Save copies of these data before changing tetrahedra_ and
  // vertex_to_tetrahedra_, so we can safely update auxiliary data structures
  // after updating the main data structures (tetrahedra_ and
  // vertex_to_tetrahedra_).
  //
  // An alternative design is to update auxiliary data structures first
  // before updating the main data structures.  In that case, we won't need
  // to save copies of these data.  We can use vertex_to_tetrahedra_[v0],
  // vertex_to_tetrahedra[v1], and GetTetrahedraOnEdge(v0, v1) right away.
  const std::vector<int> tetrahedra_of_v0(vertex_to_tetrahedra_[v0]);
  const std::vector<int> tetrahedra_of_v1(vertex_to_tetrahedra_[v1]);
  const std::vector<int> tetrahedra_of_v0v1(
      VolumeMeshRefiner::GetTetrahedraOnEdge(v0, v1));

  // Move all tetrahedra of v1 to v0, except the ones on the edge v0v1, which
  // will be marked for deletion later.
  for (int tet : tetrahedra_of_v1) {
    // Skip a tetrahedron on edge v0v1.
    if (std::find(tetrahedra_of_v0v1.cbegin(), tetrahedra_of_v0v1.cend(),
                  tet) != tetrahedra_of_v0v1.cend()) {
      continue;
    }
    // Make a copy of the tetrahedron to prevent any race condition in the
    // future.  If there is no code change in the future, we could use a
    // const reference instead of a const copy.
    const VolumeElement tetrahedron(tetrahedra_.at(tet));
    std::array<int, 4> v{tetrahedron.vertex(0), tetrahedron.vertex(1),
                         tetrahedron.vertex(2), tetrahedron.vertex(3)};
    auto it = std::find(v.begin(), v.end(), v1);
    DRAKE_THROW_UNLESS(it != v.end());
    *it = v0;
    tetrahedra_.at(tet) = VolumeElement(v[0], v[1], v[2], v[3]);
    vertex_to_tetrahedra_[v0].push_back(tet);
  }
  // Clear all of vertex_to_tetrahedra_[v1].
  vertex_to_tetrahedra_[v1] = std::vector<int>({});
  // Clear some of vertex_to_tetrahedra_[v0].
  for (const int tet : tetrahedra_of_v0v1) {
    std::erase(vertex_to_tetrahedra_[v0], tet);
  }

  // Update auxiliary data structures.

  is_vertex_deleted_[v1] = true;
  for (const int tet : tetrahedra_of_v0v1) {
    if (!is_tet_deleted_[tet]) {
      is_tet_deleted_[tet] = true;
      ++num_tet_deleted_;
    }
  }
  // Important: update is_tet_deleted_ **before** is_tet_morphed_.
  for (int tet : tetrahedra_of_v0) {
    if (!is_tet_deleted_[tet]) {
      is_tet_morphed_[tet] = true;
    }
  }
  for (int tet : tetrahedra_of_v1) {
    if (!is_tet_deleted_[tet]) {
      is_tet_morphed_[tet] = true;
    }
  }
}

VolumeMeshCoarsener::VolumeMeshCoarsener(
    const VolumeMeshFieldLinear<double, double>& sdfield_M,
    const TriangleSurfaceMesh<double>& original_M)
    : VolumeMeshRefiner(sdfield_M.mesh()),
      original_boundary_(TriangleSurfaceMesh<double>(original_M)) {
  VolumeMeshRefiner::tetrahedra_ = VolumeMeshRefiner::input_mesh_.tetrahedra();
  VolumeMeshRefiner::vertices_ = VolumeMeshRefiner::input_mesh_.vertices();
  VolumeMeshRefiner::ResetVertexToTetrahedra();

  DRAKE_THROW_UNLESS(ssize(vertices_) == ssize(sdfield_M.values()));
  signed_distances_ = sdfield_M.values();

  is_vertex_deleted_ = std::vector<bool>(vertices_.size(), false);
  is_tet_deleted_ = std::vector<bool>(tetrahedra_.size(), false);
  num_tet_deleted_ = 0;
  is_tet_morphed_ = std::vector<bool>(tetrahedra_.size(), false);

  support_boundary_mesh_ = ConvertVolumeToSurfaceMeshWithBoundaryVertices(
      input_mesh_, &boundary_to_volume_, nullptr);
  for (int i = 0; i < support_boundary_mesh_.num_vertices(); ++i) {
    const int v = boundary_to_volume_.at(i);
    volume_to_boundary_[v] = i;
  }
}

void VolumeMeshCoarsener::InitializeVertexQEFs() {
  int num_vertices = vertices_.size();
  // Initialize each vertex_Qs_[v] from (x,y,z,sdf).
  vertex_Qs_.clear();
  for (int v = 0; v < num_vertices; ++v) {
    QEF Q = QEF::Zero();
    Q.p(0) = vertices_[v].x();
    Q.p(1) = vertices_[v].y();
    Q.p(2) = vertices_[v].z();
    Q.p(3) = signed_distances_[v];
    vertex_Qs_.push_back(Q);
  }
  int num_tetrahedra = tetrahedra_.size();
  // Contribute every tetrahedron's QEM to its four vertices.
  for (int tet = 0; tet < num_tetrahedra; ++tet) {
    UpdateVerticesQuadricsFromTet(tet);
  }
  // Contribute every boundary triangle's QEM to its three vertices.
  for (int boundary_tri = 0; boundary_tri < ssize(boundary_to_volume_);
       ++boundary_tri) {
    UpdateVerticesQuadricsFromBoundaryFace(boundary_tri);
  }
}

VolumeMesh<double> VolumeMeshCoarsener::coarsen(double fraction) {
  const int num_input_tetrahedra = tetrahedra_.size();
  drake::log()->info("Number of input tetrahedra = {}.", num_input_tetrahedra);
  const int target_num_tetrahedra =
      static_cast<int>(fraction * num_input_tetrahedra);
  drake::log()->info("Target number of tetrahedra = {}.",
                     target_num_tetrahedra);

  InitializeVertexQEFs();

  int num_edge_contraction = 0;
  // Increase max_error_threshold in the range (0, kMaxError] linearly from
  // one iteration to the next.
  constexpr double kMaxError = 1e-6;
  constexpr int kNumIteration = 10000;
  const double delta_max_error = kMaxError / kNumIteration;
  double max_error_threshold = 0;
  for (int iteration = 0; iteration < kNumIteration; ++iteration) {
    max_error_threshold += delta_max_error;
    double min_error = std::numeric_limits<double>::max();
    double max_error = std::numeric_limits<double>::min();
    if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
      break;
    }
    is_tet_morphed_ = std::vector<bool>(tetrahedra_.size(), false);
    int num_edge_considered_for_contraction_in_this_iteration = 0;
    for (int tet = 0; tet < num_input_tetrahedra; ++tet) {
      if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
        break;
      }
      if (is_tet_deleted_[tet]) {
        continue;
      }
      if (is_tet_morphed_[tet]) {
        continue;
      }
      const VolumeElement& tetrahedron = tetrahedra_[tet];
      // Check for a valid edge contraction of edge(i,j) of the tetrahedron.
      for (std::pair<int, int> ij : std::vector<std::pair<int, int>>{
               {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}) {
        int vi = tetrahedron.vertex(ij.first);
        int vj = tetrahedron.vertex(ij.second);

        DRAKE_THROW_UNLESS(!is_vertex_deleted_[vi]);
        DRAKE_THROW_UNLESS(!is_vertex_deleted_[vj]);

        QEF edge_Q = QEF::Zero();
        edge_Q.Sum(vertex_Qs_[vi], vertex_Qs_[vj]);

        // Thresholding on expected error if we contract the edge(vi,vj).
        if (edge_Q.e < min_error) {
          min_error = edge_Q.e;
        }
        if (edge_Q.e > max_error) {
          max_error = edge_Q.e;
        }
        if (edge_Q.e >= max_error_threshold) {
          continue;
        }
        const Vector3d optimal_point{edge_Q.p.x(), edge_Q.p.y(), edge_Q.p.z()};
        const double optimal_value = edge_Q.p.w();

        ++num_edge_considered_for_contraction_in_this_iteration;
        if (IsEdgeContractible(vi, vj, optimal_point, optimal_value)) {
          ContractEdge(vi, vj, optimal_point, optimal_value);
          ++num_edge_contraction;
          vertex_Qs_[vi] = edge_Q;
          vertex_Qs_[vj] = edge_Q;
          break;
        }
      }  // for (std::pair<int, int> ij
    }  // for (int tet
    // drake::log()->info("");
    // drake::log()->info("iteration {}", iteration);
    // drake::log()->info("Number of edge contractions = {}",
    //                    num_edge_contraction);
    // drake::log()->info("Number of deleted tetrahedra = {}",
    //                    num_tet_deleted_);
    // drake::log()->info("Minimum edge error = {}", min_error);
    // drake::log()->info("Maximum edge error = {}", max_error);
    // drake::log()->info(
    //     "num_edge_considered_for_contraction_in_this_iteration = {}",
    //     num_edge_considered_for_contraction_in_this_iteration);
  }  // for iteration
  drake::log()->info("");
  drake::log()->info("End iterations.");
  drake::log()->info("");
  drake::log()->info("Number of edge contractions = {}", num_edge_contraction);
  drake::log()->info("Number of deleted tetrahedra = {}", num_tet_deleted_);

  std::vector<VolumeElement> valid_tetrahedra;
  for (int tet = 0; tet < ssize(tetrahedra_); ++tet) {
    if (!is_tet_deleted_[tet]) {
      valid_tetrahedra.push_back(tetrahedra_[tet]);
    }
  }

  drake::log()->info("Number of output tetrahedra = {}",
                     valid_tetrahedra.size());

  // TODO(DamrongGuoy) Remove deleted vertices and don't forget to renumber
  //  vertices (using std::map) in each valid_tetrahedra's record.

  return {std::move(valid_tetrahedra), std::move(vertices_)};
}

//***************************************************************************
// These two numerical routines follow Section 3.4 "Numerical Issues" in:
//
//   [Huy2007] Huy, Vo; Callahan, Steven; Lindstrom, Peter; Pascucci, Valerio;
//   and Silva, Claudio. (2007). Streaming Simplification of Tetrahedral
//   Meshes.  IEEE transactions on visualization and computer graphics.
//   13. 145-55. 10.1109/TVCG.2007.21.
//
// with a reference implementation in:
// - vtkUnstructuredGridQuadricDecimationSymMat4::ConjugateR()
// - vtkUnstructuredGridQuadricDecimationTetra::UpdateQuadric()
//***************************************************************************

void SymMat4::ConjugateR(const drake::geometry::internal::SymMat4& A1,
                         const drake::geometry::internal::SymMat4& A2,
                         const Eigen::Vector4d& p1, Eigen::Vector4d* x) const {
  // The paper [Huy2007], page 7, uses κₘₐₓ = 10⁴. The VTK implementation uses
  // κₘₐₓ = 10³.  For now, we follow the paper.
  constexpr double kKappaMax = 1e4;
  // tr(A) / (n * κₘₐₓ) with n = 4 for tetrahedral meshes.
  double exit_threshold =
      (M_(0, 0) + M_(1, 1) + M_(2, 2) + M_(3, 3)) / (4 * kKappaMax);
  Vector4d r = (A1 - A2) * (p1 - (*x));  // Nagative gradient.
  Vector4d displace = Vector4d::Zero();  // p in the paper.
  for (int k = 0; k < 4; ++k) {
    const double s = r.squaredNorm();
    // TODO(DamrongGuoy): The "==" should be enough; we just follow VTK to use
    //  "<=" here.  Change it to "==" later.
    if (s <= 0) {
      break;
    }
    displace += (r / s);                    // displace = displace + r/∥r∥²
    const Vector4d q = (*this) * displace;  // q = Ap
    const double t = displace.dot(q);
    if (s * t <= exit_threshold) {
      break;
    }
    r -= (q / t);            // Update negative gradient.
    (*x) += (displace / t);  // Move x.
  }
}

void VolumeMeshCoarsener::UpdateVerticesQuadricsFromTet(int tet) {
  DRAKE_THROW_UNLESS(0 <= tet && tet < ssize(tetrahedra_));
  const int v0 = tetrahedra_[tet].vertex(0);
  const int v1 = tetrahedra_[tet].vertex(1);
  const int v2 = tetrahedra_[tet].vertex(2);
  const int v3 = tetrahedra_[tet].vertex(3);
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertex_Qs_));
  // Coefficients of each vector has unit in meters.
  // The coordinates (x,y,z) are in meters.
  // The signed distance is also in meters.
  const Vector4d a = vertex_Qs_[v1].p - vertex_Qs_[v0].p;
  const Vector4d b = vertex_Qs_[v2].p - vertex_Qs_[v0].p;
  const Vector4d c = vertex_Qs_[v3].p - vertex_Qs_[v0].p;

  // TODO(DamrongGuoy): Use a more compact spelling of Eigen than this very
  // long formula of the generalized cross product of three 4-vectors.

  // n is the generalized cross product of the three Vector4d a, b, c.
  // See the determinant formula in Section 3.4 "Numerical Issues" of
  // [Huy2007].
  //
  //            | Fx Fy Fz Fw |
  //   n =  det | <--- a ---> | ; Fx,Fy,Fz,Fw are the basis vectors.
  //            | <--- b ---> |
  //            | <--- c ---> |
  //
  // Each coefficient of n has unit in cubic meters.
  //
  const Vector4d n = {
      // clang-format off
      a.y() * (b.z() * c.w() - b.w() * c.z()) +
      a.z() * (b.w() * c.y() - b.y() * c.w()) +
      a.w() * (b.y() * c.z() - b.z() * c.y()),

      a.z() * (b.x() * c.w() - b.w() * c.x()) +
      a.w() * (b.z() * c.x() - b.x() * c.z()) +
      a.x() * (b.w() * c.z() - b.z() * c.w()),

      a.w() * (b.x() * c.y() - b.y() * c.x()) +
      a.x() * (b.y() * c.w() - b.w() * c.y()) +
      a.y() * (b.w() * c.x() - b.x() * c.w()),

      a.x() * (b.z() * c.y() - b.y() * c.z()) +
      a.y() * (b.x() * c.z() - b.z() * c.x()) +
      a.z() * (b.y() * c.x() - b.x() * c.y()),
      // clang-format on
  };
  // Outer product of 4-vector n gives the 4x4 symmetric matrix A.
  // After this step, coefficients of A have units in meter^6.
  SymMat4 A = SymMat4::FromOuterProductOfVector4d(n);
  // Divide A by the volume of the tetrahedron gives the fundamental quadric
  // matrix with the units of its coefficients in cubic meters.
  const double tetrahedron_volume =
      CalcTetrahedronVolume(tet, tetrahedra_, vertices_);
  DRAKE_THROW_UNLESS(tetrahedron_volume > 0);
  A /= tetrahedron_volume;
  // Divide the tetrahedrn's quadric matrix to its 4 vertices equally.
  A /= 4.0;

  vertex_Qs_[v0].A += A;
  vertex_Qs_[v1].A += A;
  vertex_Qs_[v2].A += A;
  vertex_Qs_[v3].A += A;
}

void VolumeMeshCoarsener::UpdateVerticesQuadricsFromBoundaryFace(
    int boundary_tri) {
  const SurfaceTriangle& triangle =
      support_boundary_mesh_.triangles().at(boundary_tri);
  const int boundary_vertex0 = triangle.vertex(0);
  const int boundary_vertex1 = triangle.vertex(1);
  const int boundary_vertex2 = triangle.vertex(2);
  const int v0 = boundary_to_volume_.at(boundary_vertex0);
  const int v1 = boundary_to_volume_.at(boundary_vertex1);
  const int v2 = boundary_to_volume_.at(boundary_vertex2);

  Vector4d e1, e2;
  e1 = vertex_Qs_.at(v1).p - vertex_Qs_.at(v0).p;
  e2 = vertex_Qs_.at(v2).p - vertex_Qs_.at(v0).p;
  e1.normalize();
  e2 = e2 - e1 * e2.dot(e1);
  e2.normalize();

  // A = I - e1*e1ᵀ - e2*e2ᵀ
  SymMat4 A = SymMat4::Identity() - SymMat4::FromOuterProductOfVector4d(e1) -
              SymMat4::FromOuterProductOfVector4d(e2);

  // TODO(DamrongGuoy): Check the units.  It looks like the matrix A's
  //  coefficients have units in squared meters not cubic meters.

  // TODO(DamrongGuoy): VTK has "boundaryWeight" parameter (default 100) that
  //  multiplies A for even larger weight. Is this a hack?
  constexpr double kBoundaryWeight = 100;

  // Multiply by area of the triangle and share (/3) it among three vertices.
  A *= support_boundary_mesh_.area(boundary_tri) / 3 * kBoundaryWeight;
  vertex_Qs_.at(v0).A += A;
  vertex_Qs_.at(v1).A += A;
  vertex_Qs_.at(v2).A += A;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
