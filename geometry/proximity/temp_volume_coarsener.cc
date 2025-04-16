#include "drake/geometry/proximity/temp_volume_coarsener.h"

#include <algorithm>

// You might see these files in:
// bazel-drake/external/+internal_repositories+vtk_internal/...
//
// To ease build system upkeep, we annotate VTK includes with their deps.
#include <Eigen/Eigenvalues>
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

#include "drake/common/fmt_eigen.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/calc_signed_distance_to_surface_mesh.h"
#include "drake/geometry/proximity/field_intersection.h"
#include "drake/geometry/proximity/hydroelastic_internal.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/make_ellipsoid_field.h"
#include "drake/geometry/proximity/make_ellipsoid_mesh.h"
#include "drake/geometry/proximity/mesh_distance_boundary.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;
using math::RotationMatrixd;

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

//-------------------------------------------------------------------------
// Main VolumeMeshCoarsener
//-------------------------------------------------------------------------

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

double VolumeMeshCoarsener::CalcMinIncidentTetrahedronVolume(
    const int vertex) const {
  DRAKE_THROW_UNLESS(0 <= vertex && vertex < ssize(vertex_to_tetrahedra_));
  double min_volume = std::numeric_limits<double>::max();
  for (const int tet : vertex_to_tetrahedra_.at(vertex)) {
    const double volume = CalcTetrahedronVolume(tet, tetrahedra_, vertices_);
    if (volume < min_volume) {
      min_volume = volume;
    }
  }
  return min_volume;
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
        return CalcTetrahedronVolume(tet, tetrahedra, vertices) > kTinyVolume;
      });
}

bool VolumeMeshCoarsener::IsEdgeContractible(const int v0, const int v1,
                                             const Vector3d& new_position,
                                             const double new_scalar) {
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertex_to_tetrahedra_));
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(signed_distances_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(signed_distances_));

  // Prohibit edge contraction between a boundary vertex and a non-boundary
  // vertex.
  if (volume_to_boundary_.contains(v0)) {
    if (!volume_to_boundary_.contains(v1)) {
      return false;
    }
  }
  if (volume_to_boundary_.contains(v1)) {
    if (!volume_to_boundary_.contains(v0)) {
      return false;
    }
  }

  // Prohibit edge contraction to the new position with different sign of
  // the new scalar value.
  if ((new_scalar < 0 && signed_distances_[v0] > 0) ||
      (new_scalar > 0 && signed_distances_[v0] < 0) ||
      (new_scalar < 0 && signed_distances_[v1] > 0) ||
      (new_scalar > 0 && signed_distances_[v1] < 0)) {
    return false;
  }

  // unused(new_position);

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
      v0, tetrahedra_, vertices_, vertex_to_tetrahedra_, v1, kTinyVolume_);
  const bool is_contractible1 = AreAllMorphedTetrahedraPositive(
      v1, tetrahedra_, vertices_, vertex_to_tetrahedra_, v0, kTinyVolume_);

  // Roll-back. Later the actual edge contraction will happen if it's
  // contractible.
  vertices_[v0] = saved_v0_position;
  vertices_[v1] = saved_v1_position;
  signed_distances_[v0] = saved_value0;
  signed_distances_[v1] = saved_value1;

  return is_contractible0 && is_contractible1;
}

bool VolumeMeshCoarsener::IsVertexMovable(int vertex,
                                          const Eigen::Vector3d& new_position) {
  DRAKE_THROW_UNLESS(0 <= vertex && vertex < ssize(vertices_));
  DRAKE_THROW_UNLESS(0 <= vertex && vertex < ssize(vertex_to_tetrahedra_));
  const Vector3d saved_old_position = vertices_[vertex];

  // Temporary set the vertex to the new position. Later we will roll back.
  vertices_[vertex] = new_position;

  bool found_negative_tetrahedron = false;
  for (const int tet : vertex_to_tetrahedra_[vertex]) {
    if (CalcTetrahedronVolume(tet, tetrahedra_, vertices_) <= kTinyVolume_) {
      found_negative_tetrahedron = true;
      break;
    }
  }

  // Roll back.
  vertices_[vertex] = saved_old_position;
  return !found_negative_tetrahedron;
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
  const double min_tet_volume_before =
      std::min(CalcMinIncidentTetrahedronVolume(v0),
               CalcMinIncidentTetrahedronVolume(v1));
  DRAKE_THROW_UNLESS(min_tet_volume_before >= kTinyVolume_);

  // The caller is supposed to check for contractibility before calling
  // this function.
  DRAKE_THROW_UNLESS(IsEdgeContractible(v0, v1, new_position, new_scalar));

  // Save old parts of the mesh before edge contraction.
  const VolumeMesh<double> star_v0_before = DebugTetrahedraOfVertex(v0);
  const VolumeMesh<double> star_v1_before = DebugTetrahedraOfVertex(v1);
  const VolumeMesh<double> star_v0v1_before =
      DebugTetrahedraOfBothVertex(v0, v1);
  const VolumeMesh<double> local_mesh_v0_v1_before =
      DebugTetrahedraOfEitherVertices(v0, v1);
  const QEF v0_Q_before = vertex_Qs_.at(v0);
  const QEF v1_Q_before = vertex_Qs_.at(v1);
  const Vector3d vertex_v0_before = vertices_[v0];
  const Vector3d vertex_v1_before = vertices_[v1];

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
  // Clear tetrahedra on edges v0v1 from vertex_to_tetrahedra_[v0].
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

  // Update edge_QEFs of all tetrahedra of v0 (including the ones recently
  // moved from v1. They are the newly morphed tetrahedra.
  // There are a number of redundant calculations since an edge could
  // be shared by multiple tetrahedra.
  for (const int tet : vertex_to_tetrahedra_[v0]) {
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const int vi = tetrahedra_.at(tet).vertex(i);
        const int vj = tetrahedra_.at(tet).vertex(j);
        // Force new calculation since we didn't keep track which edges
        // have changed.
        edge_QEFs_[Edge(vi, vj)] =
            QEF::Combine(vertex_Qs_.at(vi), vertex_Qs_.at(vj));
      }
    }
  }
  for (const int tet : vertex_to_tetrahedra_[v0]) {
    Edge dummy;
    tetrahedron_queue_.emplace(tet,
                               CalcOrFetchTetrahedronMinEdgeCost(tet, &dummy));
  }

  // Check that the min tetrahedron volume:
  // 1. didn't go below the volume threshold, and
  // 2. didn't shrink beyond a factor of 10 after the edge contraction.
  const double min_tet_volume_after = CalcMinIncidentTetrahedronVolume(v0);
  DRAKE_THROW_UNLESS(min_tet_volume_after >= kTinyVolume_);
  // if (!(min_tet_volume_before / min_tet_volume_after < 10.0))
  if (!(min_tet_volume_after > 0)) {
    drake::log()->warn(fmt::format("v0 = {}, v1 = {}", v0, v1));
    drake::log()->warn(fmt::format("vertex_v0_beforeᵀ = {}",
                                   fmt_eigen(vertex_v0_before.transpose())));
    drake::log()->warn(fmt::format("vertex_v1_beforeᵀ = {}\n",
                                   fmt_eigen(vertex_v1_before.transpose())));

    drake::log()->warn(
        fmt::format("v0_Q_before.A = \n{}", fmt_eigen(v0_Q_before.A.Mat4d())));
    drake::log()->warn(fmt::format("v0_Q_before.pᵀ = \n{}",
                                   fmt_eigen(v0_Q_before.p.transpose())));
    LogAndWriteQ(v0, v0_Q_before, "ContractEdge_before_shrink10X");
    drake::log()->warn(fmt::format("v0_Q_before.e = {}\n", v0_Q_before.e));

    drake::log()->warn(
        fmt::format("v1_Q_before.A = \n{}", fmt_eigen(v1_Q_before.A.Mat4d())));
    drake::log()->warn(fmt::format("v1_Q_before.pᵀ = \n{}",
                                   fmt_eigen(v1_Q_before.p.transpose())));
    LogAndWriteQ(v1, v1_Q_before, "ContractEdge_before_shrink10X");
    drake::log()->warn(fmt::format("v1_Q_before.e = {}\n", v1_Q_before.e));

    drake::log()->warn(fmt::format("new_positionᵀ = \n{}",
                                   fmt_eigen(new_position.transpose())));
    drake::log()->warn(fmt::format("new_scalar = {}\n", new_scalar));

    WriteSavedTetrahedraOfVertexBeforeEdgeContration(
        v0, star_v0_before, "ContractEdge_before_shrink10X_star_vertex");
    WriteSavedTetrahedraOfVertexBeforeEdgeContration(
        v1, star_v1_before, "ContractEdge_before_shrink10X_star_vertex");
    WriteSavedTetrahedraBeforeEdgeContraction(
        v0, v1, star_v0v1_before, "ContractEdge_before_shrink10X_star_edge");
    WriteSavedTetrahedraOfEitherVerticesBeforeEdgeContraction(
        v0, v1, local_mesh_v0_v1_before,
        "ContractEdge_before_shrink10X_closed_star_edge");
    WriteTetrahedraAfterEdgeContraction(v0, v1, "ContractEdge_after_shrink10X");
  }
  // DRAKE_THROW_UNLESS(min_tet_volume_before / min_tet_volume_after < 10.0);
  DRAKE_THROW_UNLESS(min_tet_volume_after > 0);
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
  const int num_tetrahedra = tetrahedra_.size();
  // Contribute every tetrahedron's QEM to its four vertices.
  for (int tet = 0; tet < num_tetrahedra; ++tet) {
    UpdateVerticesQuadricsFromTet(tet);
  }
  // Contribute every boundary triangle's QEM to its three vertices.
  const int num_boundary_triangles = support_boundary_mesh_.num_triangles();
  for (int boundary_tri = 0; boundary_tri < num_boundary_triangles;
       ++boundary_tri) {
    UpdateVerticesQuadricsFromBoundaryFace(boundary_tri);
  }
}

void VolumeMeshCoarsener::InitializeTetrahedronQueue() {
  while (!tetrahedron_queue_.empty()) tetrahedron_queue_.pop();
  edge_QEFs_.clear();
  for (int tet = 0; tet < ssize(tetrahedra_); ++tet) {
    Edge dummy;
    tetrahedron_queue_.emplace(tet,
                               CalcOrFetchTetrahedronMinEdgeCost(tet, &dummy));
  }
}

double VolumeMeshCoarsener::CalcOrFetchTetrahedronMinEdgeCost(int tet,
                                                              Edge* min_edge) {
  const VolumeElement& tetrahedron = tetrahedra_[tet];
  double tet_cost = std::numeric_limits<double>::max();
  // Access the six edges of the tetrahedron.
  for (std::pair<int, int> ij : std::vector<std::pair<int, int>>{
           {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}) {
    int vi = tetrahedron.vertex(ij.first);
    int vj = tetrahedron.vertex(ij.second);
    Edge edge(vi, vj);

    QEF edge_Q;
    if (edge_QEFs_.contains(edge)) {
      edge_Q = edge_QEFs_.at(edge);
    } else {
      edge_Q = QEF::Combine(vertex_Qs_[vi], vertex_Qs_[vj]);
      edge_QEFs_[edge] = edge_Q;
    }
    if (edge_Q.e < tet_cost) {
      tet_cost = edge_Q.e;
      *min_edge = Edge(vi, vj);
    }
  }
  return tet_cost;
}

#if 0
VolumeMesh<double> VolumeMeshCoarsener::coarsen(double fraction) {
  const int num_input_tetrahedra = tetrahedra_.size();
  drake::log()->info("Number of input tetrahedra = {}.", num_input_tetrahedra);
  const int target_num_tetrahedra =
      static_cast<int>(fraction * num_input_tetrahedra);
  drake::log()->info("Target number of tetrahedra = {}.",
                     target_num_tetrahedra);

  InitializeVertexQEFs();

  int num_total_edge_contraction = 0;
  // Increase max_error_threshold in the range (0, kLastErrorThreshold] linearly
  // from one iteration to the next.
  constexpr double kFirstErrorThreshold = 1e-12;
  constexpr double kLastErrorThreshold = 1e-6;
  constexpr double kGrowthErrorThreshold =
      kLastErrorThreshold / kFirstErrorThreshold;
  constexpr int kNumIterations = 1000;
  constexpr int kNumReports = 10;
  constexpr int kReportPeriod = kNumIterations / kNumReports;
  double max_error_threshold = 0;
  int num_perform_iterations = 0;
  for (int iteration = 0; iteration < kNumIterations; ++iteration) {
    // If we exhaust all iterations, both num_performed_iterations and
    // `iteration` variables will become kNumIterations.
    // If we exit early, num_performed_iterations will be 1 + `iteration`
    // because `iteration` starts at 0 not 1.
    ++num_perform_iterations;
    max_error_threshold =
        kFirstErrorThreshold *
        std::pow(kGrowthErrorThreshold,
                 (static_cast<double>(iteration) / (kNumIterations - 1)));
    double min_edge_error = std::numeric_limits<double>::max();
    double max_edge_error = std::numeric_limits<double>::min();
    if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
      break;
    }
    is_tet_morphed_ = std::vector<bool>(tetrahedra_.size(), false);
    int num_edge_contraction_in_this_iteration = 0;
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

        QEF edge_Q = QEF::Combine(vertex_Qs_[vi], vertex_Qs_[vj]);

        // Thresholding on expected error if we contract the edge(vi,vj).
        if (edge_Q.e < min_edge_error) {
          min_edge_error = edge_Q.e;
        }
        if (edge_Q.e > max_edge_error) {
          max_edge_error = edge_Q.e;
        }
        if (edge_Q.e >= max_error_threshold) {
          continue;
        }
        const Vector3d optimal_point{edge_Q.p.x(), edge_Q.p.y(), edge_Q.p.z()};
        const double optimal_value = edge_Q.p.w();

        ++num_edge_considered_for_contraction_in_this_iteration;
        if (IsEdgeContractible(vi, vj, optimal_point, optimal_value)) {
          ContractEdge(vi, vj, optimal_point, optimal_value);
          ++num_total_edge_contraction;
          ++num_edge_contraction_in_this_iteration;
          vertex_Qs_[vi] = edge_Q;
          vertex_Qs_[vj] = edge_Q;
          break;
        }
      }  // for (std::pair<int, int> ij
    }  // for (int tet

    if (iteration % kReportPeriod == 0) {
      drake::log()->info("");
      drake::log()->info(
          "iteration {}, max_error_threshold {}, "
          "max_edge_error = {}, min_edge_error = {}",
          iteration, max_error_threshold, max_edge_error, min_edge_error);
      drake::log()->info(
          "num_edge_contraction_in_this_iteration = {}, "
          "num_edge_considered_for_contraction_in_this_iteration = {}, "
          "num_total_edge_contraction = {}, "
          "num_tet_deleted_ = {}, "
          "num_input_tetrahedra - num_tet_deleted_ = {}",
          num_edge_contraction_in_this_iteration,
          num_edge_considered_for_contraction_in_this_iteration,
          num_total_edge_contraction, num_tet_deleted_,
          num_input_tetrahedra - num_tet_deleted_);
    }
  }  // for iteration
  drake::log()->info("");
  drake::log()->info("End iterations: num_perform_iterations = {}",
                     num_perform_iterations);
  drake::log()->info("");
  drake::log()->info("Number of edge contractions = {}",
                     num_total_edge_contraction);
  drake::log()->info("Number of deleted tetrahedra = {}", num_tet_deleted_);

  std::vector<VolumeElement> valid_tetrahedra;
  for (int tet = 0; tet < ssize(tetrahedra_); ++tet) {
    if (!is_tet_deleted_[tet]) {
      valid_tetrahedra.push_back(tetrahedra_[tet]);
    }
  }

  // TODO(DamrongGuoy) Remove deleted vertices and don't forget to renumber
  //  vertices (using std::map) in each valid_tetrahedra's record.
  const VolumeMesh<double> coarsen_mesh{std::move(valid_tetrahedra),
                                        std::move(vertices_)};
  drake::log()->info(
      "Number of output tetrahedra: coarsen_mesh.num_elements() = {}",
      coarsen_mesh.num_elements());
  drake::log()->info("coarsen_mesh.CalcMinTetrahedralVolume() = {}",
                     coarsen_mesh.CalcMinTetrahedralVolume());

  return coarsen_mesh;
}
#endif

VolumeMesh<double> VolumeMeshCoarsener::coarsen(double fraction) {
  const int num_input_tetrahedra = tetrahedra_.size();
  drake::log()->info("Number of input tetrahedra = {}.", num_input_tetrahedra);
  const int target_num_tetrahedra =
      static_cast<int>(fraction * num_input_tetrahedra);
  drake::log()->info("Target number of tetrahedra = {}.",
                     target_num_tetrahedra);
  drake::log()->info("input_mesh_.CalcMinTetrahedralVolume() = {}.",
                     input_mesh_.CalcMinTetrahedralVolume());

  InitializeVertexQEFs();
  InitializeTetrahedronQueue();

  int num_total_edge_contraction = 0;
  // Each edge contraction should remove at least two tetrahedra.
  const int num_iterations = (num_input_tetrahedra - target_num_tetrahedra) / 2;
  constexpr int kNumReports = 10;
  const int report_period =
      num_iterations >= kNumReports ? num_iterations / kNumReports : 1;
  int iteration = 0;
  int next_report_iteration = report_period;
  double min_edge_error = std::numeric_limits<double>::max();
  double max_edge_error = std::numeric_limits<double>::min();

  while (!tetrahedron_queue_.empty()) {
    if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
      break;
    }
    bool found_edge_to_contract = false;
    QEF Q_of_edge_to_contract;
    int selected_vertex0 = -1;
    int selected_vertex1 = -1;
    Vector3d optimal_point{0, 0, 0};
    double optimal_value{0};

    const int tet = tetrahedron_queue_.top().tet;
    const double cost = tetrahedron_queue_.top().cost;
    tetrahedron_queue_.pop();
    // We couldn't update the priority queue when tetrahedra change. Some
    // entries could have been out of date. We check here that the
    // tetrahedron wasn't deleted and its cost didn't change.
    if (is_tet_deleted_[tet]) {
      continue;
    }
    Edge min_edge;
    if (cost != CalcOrFetchTetrahedronMinEdgeCost(tet, &min_edge)) {
      continue;
    }
    int vi = min_edge.first();
    int vj = min_edge.second();
    DRAKE_THROW_UNLESS(!is_vertex_deleted_[vi]);
    DRAKE_THROW_UNLESS(!is_vertex_deleted_[vj]);

    QEF edge_Q = QEF::Combine(vertex_Qs_[vi], vertex_Qs_[vj]);
    Vector3d xyz{edge_Q.p.x(), edge_Q.p.y(), edge_Q.p.z()};
    double value = edge_Q.p.w();
    if (edge_Q.e > max_edge_error) {
      max_edge_error = edge_Q.e;
    }
    if (IsEdgeContractible(vi, vj, xyz, value)) {
      found_edge_to_contract = true;
      ++iteration;
      min_edge_error = edge_Q.e;
      Q_of_edge_to_contract = edge_Q;
      selected_vertex0 = vi;
      selected_vertex1 = vj;
      optimal_point = xyz;
      optimal_value = value;
    }

    if (found_edge_to_contract) {
      ContractEdge(selected_vertex0, selected_vertex1, optimal_point,
                   optimal_value);
      vertex_Qs_[selected_vertex0] = Q_of_edge_to_contract;
      vertex_Qs_[selected_vertex1] = Q_of_edge_to_contract;
      ++num_total_edge_contraction;

      if (volume_to_boundary_.contains(selected_vertex0) ||
          volume_to_boundary_.contains(selected_vertex1)) {
        double offset_distance = 0;
        int count_num_boundary_vertices = 0;
        if (volume_to_boundary_.contains(selected_vertex0)) {
          offset_distance += signed_distances_[selected_vertex0];
          ++count_num_boundary_vertices;
        }
        if (volume_to_boundary_.contains(selected_vertex1)) {
          offset_distance += signed_distances_[selected_vertex1];
          ++count_num_boundary_vertices;
        }
        DRAKE_THROW_UNLESS(count_num_boundary_vertices != 0);
        offset_distance /= count_num_boundary_vertices;

        const Vector4d proj = CalcProjectionToOffsetSurface(
            optimal_point, original_boundary_, offset_distance);
        const Vector3d position{proj.x(), proj.y(), proj.z()};

        if (IsVertexMovable(selected_vertex0, position)) {
          const Vector4d new_p{proj.x(), proj.y(), proj.z(), proj.w()};
          QEF moved_Q = Q_of_edge_to_contract.WithMinimizerMoveTo(new_p);

          vertex_Qs_[selected_vertex0] = moved_Q;
          vertex_Qs_[selected_vertex1] = moved_Q;

          const double sdf = proj.w();
          vertices_[selected_vertex0] = position;
          vertices_[selected_vertex1] = position;
          signed_distances_[selected_vertex0] = sdf;
          signed_distances_[selected_vertex1] = sdf;
        }
      }
    }
    if (iteration == next_report_iteration) {
      drake::log()->info(
          "iteration {}, max_edge_error = {}, "
          "num_total_edge_contraction = {}, "
          "num_tet_deleted_ = {}, "
          "num_input_tetrahedra - num_tet_deleted_ = {}",
          iteration, max_edge_error, num_total_edge_contraction,
          num_tet_deleted_, num_input_tetrahedra - num_tet_deleted_);
      if (found_edge_to_contract) {
        drake::log()->info("Contracted edge with min_edge_error = {}",
                           min_edge_error);
      }
      next_report_iteration += report_period;
    }
  }  // while(!tetrahedron_queue_.empty()

  drake::log()->info("");
  drake::log()->info("End iterations: iteration = {}", iteration);
  drake::log()->info("");
  drake::log()->info("Number of edge contractions = {}",
                     num_total_edge_contraction);
  drake::log()->info("Number of deleted tetrahedra = {}", num_tet_deleted_);

  std::vector<VolumeElement> valid_tetrahedra;
  for (int tet = 0; tet < ssize(tetrahedra_); ++tet) {
    if (!is_tet_deleted_[tet]) {
      valid_tetrahedra.push_back(tetrahedra_[tet]);
    }
  }

  const VolumeMesh<double> coarsen_mesh =
      CompactMesh(valid_tetrahedra, vertices_);

  drake::log()->info(
      "Number of output tetrahedra: coarsen_mesh.num_elements() = {}",
      coarsen_mesh.num_elements());
  drake::log()->info("coarsen_mesh.CalcMinTetrahedralVolume() = {}",
                     coarsen_mesh.CalcMinTetrahedralVolume());

  return coarsen_mesh;
}

#if 0
VolumeMesh<double> VolumeMeshCoarsener::coarsen(double fraction) {
  const int num_input_tetrahedra = tetrahedra_.size();
  drake::log()->info("Number of input tetrahedra = {}.", num_input_tetrahedra);
  const int target_num_tetrahedra =
      static_cast<int>(fraction * num_input_tetrahedra);
  drake::log()->info("Target number of tetrahedra = {}.",
                     target_num_tetrahedra);
  drake::log()->info("input_mesh_.CalcMinTetrahedralVolume() = {}.",
                     input_mesh_.CalcMinTetrahedralVolume());

  InitializeVertexQEFs();

  int num_total_edge_contraction = 0;
  // Each edge contraction should remove at least two tetrahedra.
  const int num_iterations = (num_input_tetrahedra - target_num_tetrahedra) / 2;
  constexpr int kNumReports = 10;
  const int report_period =
      num_iterations >= kNumReports ? num_iterations / kNumReports : 1;
  int num_perform_iterations = 0;
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    // If we exhaust all iterations, both num_performed_iterations and
    // `iteration` variables will become num_iterations.
    // If we exit early, num_performed_iterations will be 1 + `iteration`
    // because `iteration` starts at 0 not 1.
    ++num_perform_iterations;
    if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
      break;
    }
    bool found_edge_to_contract = false;
    QEF Q_of_edge_to_contract;
    int selected_vertex0 = -1;
    int selected_vertex1 = -1;
    Vector3d optimal_point{0, 0, 0};
    double optimal_value{0};
    // Vertices of the last candidate edges to contract that got rejection.
    int last_rejected_vi = -1;
    int last_rejected_vj = -1;
    double min_edge_error = std::numeric_limits<double>::max();
    double max_edge_error = std::numeric_limits<double>::min();
    for (int tet = 0; tet < num_input_tetrahedra; ++tet) {
      if (num_input_tetrahedra - num_tet_deleted_ <= target_num_tetrahedra) {
        break;
      }
      if (is_tet_deleted_[tet]) {
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

        QEF edge_Q = QEF::Combine(vertex_Qs_[vi], vertex_Qs_[vj]);
        Vector3d xyz{edge_Q.p.x(), edge_Q.p.y(), edge_Q.p.z()};
        double value = edge_Q.p.w();
        if (edge_Q.e > max_edge_error) {
          max_edge_error = edge_Q.e;
        }
        if (IsEdgeContractible(vi, vj, xyz, value)) {
          if (edge_Q.e < min_edge_error) {
            found_edge_to_contract = true;
            min_edge_error = edge_Q.e;
            Q_of_edge_to_contract = edge_Q;
            selected_vertex0 = vi;
            selected_vertex1 = vj;
            optimal_point = xyz;
            optimal_value = value;
          }
        } else {
          last_rejected_vi = vi;
          last_rejected_vj = vj;
        }
      }  // for (std::pair<int, int> ij
    }  // for (int tet
    if (found_edge_to_contract) {
      ContractEdge(selected_vertex0, selected_vertex1, optimal_point,
                   optimal_value);
      vertex_Qs_[selected_vertex0] = Q_of_edge_to_contract;
      vertex_Qs_[selected_vertex1] = Q_of_edge_to_contract;
      ++num_total_edge_contraction;

      if (volume_to_boundary_.contains(selected_vertex0) ||
          volume_to_boundary_.contains(selected_vertex1)) {
        double offset_distance = 0;
        int count_num_boundary_vertices = 0;
        if (volume_to_boundary_.contains(selected_vertex0)) {
          offset_distance += signed_distances_[selected_vertex0];
          ++count_num_boundary_vertices;
        }
        if (volume_to_boundary_.contains(selected_vertex1)) {
          offset_distance += signed_distances_[selected_vertex1];
          ++count_num_boundary_vertices;
        }
        DRAKE_THROW_UNLESS(count_num_boundary_vertices != 0);
        offset_distance /= count_num_boundary_vertices;

        const Vector4d proj = CalcProjectionToOffsetSurface(
            optimal_point, original_boundary_, offset_distance);
        const Vector3d position{proj.x(), proj.y(), proj.z()};

        if (IsVertexMovable(selected_vertex0, position)) {
          const Vector4d new_p{proj.x(), proj.y(), proj.z(), proj.w()};
          QEF moved_Q = Q_of_edge_to_contract.WithMinimizerMoveTo(new_p);

          vertex_Qs_[selected_vertex0] = moved_Q;
          vertex_Qs_[selected_vertex1] = moved_Q;

          const double sdf = proj.w();
          vertices_[selected_vertex0] = position;
          vertices_[selected_vertex1] = position;
          signed_distances_[selected_vertex0] = sdf;
          signed_distances_[selected_vertex1] = sdf;
        }
      }
    } else {  // !found_edge_to_contract
      WriteTetrahedraBeforeEdgeContraction(last_rejected_vi, last_rejected_vj,
                                           "coarsen_reject");
    }

    if (iteration % report_period == 0) {
      drake::log()->info("");
      drake::log()->info("iteration {}, max_edge_error = {}", iteration,
                         max_edge_error);
      if (found_edge_to_contract) {
        drake::log()->info("Contracted edge with min_edge_error = {}",
                           min_edge_error);
      }
      drake::log()->info(
          "num_total_edge_contraction = {}, "
          "num_tet_deleted_ = {}, "
          "num_input_tetrahedra - num_tet_deleted_ = {}",
          num_total_edge_contraction, num_tet_deleted_,
          num_input_tetrahedra - num_tet_deleted_);
    }
    if (!found_edge_to_contract) {
      break;
    }
  }  // for iteration
  drake::log()->info("");
  drake::log()->info("End iterations: num_perform_iterations = {}",
                     num_perform_iterations);
  drake::log()->info("");
  drake::log()->info("Number of edge contractions = {}",
                     num_total_edge_contraction);
  drake::log()->info("Number of deleted tetrahedra = {}", num_tet_deleted_);

  std::vector<VolumeElement> valid_tetrahedra;
  for (int tet = 0; tet < ssize(tetrahedra_); ++tet) {
    if (!is_tet_deleted_[tet]) {
      valid_tetrahedra.push_back(tetrahedra_[tet]);
    }
  }

  const VolumeMesh<double> coarsen_mesh =
      CompactMesh(valid_tetrahedra, vertices_);

  drake::log()->info(
      "Number of output tetrahedra: coarsen_mesh.num_elements() = {}",
      coarsen_mesh.num_elements());
  drake::log()->info("coarsen_mesh.CalcMinTetrahedralVolume() = {}",
                     coarsen_mesh.CalcMinTetrahedralVolume());

  return coarsen_mesh;
}
#endif

VolumeMesh<double> VolumeMeshCoarsener::CompactMesh(
    const std::vector<VolumeElement>& tetrahedra,
    const std::vector<Vector3d>& vertices) const {
  std::vector<Vector3d> new_vertices;
  std::vector<VolumeElement> new_tetrahedra;
  // old_to_new[u] = v means the old u-th vertex in vertices corresponds to
  // the new v-th vertex in new_vertices.
  std::unordered_map<int, int> old_to_new;
  for (const VolumeElement old_tet : tetrahedra) {
    for (int i = 0; i < 4; ++i) {
      int old_vertex_index = old_tet.vertex(i);
      if (!old_to_new.contains(old_vertex_index)) {
        old_to_new[old_vertex_index] = new_vertices.size();
        new_vertices.push_back(vertices.at(old_vertex_index));
      }
    }
    new_tetrahedra.emplace_back(
        old_to_new.at(old_tet.vertex(0)), old_to_new.at(old_tet.vertex(1)),
        old_to_new.at(old_tet.vertex(2)), old_to_new.at(old_tet.vertex(3)));
  }

  return {std::move(new_tetrahedra), std::move(new_vertices)};
}

//--------------------------------------------------------
// Visual debugging facilities
//--------------------------------------------------------

VolumeMesh<double> VolumeMeshCoarsener::DebugTetrahedraOfVertex(int v0) const {
  std::vector<VolumeElement> tetrahedra_to_write;
  for (const int tet : vertex_to_tetrahedra_.at(v0)) {
    tetrahedra_to_write.push_back(tetrahedra_.at(tet));
  }
  return CompactMesh(tetrahedra_to_write, vertices_);
}

void VolumeMeshCoarsener::WriteTetrahedraOfVertex(
    int v0, const std::string& file_name) {
  WriteVolumeMeshToVtk(file_name, DebugTetrahedraOfVertex(v0),
                       "VolumeMeshCoarsener::WriteTetrahedraOfVertex");
}

void VolumeMeshCoarsener::WriteTetrahedraOfFirstExcludeSecond(
    int first_vertex, int second_vertex, const std::string& file_name) {
  const std::vector<int> tetrahedra_of_both =
      VolumeMeshRefiner::GetTetrahedraOnEdge(first_vertex, second_vertex);
  std::vector<VolumeElement> tetrahedra_to_write;
  for (const int tet : vertex_to_tetrahedra_.at(first_vertex)) {
    if (std::find(tetrahedra_of_both.cbegin(), tetrahedra_of_both.cend(),
                  tet) == tetrahedra_of_both.end()) {
      tetrahedra_to_write.push_back(tetrahedra_.at(tet));
    }
  }
  WriteVolumeMeshToVtk(
      file_name, CompactMesh(tetrahedra_to_write, vertices_),
      "VolumeMeshCoarsener::WriteTetrahedraOfFirstExcludeSecond");
}

VolumeMesh<double> VolumeMeshCoarsener::DebugTetrahedraOfBothVertex(
    int v0, int v1) const {
  std::vector<VolumeElement> tetrahedra_to_write;
  for (const int tet : VolumeMeshRefiner::GetTetrahedraOnEdge(v0, v1)) {
    tetrahedra_to_write.push_back(tetrahedra_.at(tet));
  }
  return CompactMesh(tetrahedra_to_write, vertices_);
}

VolumeMesh<double> VolumeMeshCoarsener::DebugTetrahedraOfEitherVertices(
    int v0, int v1) const {
  std::set<int> tet_indices;
  for (const int tet : vertex_to_tetrahedra_.at(v0)) {
    tet_indices.insert(tet);
  }
  for (const int tet : vertex_to_tetrahedra_.at(v1)) {
    tet_indices.insert(tet);
  }
  std::vector<VolumeElement> tetrahedra_to_write;
  for (const int tet : tet_indices) {
    tetrahedra_to_write.push_back(tetrahedra_.at(tet));
  }
  return CompactMesh(tetrahedra_to_write, vertices_);
}

void VolumeMeshCoarsener::WriteTetrahedraOfBothVertices(
    int first_vertex, int second_vertex, const std::string& file_name) {
  WriteVolumeMeshToVtk(file_name,
                       DebugTetrahedraOfBothVertex(first_vertex, second_vertex),
                       "VolumeMeshCoarsener::WriteTetrahedraOfBothVertices");
}

void VolumeMeshCoarsener::WriteTetrahedraBeforeEdgeContraction(
    int v0, int v1, const std::string& prefix_file_name) {
  WriteTetrahedraOfFirstExcludeSecond(
      v0, v1,
      fmt::format("{}_before_v{}_not_v{}_tets.vtk", prefix_file_name, v0, v1));
  WriteTetrahedraOfFirstExcludeSecond(
      v1, v0,
      fmt::format("{}_before_v{}_not_v{}_tets.vtk", prefix_file_name, v1, v0));
  WriteTetrahedraOfBothVertices(
      v0, v1,
      fmt::format("{}_before_v{}_and_v{}_tets.vtk", prefix_file_name, v0, v1));
}

void VolumeMeshCoarsener::WriteSavedTetrahedraBeforeEdgeContraction(
    int v0, int v1, const VolumeMesh<double>& tetrahedra_on_edge_v0_v1,
    const std::string& prefix_file_name) {
  WriteVolumeMeshToVtk(
      fmt::format("{}_before_v{}_and_v{}.vtk", prefix_file_name, v0, v1),
      tetrahedra_on_edge_v0_v1,
      "VolumeMeshCoarsener::WriteTetrahedraBeforeEdgeContraction");
}

void VolumeMeshCoarsener::WriteSavedTetrahedraOfVertexBeforeEdgeContration(
    int v, const VolumeMesh<double>& tetrahedra_on_vertex,
    const std::string& prefix_file_name) {
  WriteVolumeMeshToVtk(
      fmt::format("{}_before_v{}.vtk", prefix_file_name, v),
      tetrahedra_on_vertex,
      "VolumeMeshCoarsener::WriteSavedTetrahedraOfVertexBeforeEdgeContration");
}

void VolumeMeshCoarsener::
    WriteSavedTetrahedraOfEitherVerticesBeforeEdgeContraction(
        int v0, int v1, const VolumeMesh<double>& tetrahedra_on_either_vertices,
        const std::string& prefix_file_name) {
  WriteVolumeMeshToVtk(
      fmt::format("{}_before_v{}_or_v{}.vtk", prefix_file_name, v0, v1),
      tetrahedra_on_either_vertices,
      "VolumeMeshCoarsener::"
      "WriteSavedTetrahedraOfEitherVerticesBeforeEdgeContraction");
}

void VolumeMeshCoarsener::WriteTetrahedraAfterEdgeContraction(
    const int v0, const int v1, const std::string& prefix_file_name) {
  WriteTetrahedraOfVertex(
      v0, fmt::format("{}_after_v{}_v{}_tets.vtk", prefix_file_name, v0, v1));
}

void VolumeMeshCoarsener::LogAndWriteQ(int v, const QEF& v_Q,
                                       const std::string& prefix_file_name) {
  // B = 3x3 block of the spatial coordinates of the 4x4 A of Q.
  Matrix3d B = v_Q.A.Mat4d().block(0, 0, 3, 3);
  drake::log()->warn(
      fmt::format("v{}'s A's 3x3 spatial block =\n{}", v, fmt_eigen(B)));

  // This debugging tool is a very simplified version of the more rigorous
  // calculation in:
  //
  // template <typename T>
  // Vector3<double> RotationalInertia<T>::
  // CalcPrincipalMomentsAndMaybeAxesOfInertia(
  //    math::RotationMatrix<double>* principal_directions) const;
  //

  Eigen::SelfAdjointEigenSolver<Matrix3d> es;
  es.compute(B, Eigen::ComputeEigenvectors);
  DRAKE_THROW_UNLESS(es.info() == Eigen::Success);

  Vector3d lambdas = es.eigenvalues();
  drake::log()->warn(fmt::format("v{}'s A's spatial eigenvalues = {}", v,
                                 fmt_eigen(lambdas.transpose())));
  DRAKE_THROW_UNLESS(lambdas[0] >= 0);
  DRAKE_THROW_UNLESS(lambdas[1] >= 0);
  DRAKE_THROW_UNLESS(lambdas[2] >= 0);
  DRAKE_THROW_UNLESS(lambdas.maxCoeff() > 0);
  const Vector3d ev0 = es.eigenvectors().col(0);
  const Vector3d ev1 = es.eigenvectors().col(1);
  const Vector3d ev2 = ev0.cross(ev1).normalized();
  drake::log()->warn(fmt::format("v{}'s A's 1st eigen-vectorᵀ = {}", v,
                                 fmt_eigen(ev0.transpose())));
  drake::log()->warn(fmt::format("v{}'s A's 2nd eigen-vectorᵀ = {}", v,
                                 fmt_eigen(ev1.transpose())));
  drake::log()->warn(fmt::format("v{}'s A's 3rd eigen-vectorᵀ = {}", v,
                                 fmt_eigen(ev2.transpose())));
  // E is the frame of the ellipsoid.
  const RotationMatrixd R_WE =
      RotationMatrixd::MakeFromOrthonormalColumns(ev0, ev1, ev2);
  const RigidTransformd X_WE(R_WE, Vector3d(v_Q.p.x(), v_Q.p.y(), v_Q.p.z()));

  const double scale = 0.001;
  const Ellipsoid ellipsoid_M(scale * lambdas);
  // Start in frame E of the ellipsoid.
  VolumeMesh<double> ellipsoid_mesh = MakeEllipsoidVolumeMesh<double>(
      ellipsoid_M, 0.0002, TessellationStrategy::kDenseInteriorVertices);
  // Transform to World frame.
  ellipsoid_mesh.TransformVertices(X_WE);
  WriteVolumeMeshToVtk(
      fmt::format("{}_v{}_QEF_Ellipsoid.vtk", prefix_file_name, v),
      ellipsoid_mesh, "VolumeMeshCoarsener::LogAndWriteQ");
}

VolumeMesh<double> VolumeMeshCoarsener::HackNegativeToPositiveVolume(
    const VolumeMesh<double>& mesh, int* num_negative_volume_tetrahedra) {
  std::vector<VolumeElement> positive_tetrahedra;
  int count_skip_or_swap = 0;
  for (int e = 0; e < mesh.num_elements(); ++e) {
    VolumeElement tet = mesh.tetrahedra()[e];
    const double tet_volume = mesh.CalcTetrahedronVolume(e);
    if (std::abs(tet_volume) < 1e-13) {
      ++count_skip_or_swap;
      continue;
    }
    if (mesh.CalcTetrahedronVolume(e) < 0) {
      int v0 = tet.vertex(0);
      int v1 = tet.vertex(1);
      int v2 = tet.vertex(2);
      int v3 = tet.vertex(3);
      // Swap v0 and v1 to flip the signed volume.
      tet = VolumeElement(v1, v0, v2, v3);
      ++count_skip_or_swap;
    }
    positive_tetrahedra.push_back(tet);
  }
  if (num_negative_volume_tetrahedra != nullptr) {
    *num_negative_volume_tetrahedra = count_skip_or_swap;
  }
  return {std::move(positive_tetrahedra),
          std::vector<Vector3d>(mesh.vertices())};
}

//-------------------------------------------------------------------------
// Treatment of Triangulated Boundary Surface
//-------------------------------------------------------------------------

Vector4d CalcProjectionToOffsetSurface(const Vector3d& p,
                                       const MeshDistanceBoundary& boundary,
                                       const double offset_distance) {
  const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
      p, boundary.tri_mesh(), boundary.tri_bvh(),
      std::get<FeatureNormalSet>(boundary.feature_normal()));

  Vector3d xyz =
      d.nearest_point + offset_distance * d.gradient.stableNormalized();
  return {xyz.x(), xyz.y(), xyz.z(), offset_distance};
}

//-------------------------------------------------------------------------
// Quadric Error Metric Functions
//-------------------------------------------------------------------------

//***************************************************************************
// Some of these numerical routines follow Section 3.4 "Numerical Issues" in:
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

Vector4d QEF::CalcCombinedMinimizer(const QEF& Q1, const QEF& Q2) {
  const SymMat4& A1 = Q1.A;
  const SymMat4& A2 = Q2.A;
  const Vector4d& p1 = Q1.p;
  const Vector4d& p2 = Q2.p;

  // Solve for x in (A₁+A₂)x = (A₁p₁ + A₂p₂).
  const SymMat4 A = A1 + A2;
  const Vector4d b = A1 * p1 + A2 * p2;
  // The example in https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
  // specified ComputeThinU and ComputeThinV in the template arguments:
  // Eigen::JacobiSVD<Matrix4d, Eigen::ComputeThinU | Eigen::ComputeThinV>.
  // However, I found that I need to do it in the constructor arguments instead.
  // Otherwise, the solve() gave me random noises of numbers near zero.
  Eigen::JacobiSVD<Matrix4d> svd(A.Mat4d(),
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vector4d x = svd.solve(b);
  DRAKE_THROW_UNLESS(svd.info() == Eigen::Success);

  // if (!(x.norm() < 1e10)) {
  //   const Matrix4d& M = A.Mat4d();
  //   drake::log()->warn(fmt::format("M = \n{}", fmt_eigen(M)));
  //   drake::log()->warn(fmt::format("bᵀ = \n{}", fmt_eigen(b.transpose())));
  //   throw std::logic_error(
  //       fmt::format("QEF::CalcCombinedMinimizer: xᵀ = \n{}",
  //                   fmt_eigen(x.transpose())));
  //   DRAKE_THROW_UNLESS(x.norm() < 1e10);
  // }

  // Constrain x to be between p1 and p2 inclusively.
  // Without clamping, sometimes got nan or abnormal numbers like 3e+138
  // or nan.
  // {
  //   const Vector4d p12 = p2 - p1;
  //   if (p12.norm() < 1e-14) {
  //     x = p1;
  //   } else {
  //     const double w = (x - p1).dot(p12) / p12.dot(p12);
  //     if (w <= 0) {
  //       x = p1;
  //     } else if (w < 1) {
  //       x = (1 - w) * p1 + w * p2;
  //     } else {
  //       x = p2;
  //     }
  //   }
  // }

  return x;

#if 0
  // We will use a custom conjugate gradient method to solve for p in
  // (A₁+A₂)p = (A₁p₁ + A₂p₂). See the algorithm in Fig. 5 of [Huy2007],
  // page 7.
  //
  // The paper [Huy2007] uses κₘₐₓ = 10⁴. The VTK implementation uses
  // κₘₐₓ = 10³.  For now, we follow the VTK.
  constexpr double kKappaMax = 1e3;
  // tr(A) / (n * κₘₐₓ) with n = 4 for tetrahedral meshes.
  const double exit_threshold = A.trace() / (4 * kKappaMax);

  // Solve for x in (A₁+A₂)x = (A₁p₁ + A₂p₂) with the initial guess of x as
  // (p₁+p₂)/2.
  Vector4d x = (p1 + p2) / 2;

  // With the initial guess x = (p₁+p₂)/2, we have the initial residual:
  //      r = (A₁p₁+A₂p₂) - (A₁+A₂)x
  //        = (A₁p₁+A₂p₂) - (A₁+A₂)(p₁+p₂)/2
  //        = A₁p₁/2 + A₂p₂/2 - A₁p₂/2 - A₂p₁/2
  //        = A₁p₁/2 - A₂p₁/2 + A₂p₂/2 - A₁p₂/2
  //        = (A₁-A₂)p₁/2 - (A₁-A₂)p₂/2
  //        = (A₁ - A₂)(p₁-p₂)/2
  Vector4d r = (A1 - A2) * (p1 - p2) / 2;
  // Approximated gradient direction.
  Vector4d d = Vector4d::Zero();
  // At most four iterations for a 4x4 linear system.
  for (int k = 0; k < 4; ++k) {
    const double s = r.squaredNorm();  // s = ∥r∥²
    if (s <= 0) {
      break;
    }
    d += (r / s);                   // d = d + r/∥r∥²
    const Vector4d q = A * d;       // q = Ad
    const double t = d.dot(q);      // t = dᵀAd
    if (s * t <= exit_threshold) {  // Check ∥r∥²(dᵀAd)  to exit.
      break;
    }
    r -= (q / t);  // Update residual vector, r = r - Ad/(dᵀAd)
    x += (d / t);  // Update solution vector, x = x + d/(dᵀAd)
  }

  // Constrain x to be between p1 and p2 inclusively.
  {
    const Vector4d p12 = p2 - p1;
    if (p12.norm() < 1e-14) {
      x = p1;
    } else {
      const double w = (x - p1).dot(p12) / p12.dot(p12);
      if (w <= 0) {
        x = p1;
      } else if (w < 1) {
        x = (1 - w) * p1 + w * p2;
      } else {
        x = p2;
      }
    }
  }

  return x;
#endif
}

double QEF::CalcCombinedMinError(const QEF& Q1, const QEF& Q2,
                                 const Eigen::Vector4d& p) {
  const SymMat4& A1 = Q1.A;
  const SymMat4& A2 = Q2.A;
  const Eigen::Vector4d& p1 = Q1.p;
  const Eigen::Vector4d& p2 = Q2.p;
  const double e1 = Q1.e;
  const double e2 = Q2.e;
  return e1 + e2 + (p - p1).dot(A1 * (p - p1)) + (p - p2).dot(A2 * (p - p2));
}

void VolumeMeshCoarsener::UpdateVerticesQuadricsFromTet(int tet) {
  DRAKE_THROW_UNLESS(0 <= tet && tet < ssize(tetrahedra_));
  const int v0 = tetrahedra_[tet].vertex(0);
  const int v1 = tetrahedra_[tet].vertex(1);
  const int v2 = tetrahedra_[tet].vertex(2);
  const int v3 = tetrahedra_[tet].vertex(3);
  DRAKE_THROW_UNLESS(0 <= v0 && v0 < ssize(vertex_Qs_));
  DRAKE_THROW_UNLESS(0 <= v1 && v1 < ssize(vertex_Qs_));
  DRAKE_THROW_UNLESS(0 <= v2 && v2 < ssize(vertex_Qs_));
  DRAKE_THROW_UNLESS(0 <= v3 && v3 < ssize(vertex_Qs_));
  // Coefficients of each vector has unit in meters.
  // The coordinates (x,y,z) are in meters.
  // The signed distance is also in meters.
  // After subtraction, we normalize the vectors for numerical stability,
  // and they become unit less.
  const Vector4d a = (vertex_Qs_[v1].p - vertex_Qs_[v0].p).stableNormalized();
  const Vector4d b = (vertex_Qs_[v2].p - vertex_Qs_[v0].p).stableNormalized();
  const Vector4d c = (vertex_Qs_[v3].p - vertex_Qs_[v0].p).stableNormalized();

  // n is the generalized cross product of the three unit Vector4d a, b, c.
  // See the determinant formula in Section 3.4 "Numerical Issues" of
  // [Huy2007].
  //
  //            | Fx Fy Fz Fw |
  //   n =  det | <--- a ---> | ; Fx,Fy,Fz,Fw are the basis vectors.
  //            | <--- b ---> |
  //            | <--- c ---> |
  //
  const Vector4d n =
      Vector4d{
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
      }
          .stableNormalized();
  // Outer product of 4-vector n gives the 4x4 symmetric matrix A.
  SymMat4 A = SymMat4::FromOuterProductOfVector4d(n);

  // Multiply A by the volume of the tetrahedron gives the fundamental
  // quadric matrix with the units of its coefficients in cubic meters.
  // const double tetrahedron_volume =
  //     CalcTetrahedronVolume(tet, tetrahedra_, vertices_);
  // DRAKE_THROW_UNLESS(tetrahedron_volume > 0);
  // A *= tetrahedron_volume;

  // Divide the tetrahedrn's quadric matrix to its 4 vertices equally.
  A /= 4.0;

  vertex_Qs_[v0].A += A;
  vertex_Qs_[v1].A += A;
  vertex_Qs_[v2].A += A;
  vertex_Qs_[v3].A += A;
}

// This numerical routine is from [Garland & Zhou] with reference code
// from vtkUnstructuredGridQuadricDecimationFace::UpdateQuadric().
//
void VolumeMeshCoarsener::UpdateVerticesQuadricsFromBoundaryFace(
    int boundary_tri) {
  DRAKE_THROW_UNLESS(0 <= boundary_tri &&
                     boundary_tri < support_boundary_mesh_.num_triangles());
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
  e1.stableNormalize();
  e2 = e2 - e1 * e2.dot(e1);
  e2.stableNormalize();

  // A = I - e1*e1ᵀ - e2*e2ᵀ
  // Unitless.
  SymMat4 A = SymMat4::Identity() - SymMat4::FromOuterProductOfVector4d(e1) -
              SymMat4::FromOuterProductOfVector4d(e2);

  // Multiply by area of the triangle and share (/3) it among three vertices.
  // After this step, A has units in square meters.
  // A *= support_boundary_mesh_.area(boundary_tri);

  A /= 3.0;

  // Set a large multiplicative weight to preserve boundary, so the code will
  // contract non-boundary edges first. vtkUnstructuredGridQuadricDecimation
  // uses 100.
  constexpr double kBoundaryWeight = 1e2;
  A *= kBoundaryWeight;

  vertex_Qs_.at(v0).A += A;
  vertex_Qs_.at(v1).A += A;
  vertex_Qs_.at(v2).A += A;
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
