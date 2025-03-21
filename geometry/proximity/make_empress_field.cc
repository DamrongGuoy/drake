#include "drake/geometry/proximity/make_empress_field.h"

#include <iostream>

// To ease build system upkeep, we annotate VTK includes with their deps.
#include <vtkUnstructuredGridQuadricDecimation.h>  // vtkFiltersCore

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

using Eigen::Vector3d;
using Eigen::Vector4d;
using math::RigidTransformd;
using math::RollPitchYawd;

// TODO(DamrongGuoy):  Move functions out of the anonymous namespace for
//  appropriate unit testings.
namespace {
bool IsPointInTheBand(const Vector3d& p_MV, const MeshDistanceBoundary& sdf_M,
                      const double inner_offset, const double outer_offset,
                      int* count_positive, int* count_negative,
                      int* count_zero) {
  SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
      p_MV, sdf_M.tri_mesh(), sdf_M.tri_bvh(),
      std::get<FeatureNormalSet>(sdf_M.feature_normal()));
  if (d.signed_distance < 0) {
    ++(*count_negative);
  } else if (d.signed_distance > 0) {
    ++(*count_positive);
  } else {
    ++(*count_zero);
  }
  if (-inner_offset <= d.signed_distance && d.signed_distance <= outer_offset) {
    return true;
  } else {
    return false;
  }
}

// Check whether a tetrahedron touches the band between the outer offset and
// the inner offset distances, i.e., the implicit region of points Q where
// -inner_offset <= signed_distance(Q) <= +outer_offset.
//
// @param inner_offset specifies the implicit surface of the level set
//                     at signed distance = -inner_offset (inside).
// @param outer_offset specifies to the implicit surface of the level set
//                     at signed distance = +outer_offset (outside).
//
// @pre Both inner_offset and outer_offset are positive numbers.
//
// It's not enough to use signed-distance query at the vertices of the
// tetrahedra, especially for coarser grid resolution. Ideally we should check
// whether a tetrahedron intersects the band (at a vertex, in an edge, or in
// a face, or even (pedantically) enclosing the entire band). For now, it's too
// laborious, so we will just check some sampling points inside the tetrahedron.
bool IsTetInTheBand(const int tet, const VolumeMesh<double>& tetrahedral_mesh_M,
                    const double inner_offset, const double outer_offset,
                    const MeshDistanceBoundary& sdf_M) {
  DRAKE_THROW_UNLESS(inner_offset >= 0);
  DRAKE_THROW_UNLESS(outer_offset >= 0);

  // Check whether some points in the tetrahedron are in the band.
  // For simplicity, for now, we check only the four vertices and the edge's
  // midpoints.
  //
  // We also check whether some vertices have different sign. That's an
  // indicator that the tetrahedron cross the implicit surface of the zero-th
  // level set, i.e., the input surface mesh.

  // Four vertices.
  int count_positive = 0;
  int count_negative = 0;
  int count_zero = 0;  // Unlikely, but we want to be sure.
  for (int i = 0; i < 4; ++i) {
    const Vector3d& p_MV = tetrahedral_mesh_M.vertex(
        tetrahedral_mesh_M.tetrahedra()[tet].vertex(i));
    if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                         &count_positive, &count_negative, &count_zero)) {
      return true;
    }
    if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
      return true;
    }
  }

  // Six edges (3 + 2 + 1 + 0) : {(a,b) : 0 <= a < b < 4}.
  //  a: 0     | 1   | 2 | 3
  //  b: 1 2 3 | 2 3 | 3 | {}
  for (int a = 0; a < 4; ++a) {
    const Vector3d& p_MA = tetrahedral_mesh_M.vertex(
        tetrahedral_mesh_M.tetrahedra()[tet].vertex(a));
    for (int b = a + 1; b < 4; ++b) {
      const Vector3d p_AB_M = tetrahedral_mesh_M.edge_vector(tet, a, b);
      // V is the midpoint from A to B.
      const Vector3d p_MV = p_MA + p_AB_M / 2;
      if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                           &count_positive, &count_negative, &count_zero)) {
        return true;
      }
      if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
        return true;
      }
    }
  }

  // TODO(DamrongGuoy): Check the four face centers and one cell center?

  // TODO(DamrongGuoy): Find a more efficient way than bruteforce subsampling.
  //  Consider computing an intersection between this tetrahedron and the
  //  entire triangle surface mesh.  However, that would be more complicated
  //  than this simple subsampling.
  const int sampling_resolution = 5;
  const Vector3d& p_MA = tetrahedral_mesh_M.vertex(0);
  const Vector3d& p_MB = tetrahedral_mesh_M.vertex(1);
  const Vector3d& p_MC = tetrahedral_mesh_M.vertex(2);
  const Vector3d& p_MD = tetrahedral_mesh_M.vertex(3);
  // For example, sampling_resolution = 3 means 20 sub-samplings.
  // {(a,b,c,d) : a + b + c + d = 3, 0 <= a,b,c,d <= 3 }
  //            10          +     5     +   3   + 1 = 20
  // a: 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 | 2 2 2 | 3
  // b: 0 0 0 0 1 1 1 2 2 3 | 0 0 1 1 2 | 0 0 1 | 0
  // c: 0 1 2 3 0 1 2 0 1 0 | 0 1 0 1 0 | 0 1 0 | 0
  // d: 3 2 1 0 2 1 0 1 0 0 | 2 1 1 0 0 | 1 0 0 | 0
  for (int a = 0; a <= sampling_resolution; ++a) {
    for (int b = 0; b <= sampling_resolution - a; ++b) {
      for (int c = 0; c <= sampling_resolution - a - b; ++c) {
        int d = sampling_resolution - a - b - c;
        DRAKE_THROW_UNLESS(0 <= d && d <= sampling_resolution);
        const Vector3d p_MV =
            (a * p_MA + b * p_MB + c * p_MC + d * p_MD) / sampling_resolution;
        if (IsPointInTheBand(p_MV, sdf_M, inner_offset, outer_offset,
                             &count_positive, &count_negative, &count_zero)) {
          return true;
        }
        if (count_zero > 0 || (count_positive > 0 && count_negative > 0)) {
          return true;
        }
      }
    }
  }

  return false;
}

// TODO(DamrongGuoy):  Move MakeEmPressMesh out of anonymous namespace.
//  Right now, it's not convenient to expose MakeEmPressMesh because the
//  MeshDistanceBoundary is not available outside geometry/proximity.
//  If I expose MakeEmPressMesh(), I got build errors at the higher level;
//  for example, bazel build //tutorials/... couldn't find
//  mesh_distance_boundary.h.
VolumeMesh<double> MakeEmPressMesh(const MeshDistanceBoundary& input_M,
                                   const double grid_resolution) {
  const double out_offset = 1e-3;  // 1mm tolerance.
  const double in_offset = std::numeric_limits<double>::infinity();

  const Aabb fitted_box_M = CalcBoundingBox(input_M.tri_mesh());

  // The 10% expanded box's frame B is axis-aligned with the input mesh's
  // frame M.  Only their origins are different.
  // The Aabb stores its half-width vector, but the Box stores its "full
  // width", and hence the extra 2.0 factor below.
  const Box expanded_box_B(1.1 * 2.0 * fitted_box_M.half_width());
  const RigidTransformd X_MB(fitted_box_M.center());
  const VolumeMesh<double> background_B =
      MakeBoxVolumeMesh<double>(expanded_box_B, grid_resolution);
  // Translate/change to the common frame of reference.
  VolumeMesh<double> temp = background_B;
  temp.TransformVertices(X_MB);
  const VolumeMesh<double> background_M = temp;

  // Collect tetrahedra in the band between the inner offset and
  // the outer offset.
  std::vector<int> non_unique_tetrahedra;
  for (int tet = 0; tet < background_M.num_elements(); ++tet) {
    if (IsTetInTheBand(tet, background_M, in_offset, out_offset, input_M)) {
      non_unique_tetrahedra.push_back(tet);
    }
  }
  std::sort(non_unique_tetrahedra.begin(), non_unique_tetrahedra.end());
  auto last =
      std::unique(non_unique_tetrahedra.begin(), non_unique_tetrahedra.end());
  non_unique_tetrahedra.erase(last, non_unique_tetrahedra.end());
  // The tetrahedra are unique now; alias to a better name.
  const std::vector<int>& qualified_tetrahedra = non_unique_tetrahedra;

  // Vertex index from the background_M to the ImPress.
  std::unordered_map<int, int> old_to_new;
  std::vector<Vector3d> new_vertices;
  int count = 0;
  for (const int tet : qualified_tetrahedra) {
    for (int i = 0; i < 4; ++i) {
      const int v = background_M.tetrahedra()[tet].vertex(i);
      if (!old_to_new.contains(v)) {
        new_vertices.push_back(background_M.vertex(v));
        old_to_new[v] = count;
        ++count;
      }
    }
  }

  std::vector<VolumeElement> new_tetrahedra;
  for (const int tet : qualified_tetrahedra) {
    new_tetrahedra.emplace_back(
        old_to_new.at(background_M.tetrahedra()[tet].vertex(0)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(1)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(2)),
        old_to_new.at(background_M.tetrahedra()[tet].vertex(3)));
  }

  VolumeMesh<double> mesh_EmPress_M{std::move(new_tetrahedra),
                                    std::move(new_vertices)};

  return mesh_EmPress_M;

  // TODO(DamrongGuoy):  Would this optional step help or not?  Collect more
  //  tetrahedra that share vertices with previous qualified tetrahedra.
  //  It might mimic "snowflake" condition in [Stuart2013]; they claimed
  //  "snowflake" help.
  //  [Stuart2013] D.A. Stuart, J.A. Levine, B. Jones, and A.W. Bargteil.
  //  Automatic construction of coarse, high-quality tetrahedralizations that
  //  enclose and approximate surfaces for animation.
  //  Proceedings - Motion in Games 2013, MIG 2013, pages 191-199.
  //
}

// TODO(DamrongGuoy):  Move MakeEmPressSDField(...) out of anonymous namespace.
//  Right now, it's not convenient to expose it because the
//  MeshDistanceBoundary is not available outside geometry/proximity.
//  If I expose it, I got build errors at the higher level;
//  for example, bazel build //tutorials/... couldn't find
//  mesh_distance_boundary.h.
VolumeMeshFieldLinear<double, double> MakeEmPressSDField(
    const VolumeMesh<double>& support_mesh_M,
    const MeshDistanceBoundary& input_M) {
  std::vector<double> signed_distances;
  for (const Vector3d& tet_vertex : support_mesh_M.vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        tet_vertex, input_M.tri_mesh(), input_M.tri_bvh(),
        std::get<FeatureNormalSet>(input_M.feature_normal()));
    signed_distances.push_back(d.signed_distance);
  }
  return {std::move(signed_distances), &support_mesh_M};
}

}  // namespace

Aabb CalcBoundingBox(const VolumeMesh<double>& mesh_M) {
  Vector3d min_xyz{std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  Vector3d max_xyz{std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min()};
  for (const Vector3d& p_MV : mesh_M.vertices()) {
    for (int c = 0; c < 3; ++c) {
      if (p_MV(c) < min_xyz(c)) {
        min_xyz(c) = p_MV(c);
      }
      if (p_MV(c) > max_xyz(c)) {
        max_xyz(c) = p_MV(c);
      }
    }
  }
  return {(min_xyz + max_xyz) / 2, (max_xyz - min_xyz) / 2};
}

Aabb CalcBoundingBox(const TriangleSurfaceMesh<double>& mesh_M) {
  Vector3d min_xyz{std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  Vector3d max_xyz{std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::min()};
  for (const Vector3d& p_MV : mesh_M.vertices()) {
    for (int c = 0; c < 3; ++c) {
      if (p_MV(c) < min_xyz(c)) {
        min_xyz(c) = p_MV(c);
      }
      if (p_MV(c) > max_xyz(c)) {
        max_xyz(c) = p_MV(c);
      }
    }
  }
  return {(min_xyz + max_xyz) / 2, (max_xyz - min_xyz) / 2};
}

VolumeMeshFieldLinear<double, double> MakeEmPressSDField(
    const VolumeMesh<double>& support_mesh_M,
    const TriangleSurfaceMesh<double>& original_mesh_M) {
  // TODO(DamrongGuoy): Manage memory more carefully.  Right now it's easier
  //  to just making another copy of the input surface mesh and pass ownership
  //  to the MeshDistanceBoundary.
  return MakeEmPressSDField(
      support_mesh_M,
      MeshDistanceBoundary(TriangleSurfaceMesh<double>{original_mesh_M}));
}

std::pair<std::unique_ptr<VolumeMesh<double>>,
          std::unique_ptr<VolumeMeshFieldLinear<double, double>>>
MakeEmPressSDField(const TriangleSurfaceMesh<double>& mesh_M,
                   const double grid_resolution) {
  // TODO(DamrongGuoy): Manage memory more carefully.  Right now it's easier
  //  to just making another copy of the input surface mesh and pass ownership
  //  to the MeshDistanceBoundary.
  const MeshDistanceBoundary input_M(TriangleSurfaceMesh<double>{mesh_M});

  auto mesh_EmPress_M = std::make_unique<VolumeMesh<double>>(
      MakeEmPressMesh(input_M, grid_resolution));
  auto sdfield_EmPress_M =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          MakeEmPressSDField(*mesh_EmPress_M.get(), input_M));

  return {std::move(mesh_EmPress_M), std::move(sdfield_EmPress_M)};
}

std::pair<std::unique_ptr<VolumeMesh<double>>,
          std::unique_ptr<VolumeMeshFieldLinear<double, double>>>
CoarsenSdField(const VolumeMeshFieldLinear<double, double>& sdf_M,
               const TriangleSurfaceMesh<double>& surface_M,
               const double fraction) {
  unused(sdf_M);
  unused(surface_M);
  const double reduction_fraction = 1.0 - fraction;
  std::cout << reduction_fraction;

  return {nullptr, nullptr};
}

std::tuple<double, double, double> MeasureDeviationOfZeroLevelSet(
    const VolumeMeshFieldLinear<double, double>& sdfield_M,
    const TriangleSurfaceMesh<double>& original_M) {
  // TODO(DamrongGuoy): Manage memory in a better way.  The
  //  hydroelastic::SoftMesh want to take ownership of the input data through
  //  unique_ptr. For simplicity, we just copy both the mesh and the field.
  auto mesh_EmPress_M = std::make_unique<VolumeMesh<double>>(sdfield_M.mesh());
  auto sdfield_EmPress_M =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          std::vector<double>(sdfield_M.values()), mesh_EmPress_M.get());
  const hydroelastic::SoftMesh compliant_hydro_EmPress_M{
      std::move(mesh_EmPress_M), std::move(sdfield_EmPress_M)};

  const Aabb bounding_box_M = CalcBoundingBox(compliant_hydro_EmPress_M.mesh());
  // Frame B and frame M are axis-aligned. Only their origins are different.
  const Box box_B(2.0 * bounding_box_M.half_width());
  const RigidTransformd X_MB(bounding_box_M.center());
  auto box_mesh_B = std::make_unique<VolumeMesh<double>>(
      MakeBoxVolumeMeshWithMa<double>(box_B));
  // Using hydroelastic_modulus = 1e-6 will try to track the implicit surface
  // of the zero-level set of the pepper.
  auto box_field_B = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeBoxPressureField<double>(box_B, box_mesh_B.get(), 1e-6));
  const hydroelastic::SoftMesh compliant_box_B{std::move(box_mesh_B),
                                               std::move(box_field_B)};

  // The kTriangle argument makes the level0 surface include centroids of
  // contact polygons for more checks.
  std::unique_ptr<ContactSurface<double>> level0_M =
      ComputeContactSurfaceFromCompliantVolumes(
          GeometryId::get_new_id(), compliant_box_B, X_MB,
          GeometryId::get_new_id(), compliant_hydro_EmPress_M,
          RigidTransformd::Identity(),
          HydroelasticContactRepresentation::kTriangle);

  // TODO(DamrongGuoy): Manage memeory in a better way.  The
  //  MeshDistanceBoundary want to take ownership of the input surface mesh.
  //  For simplicity, we just make a copy.
  const MeshDistanceBoundary input_M{TriangleSurfaceMesh<double>(original_M)};

  double average_absolute_deviation = 0;
  double max_absolute_deviation = 0;
  double min_absolute_deviation = std::numeric_limits<double>::max();
  std::vector<double> deviation_values;
  for (const Vector3d& level0_vertex : level0_M->tri_mesh_W().vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        level0_vertex, input_M.tri_mesh(), input_M.tri_bvh(),
        std::get<FeatureNormalSet>(input_M.feature_normal()));
    deviation_values.push_back(d.signed_distance);
    const double absolute_deviation = std::abs(d.signed_distance);
    average_absolute_deviation += absolute_deviation;
    if (absolute_deviation < min_absolute_deviation) {
      min_absolute_deviation = absolute_deviation;
    }
    if (absolute_deviation > max_absolute_deviation) {
      max_absolute_deviation = absolute_deviation;
    }
  }
  average_absolute_deviation /= level0_M->tri_mesh_W().num_vertices();

  return {min_absolute_deviation, average_absolute_deviation,
          max_absolute_deviation};
}

double CalcRMSErrorOfSDField(
    const VolumeMeshFieldLinear<double, double>& sdfield_M,
    const TriangleSurfaceMesh<double>& original_M) {
  // TODO(DamrongGuoy): Manage memory in a better way.  The
  //  hydroelastic::SoftMesh wants to take ownership of the input
  //  signed-distance field through unique_ptr. For simplicity, we just
  //  copy both the mesh and the field.
  auto temporary_mesh_M =
      std::make_unique<VolumeMesh<double>>(sdfield_M.mesh());
  auto temporary_sdfield_M =
      std::make_unique<VolumeMeshFieldLinear<double, double>>(
          std::vector<double>(sdfield_M.values()), temporary_mesh_M.get());
  const hydroelastic::SoftMesh compliant_hydro_EmPress_M{
      std::move(temporary_mesh_M), std::move(temporary_sdfield_M)};

  const Aabb bounding_box_M = CalcBoundingBox(compliant_hydro_EmPress_M.mesh());
  // Frame B and frame M are axis-aligned. Only their origins are different.
  const Box box_B(2.0 * bounding_box_M.half_width());
  const RigidTransformd X_MB(bounding_box_M.center());
  auto box_mesh_B = std::make_unique<VolumeMesh<double>>(
      MakeBoxVolumeMeshWithMa<double>(box_B));
  // Using hydroelastic_modulus = 1e-6 will try to track the implicit surface
  // of the zero-level set of the pepper.
  auto box_field_B = std::make_unique<VolumeMeshFieldLinear<double, double>>(
      MakeBoxPressureField<double>(box_B, box_mesh_B.get(), 1e-6));
  const hydroelastic::SoftMesh compliant_box_B{std::move(box_mesh_B),
                                               std::move(box_field_B)};

  // The kTriangle argument makes the level0 surface include centroids of
  // contact polygons for more checks.
  std::unique_ptr<ContactSurface<double>> level0_M =
      ComputeContactSurfaceFromCompliantVolumes(
          GeometryId::get_new_id(), compliant_box_B, X_MB,
          GeometryId::get_new_id(), compliant_hydro_EmPress_M,
          RigidTransformd::Identity(),
          HydroelasticContactRepresentation::kTriangle);

  // TODO(DamrongGuoy): Manage memeory in a better way.  The
  //  MeshDistanceBoundary want to take ownership of the input surface mesh.
  //  For simplicity, we just make a copy.
  const MeshDistanceBoundary input_M{TriangleSurfaceMesh<double>(original_M)};

  double accumulate_squared_error = 0;
  for (const Vector3d& level0_vertex : level0_M->tri_mesh_W().vertices()) {
    const SignedDistanceToSurfaceMesh d = CalcSignedDistanceToSurfaceMesh(
        level0_vertex, input_M.tri_mesh(), input_M.tri_bvh(),
        std::get<FeatureNormalSet>(input_M.feature_normal()));
    accumulate_squared_error += d.signed_distance * d.signed_distance;
  }

  return std::sqrt(accumulate_squared_error /
                   level0_M->tri_mesh_W().num_vertices());
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
