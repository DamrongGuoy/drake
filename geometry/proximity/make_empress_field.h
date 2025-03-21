#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/aabb.h"
#include "drake/geometry/proximity/make_box_field.h"
#include "drake/geometry/proximity/make_box_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"
#include "drake/geometry/proximity/volume_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

/* Handy utilities.  Are they already available somewhere else?  */
Aabb CalcBoundingBox(const VolumeMesh<double>& mesh_M);
Aabb CalcBoundingBox(const TriangleSurfaceMesh<double>& mesh_M);

/* Given an input watertight, manifold, self-intersecting-free triangulated
 surface mesh, creates a signed-distance EmPress field, which is defined
 on an embedded tetrahedral mesh enclosing the input surface. The zero-level
 set implicit surface of the EmPress field tracks the original input surface.

 With EmPress, the compliant-hydroelastic contact model can consume an input
 surface mesh directly; users do not need to provide a body-fitted
 tetrahedral mesh.

 In the future, other input representations, e.g. Neural Implicit Signed
 Distance Field, should work too.

 @note   The returned EmPress signed-distance field is generally negative
         inside, zero on, and positive outside the geometry. This sign
         convention follows [Malladi1995] and consistent with the struct
         geometry::SignedDistanceToPoint.

 @param[in] mesh_M   Input watertight, manifold, self-intersecting-free
                     triangulated surface mesh.

 @param[in] grid_resolution  Resolution of the background grid.
 @param[in] out_offset  Include the level-set implicit surface at the
                        +out_offest distance outside the input surface.
                        It is similar to idea of margin.
 @param[in] in_offset   Include the level-set implicit surface at the
                        -in_offset distance inside the input surface.

 @pre both out_offset and in_offset are non-negative numbers.

 @return {mesh_EmPress_M, sdfield_EmPress_M}  The EmPress tetrahedral mesh
                     and the EmPress signed-distance field suitable for
                     hydroelastics. The field makes reference to the mesh.
                     Caller should keep both of them alive together.

 [Malladi1995]  Malladi, R.; Sethian, J.A.; Vemuri, B.C. (1995).
         "Shape modeling with front propagation: a level set approach".
         IEEE Transactions on Pattern Analysis and Machine Intelligence.
         17 (2): 158–175. CiteSeerX 10.1.1.33.2443.
         doi:10.1109/34.368173.
*/
std::pair<std::unique_ptr<VolumeMesh<double>>,
          std::unique_ptr<VolumeMeshFieldLinear<double, double>>>
MakeEmPressSDField(const TriangleSurfaceMesh<double>& mesh_M,
                   double grid_resolution = 0.02);

/* Overide of the above function that takes the support volume mesh and
 the original surface mesh. */
VolumeMeshFieldLinear<double, double> MakeEmPressSDField(
    const VolumeMesh<double>& support_mesh_M,
    const TriangleSurfaceMesh<double>& original_mesh_M);

/* Return a new coarsen tetrahedral mesh supporting the field.

 @param[in] sdf_M      signed distance field on a support mesh.
 @param[in] fraction   a number between 0 and 1 that control how aggressive
                       the coarsening process is. The target number of
                       tetrahedra equals the `fraction` times the number of
                       the given tetrahedra.  */
VolumeMesh<double> CoarsenSdField(
    const VolumeMeshFieldLinear<double, double>& sdf_M, double fraction);

/* Verifies EmPress signed-distance field.  Measure deviation between the
 signed-distance field and the original input surface.

 deviation = distance to the original input surface from a vertex of the
             approximated polygonal mesh of the zero-level set implicit
             surface of the EmPress signed-distance field.

 @note  Internally we exploit contact-surface computation to approximate the
        zero-level set implicit surface of the EmPress field.

 @return {min_deviation, mean_deviation, max_deviation}
         min_deviation = minimium absolute deviation, like 1 micron.
         mean_deviation = average absolute deviation, like 1mm.
         max_deviation = maximum absolute deviation, like 5mm.
*/
std::tuple<double, double, double> MeasureDeviationOfZeroLevelSet(
    const VolumeMeshFieldLinear<double, double>& sdfield_M,
    const TriangleSurfaceMesh<double>& original_M);

/* Calculates the root-mean-squared error of the zero-level set implicit
 surface of the signed-distance field `sdfield_M` from the original
 triangulated surface `original_M`.

 @return the RMS error in meters.

 @note  Internally we exploit contact-surface computation to approximate the
        zero-level set implicit surface of the signed-distance field.
*/
double CalcRMSErrorOfSDField(
    const VolumeMeshFieldLinear<double, double>& sdfield_M,
    const TriangleSurfaceMesh<double>& original_M);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
