#include "drake/geometry/proximity/surface_to_volume_mesh.h"

#include <limits>

#include "drake/tmp/vega_cdt.h"

namespace drake {
namespace geometry {

VolumeMesh<double> ConvertSurfaceToVolumeMesh(
    const TriangleSurfaceMesh<double>& surface_mesh) {
  return VegaCdt(surface_mesh);
}

}  // namespace geometry
}  // namespace drake
