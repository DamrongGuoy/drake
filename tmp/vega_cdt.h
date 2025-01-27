#pragma once

#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace geometry {

VolumeMesh<double> VegaCdt(const TriangleSurfaceMesh<double>& surface_mesh);

}  // namespace geometry
}  // namespace drake