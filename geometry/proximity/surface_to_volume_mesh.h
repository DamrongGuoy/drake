#pragma once

#include "drake/geometry/proximity/triangle_surface_mesh.h"
#include "drake/geometry/proximity/volume_mesh.h"

namespace drake {
namespace geometry {

VolumeMesh<double> ConvertSurfaceToVolumeMesh(
    const TriangleSurfaceMesh<double>& surface_mesh);

}  // namespace geometry
}  // namespace drake
