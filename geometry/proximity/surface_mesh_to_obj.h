#pragma once

#include <string>

#include "drake/geometry/proximity/triangle_surface_mesh.h"

namespace drake {
namespace geometry {
namespace internal {

void WriteSurfaceMeshToObj(const std::string& file_name,
                           const TriangleSurfaceMesh<double>& mesh);

}  // namespace internal
}  // namespace geometry
}  // namespace drake
