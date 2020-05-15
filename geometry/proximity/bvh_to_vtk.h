#pragma once

#include <string>

#include "drake/geometry/proximity/bounding_volume_hierarchy.h"

namespace drake {
namespace geometry {
namespace internal {

void WriteBVHToVtk(const std::string& file_name,
                   const BoundingVolumeHierarchy<VolumeMesh<double>>& bvh,
                   const std::string& title);

void WriteBVHToVtk(const std::string& file_name,
                   const BoundingVolumeHierarchy<SurfaceMesh<double>>& bvh,
                   const std::string& title);

}  // namespace internal
}  // namespace geometry
}  // namespace drake