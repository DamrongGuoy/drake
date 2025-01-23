#pragma once

#include "drake/geometry/proximity/volume_mesh.h"
#include "tetMesh.h"


namespace drake {
namespace geometry {

VolumeMesh<double> VegaFemTetMeshToDrakeVolumeMesh(
    const vegafem::TetMesh& vega_mesh);

}  // namespace geometry
}  // namespace drake
