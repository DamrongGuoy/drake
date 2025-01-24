#pragma once

#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/triangle_surface_mesh.h"

#include "objMesh.h"
#include "tetMesh.h"

namespace drake {
namespace geometry {

VolumeMesh<double> VegaFemTetMeshToDrakeVolumeMesh(
    const vegafem::TetMesh& vega_mesh);

vegafem::ObjMesh DrakeTriangleSurfaceMeshToVegaObjMesh(
    const TriangleSurfaceMesh<double>& drake_mesh);

VolumeMesh<double> VegaCdt(const TriangleSurfaceMesh<double>& surface_mesh);

}  // namespace geometry
}  // namespace drake
