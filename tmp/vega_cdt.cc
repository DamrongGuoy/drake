#include "vega_cdt.h"

#include <limits>

#include "vega_mesh_to_drake_mesh.h"
#include "tetMesher.h"

namespace drake {
namespace geometry {

VolumeMesh<double> VegaCdt(const TriangleSurfaceMesh<double>& surface_mesh) {
  vegafem::ObjMesh obj_mesh =
      DrakeTriangleSurfaceMeshToVegaObjMesh(surface_mesh);

  vegafem::TetMesher mesher;
  const double kUseThisForCoarsestMesh = std::numeric_limits<double>::max();
  vegafem::TetMesh* tet_mesh =
      mesher.compute(&obj_mesh, /*refinementQuality*/ kUseThisForCoarsestMesh);

  return VegaFemTetMeshToDrakeVolumeMesh(*tet_mesh);
}

}  // namespace geometry
}  // namespace drake