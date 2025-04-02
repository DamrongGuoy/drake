#include <filesystem>

#include <gflags/gflags.h>

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

DEFINE_string(input, "", "input signed distance field (VTK file).");
DEFINE_string(original, "", "original surface mesh (OBJ file).");
DEFINE_string(output, "",
              "base name for output files that will get _coarse.vtk suffix");
DEFINE_double(fraction, 0.1,
              "a number between 0 and 1 that control how aggressive\n"
              "the coarsening process is. The target number of\n"
              "tetrahedra equals the `fraction` times the number of\n"
              "the input tetrahedra. (default 0.1, i.e., shrink to 10%)");

namespace drake {
namespace geometry {
namespace internal {
namespace {

int do_main() {
  if (FLAGS_input.empty()) {
    drake::log()->error("missing input filename (VTK)");
    drake::log()->info(gflags::ProgramUsage());
    return 1;
  }
  if (FLAGS_original.empty()) {
    drake::log()->error("missing original surface-mesh file (OBJ)");
    drake::log()->info(gflags::ProgramUsage());
    return 2;
  }
  if (FLAGS_output.empty()) {
    drake::log()->error("missing output filename (VTK)");
    drake::log()->info(gflags::ProgramUsage());
    return 3;
  }

  // Make cwd be what the user expected, not the runfiles tree.
  if (const char* path = std::getenv("BUILD_WORKING_DIRECTORY")) {
    const int error = ::chdir(path);
    if (error != 0) {
      log()->warn("Could not chdir to '{}'", path);
    }
  }

  const TriangleSurfaceMesh<double> surface_mesh_M =
      ReadObjToTriangleSurfaceMesh(std::filesystem::path(FLAGS_original));
  const VolumeMesh<double> sdf_mesh_M =
      ReadVtkToVolumeMesh(std::filesystem::path(FLAGS_input));
  VolumeMeshFieldLinear<double, double> sdf_M{
      MakeScalarValuesFromVtkMesh<double>(Mesh{FLAGS_input}), &sdf_mesh_M};
  drake::log()->info(
      "Read signed distance field from {} with {} tets and {} vertices.",
      FLAGS_input, sdf_mesh_M.num_elements(), sdf_mesh_M.num_vertices());

  const VolumeMesh<double> coarse_mesh_M =
      CoarsenSdField(sdf_M, FLAGS_fraction);
  VolumeMeshFieldLinear<double, double> coarse_sdf_M =
      MakeEmPressSDField(coarse_mesh_M, surface_mesh_M);
  const std::filesystem::path coarse_file(FLAGS_output + "_coarse.vtk");
  WriteVolumeMeshFieldLinearToVtk(coarse_file.string(), "SignedDistance(meter)",
                                  coarse_sdf_M,
                                  "Coarsen signded-distance field created by "
                                  "drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed distance field to file '{}' with {} tets and {} "
      "vertices.",
      coarse_file.string(), coarse_sdf_M.mesh().tetrahedra().size(),
      coarse_sdf_M.mesh().num_vertices());
  drake::log()->info(
      "The coarsen signed distance field has RMS error = "
      "{} meters from the original surface.\n\n",
      CalcRMSErrorOfSDField(coarse_sdf_M, surface_mesh_M));

  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(R"""(

empress1 -input [file.vtk] -original [file.obj] -output [base_name]
        -fraction [number > 0, <= 1, e.g., 0.5]

The option -helpshort will explain the above parameters.

NOTE
====
The returned signed-distance field is generally negative inside, zero on,
and positive outside the geometry. This sign convention is consistent with
the struct drake::geometry::SignedDistanceToPoint.
)""");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::geometry::internal::do_main();
}
