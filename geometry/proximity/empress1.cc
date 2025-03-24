#include <filesystem>

#include <gflags/gflags.h>

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/make_mesh_from_vtk.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/optimize_sdfield.h"
#include "drake/geometry/proximity/vtk_to_volume_mesh.h"

DEFINE_string(input, "", "input signed distance field (VTK file).");
DEFINE_string(original, "", "original surface mesh (OBJ file).");
DEFINE_string(output, "",
              "base name for output files that will get "
              "_smooth.vtk and _smooth_coarse.vtk suffixes");
DEFINE_double(fraction, 0.5,
              "a number between 0 and 1 that control how aggressive\n"
              "the coarsening process is. The target number of\n"
              "tetrahedra equals the `fraction` times the number of\n"
              "the given tetrahedra. (default 0.5, i.e., 50%)");

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

  drake::log()->info(fmt::format(
      "\nAdapt the support mesh for signed distance field of input {}",
      FLAGS_input));

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
      "The input signed distance field has RMS error = "
      "{} meters from the original surface.",
      CalcRMSErrorOfSDField(sdf_M, surface_mesh_M));

  SDFieldOptimizer smoother(sdf_M, surface_mesh_M);
  const struct SDFieldOptimizer::RelaxationParameters parameters{
      .alpha_exterior = 0.03,           // dimensionless
      .alpha = 0.3,                     // dimensionless
      .beta = 0.3,                      // dimensionless
      .target_boundary_distance = 1e-3  // meters
  };
  const VolumeMesh<double> smooth_mesh_M = smoother.OptimizeVertex(parameters);
  const VolumeMeshFieldLinear<double, double> smooth_sdf_M =
      MakeEmPressSDField(smooth_mesh_M, surface_mesh_M);
  const std::filesystem::path smooth_file(FLAGS_output + "_smooth.vtk");
  WriteVolumeMeshFieldLinearToVtk(smooth_file.string(), "SignedDistance(meter)",
                                  smooth_sdf_M,
                                  "Smooth signed-distance field created by "
                                  "drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} vertices.",
      smooth_file.string(), smooth_sdf_M.mesh().tetrahedra().size(),
      smooth_sdf_M.mesh().num_vertices());
  drake::log()->info(
      "The smoothed signed distance field has RMS error = "
      "{} meters from the original surface.",
      CalcRMSErrorOfSDField(smooth_sdf_M, surface_mesh_M));

  const VolumeMesh<double> coarse_mesh_M =
      CoarsenSdField(smooth_sdf_M, FLAGS_fraction);
  VolumeMeshFieldLinear<double, double> coarse_sdf_M =
      MakeEmPressSDField(coarse_mesh_M, surface_mesh_M);
  const std::filesystem::path coarse_file(FLAGS_output + "_smooth_coarse.vtk");
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
