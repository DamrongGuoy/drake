#include <filesystem>

#include <gflags/gflags.h>

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/make_empress_field.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/optimize_sdfield.h"

DEFINE_string(input, "", "input surface mesh (OBJ file).");
DEFINE_string(output, "",
              "output signed-distance field together with "
              "its tetrahedral mesh (VTK file).");
DEFINE_double(resolution, 0.02, "Resolution of the background grid (meters).");

namespace drake {
namespace geometry {
namespace internal {
namespace {

int do_main() {
  if (FLAGS_input.empty()) {
    drake::log()->error("missing input filename");
    drake::log()->info(gflags::ProgramUsage());
    return 1;
  }
  if (FLAGS_output.empty()) {
    drake::log()->error("missing output filename");
    drake::log()->info(gflags::ProgramUsage());
    return 2;
  }

  drake::log()->info(fmt::format(
      "\nCreate EmPress signed-distance field with resolution={} for input {}",
      FLAGS_resolution, FLAGS_input));

  // Make cwd be what the user expected, not the runfiles tree.
  if (const char* path = std::getenv("BUILD_WORKING_DIRECTORY")) {
    const int error = ::chdir(path);
    if (error != 0) {
      log()->warn("Could not chdir to '{}'", path);
    }
  }

  const TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(std::filesystem::path(FLAGS_input));

  const auto [mesh_EmPress_M, sdfield_EmPress_M] =
      MakeEmPressSDField(surface_mesh, FLAGS_resolution);

  const std::filesystem::path outfile(FLAGS_output);
  WriteVolumeMeshFieldLinearToVtk(
      outfile.string(), "SignedDistance(meter)", *sdfield_EmPress_M,
      "signed-distance field created by drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} vertices.",
      outfile.string(), sdfield_EmPress_M->mesh().tetrahedra().size(),
      sdfield_EmPress_M->mesh().num_vertices());

  SDFieldOptimizer optimizer(*sdfield_EmPress_M, surface_mesh);
  const struct SDFieldOptimizer::RelaxationParameters parameters{
      .alpha_exterior = 0.03,           // dimensionless
      .alpha = 0.3,                     // dimensionless
      .beta = 0.3,                      // dimensionless
      .target_boundary_distance = 1e-3  // meters
  };
  const VolumeMesh<double> optimized_mesh =
      optimizer.OptimizeVertex(parameters);
  const VolumeMeshFieldLinear<double, double> optimized_field =
      MakeEmPressSDField(optimized_mesh, surface_mesh);

  const std::filesystem::path optimized_file(FLAGS_output + "_optimize.vtk");
  WriteVolumeMeshFieldLinearToVtk(optimized_file.string(),
                                  "SignedDistance(meter)", optimized_field,
                                  "Optimized signed-distance field created by "
                                  "drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} "
      "vertices.",
      optimized_file.string(), optimized_field.mesh().tetrahedra().size(),
      optimized_field.mesh().num_vertices());

  const VolumeMesh<double> coarsen_mesh_M =
      CoarsenSdField(optimized_field, 0.1);
  VolumeMeshFieldLinear<double, double> coarsen_sdf_M =
      MakeEmPressSDField(coarsen_mesh_M, surface_mesh);
  const std::filesystem::path coarsen_file(FLAGS_output +
                                           "_optimize_coarsen.vtk");
  WriteVolumeMeshFieldLinearToVtk(coarsen_file.string(),
                                  "SignedDistance(meter)", coarsen_sdf_M,
                                  "Coarsen signded-distance field created by "
                                  "drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} "
      "vertices.",
      coarsen_file.string(), coarsen_sdf_M.mesh().tetrahedra().size(),
      coarsen_sdf_M.mesh().num_vertices());

  drake::log()->info(
      "The final approximated signed-distance field has RMS error = "
      "{} meters from the original surface.\n\n",
      CalcRMSErrorOfSDField(coarsen_sdf_M, surface_mesh));

  return 0;
}

}  // namespace
}  // namespace internal
}  // namespace geometry
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(R"""(

empress -input [file.obj] -output [file.vtk]
        -resolution [default 0.02, i.e., 2cm]

The option -helpshort will explain the above parameters.

Create a signed-distance field for EmPress (Embedded Pressure Field for
hydroelastic contact models) from an input watertight, manifold,
self-intersecting-free triangulated surface mesh.  It also creates
an embedded tetrahedral mesh enclosing the input surface to support the field.
The zero-level set implicit surface of the EmPress field tracks the
original input surface.

NOTE
====
The returned signed-distance field is generally negative inside, zero on,
and positive outside the geometry. This sign convention is consistent with
the struct drake::geometry::SignedDistanceToPoint.
)""");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::geometry::internal::do_main();
}
