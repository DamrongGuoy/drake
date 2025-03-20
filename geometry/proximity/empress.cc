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

  TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(std::filesystem::path(FLAGS_input));

  const auto [mesh_EmPress_M, sdfield_EmPress_M] =
      MakeEmPressSDField(surface_mesh, FLAGS_resolution);

  std::filesystem::path outfile(FLAGS_output);
  internal::WriteVolumeMeshFieldLinearToVtk(
      outfile.string(), "SignedDistance(meter)", *sdfield_EmPress_M,
      "signed-distance field created by drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} vertices.",
      outfile.string(), sdfield_EmPress_M->mesh().tetrahedra().size(),
      sdfield_EmPress_M->mesh().num_vertices());

  SDFieldOptimizer optimizer(*sdfield_EmPress_M, surface_mesh);
  struct SDFieldOptimizer::RelaxationParameters parameters{
      .alpha_exterior = 0.03,           // dimensionless
      .alpha = 0.3,                     // dimensionless
      .beta = 0.3,                      // dimensionless
      .target_boundary_distance = 1e-3  // meters
  };
  VolumeMesh<double> optimized_mesh = optimizer.OptimizeVertex(parameters);
  VolumeMeshFieldLinear<double, double> optimized_field =
      MakeEmPressSDField(optimized_mesh, surface_mesh);

  std::filesystem::path optimized_file(FLAGS_output + "_optimize.vtk");
  WriteVolumeMeshFieldLinearToVtk(optimized_file.string(),
                                  "SignedDistance(meter)", optimized_field,
                                  "Optimized signed-distance field created by "
                                  "drake/geometry/proximity/empress");
  drake::log()->info(
      "wrote signed-distance field to file '{}' with {} tets and {} "
      "vertices.\n\n",
      optimized_file.string(), optimized_field.mesh().tetrahedra().size(),
      optimized_field.mesh().num_vertices());
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
