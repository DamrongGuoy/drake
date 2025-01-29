#include <filesystem>

#include <gflags/gflags.h>

#include "drake/common/text_logging.h"
#include "drake/geometry/proximity/mesh_to_vtk.h"
#include "drake/geometry/proximity/obj_to_surface_mesh.h"
#include "drake/geometry/proximity/surface_to_volume_mesh.h"

namespace drake {
namespace geometry {
namespace {

int do_main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "[INPUT-FILE] [OUTPUT-FILE]\n"
      "\n"
      "mtetm ((M)ake (Tet)rahedral (M)esh)\n"
      "0. Read input .obj file into a triangle surface mesh.\n"
      "1. Convert the surface to tetrahedral mesh.\n"
      "(2. Refine tetrahedral mesh for hydroelastics (future).)\n"
      "3. Write output .vtk file.\n"
      "\n"
      "After building this tool from source; for example:\n"
      "    drake $ bazel build //geometry/proximity:mtetm\n"
      "Run it in your data directory; for example:\n"
      "    data $ /path/to/drake/bazel-bin/geometry/proximity/mtetm "
      "input.obj output.vtk");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
    drake::log()->error("missing input filename");
    return 1;
  }
  if (argc < 3) {
    drake::log()->error("missing output filename");
    return 2;
  }

  // Make cwd be what the user expected, not the runfiles tree.
  if (const char* path = std::getenv("BUILD_WORKING_DIRECTORY")) {
    const int error = ::chdir(path);
    if (error != 0) {
      log()->warn("Could not chdir to '{}'", path);
    }
  }

  TriangleSurfaceMesh<double> surface_mesh =
      ReadObjToTriangleSurfaceMesh(std::filesystem::path(argv[1]));
  VolumeMesh<double> volume_mesh = ConvertSurfaceToVolumeMesh(surface_mesh);

  std::filesystem::path outfile(argv[2]);
  internal::WriteVolumeMeshToVtk(outfile.string(), volume_mesh,
                                 "created by drake/geometry/proximity/mtetm");
  drake::log()->info(
      "wrote tetrahedral mesh to file '{}' with {} tets and {} vertices.",
      outfile.string(), volume_mesh.tetrahedra().size(),
      volume_mesh.vertices().size());
  return 0;
}

}  // namespace
}  // namespace geometry
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::geometry::do_main(argc, argv);
}
