#include "drake/geometry/proximity/bvh_to_vtk.h"

#include <fstream>
#include <iostream>

#include <fmt/format.h>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace {

using Eigen::Vector3d;

void WriteVtkHeader(std::ofstream& out, const std::string& title) {
  out << "# vtk DataFile Version 3.0\n";
  out << title << std::endl;
  out << "ASCII\n";
  // An extra blank line makes the file more human readable.
  out << std::endl;
}

// Return the height of the `node`.
// depth = number of edges to the root node. root.depth = 0
// height = largest number of edges to a leaf. leaf.height = 0
template <typename MeshType>
int GetAllNodes(const BvNode<MeshType>& node, int depth,
                 std::vector<const BvNode<MeshType>*>& all_nodes,
                 std::vector<int>& all_depths,
                 std::vector<int>& all_heights) {
  all_nodes.push_back(&node);
  all_depths.push_back(depth);
  // Remember the node index and initialize its entry to 0.
  int node_index = all_heights.size();
  all_heights.push_back(0);
  if (node.is_leaf()) {
    return 0;
  }
  int max_child_height = std::max(
      GetAllNodes(node.left(), depth + 1, all_nodes, all_depths, all_heights),
      GetAllNodes(node.right(), depth + 1, all_nodes, all_depths, all_heights));

  all_heights[node_index] = max_child_height + 1;
  return all_heights[node_index];
}

// Write each BVH node as an independent hexahedral element. Vertices of
// hexahedral elements are not shared.
template <typename MeshType>
void WriteVtkUnstructuredGrid(
    std::ofstream& out, const std::vector<const BvNode<MeshType>*>& bvh_nodes) {
  std::vector<Vector3d> vertices;
  for (const BvNode<MeshType>* node : bvh_nodes) {
    const Aabb& box = node->aabb();
    Vector3d upper = box.upper();
    Vector3d lower = box.lower();
    double x[2] = {lower.x(), upper.x()};
    double y[2] = {lower.y(), upper.y()};
    double z[2] = {lower.z(), upper.z()};
    vertices.emplace_back(x[0], y[0], z[0]);
    vertices.emplace_back(x[1], y[0], z[0]);
    vertices.emplace_back(x[1], y[1], z[0]);
    vertices.emplace_back(x[0], y[1], z[0]);
    vertices.emplace_back(x[0], y[0], z[1]);
    vertices.emplace_back(x[1], y[0], z[1]);
    vertices.emplace_back(x[1], y[1], z[1]);
    vertices.emplace_back(x[0], y[1], z[1]);
  }

  out << "DATASET UNSTRUCTURED_GRID\n";
  out << "POINTS " << vertices.size() << " double\n";
  for (const auto& vertex : vertices) {
    out << fmt::format("{:12.8f} {:12.8f} {:12.8f}\n", vertex[0], vertex[1],
                       vertex[2]);
  }
  out << std::endl;

  const int num_elements = bvh_nodes.size();
  constexpr int num_vertices_per_element = 8;
  const int num_integers = num_elements * (num_vertices_per_element + 1);
  out << "CELLS " << num_elements << " " << num_integers << std::endl;
  for (int i = 0; i < num_elements; ++i) {
    out << fmt::format("{}", num_vertices_per_element);
    out << fmt::format(" {:6d}", 8 * i);
    out << fmt::format(" {:6d}", 8 * i + 1);
    out << fmt::format(" {:6d}", 8 * i + 2);
    out << fmt::format(" {:6d}", 8 * i + 3);
    out << fmt::format(" {:6d}", 8 * i + 4);
    out << fmt::format(" {:6d}", 8 * i + 5);
    out << fmt::format(" {:6d}", 8 * i + 6);
    out << fmt::format(" {:6d}", 8 * i + 7);
    out << std::endl;
  }
  out << std::endl;

  constexpr int kVtkCellTypeHexahedron = 12;
  out << "CELL_TYPES " << num_elements << std::endl;
  for (int i = 0; i < num_elements; ++i) {
    out << fmt::format("{}\n", kVtkCellTypeHexahedron);
  }
  out << std::endl;
}

void WriteVtkCellDataDepthHeight(std::ofstream& out,
                      const std::vector<int>& depths,
                      const std::vector<int>& heights) {
  out << fmt::format("CELL_DATA {}\n", depths.size());

  out << fmt::format("SCALARS BVH_node_depth int 1\n");
  out << "LOOKUP_TABLE default\n";
  for (const auto depth : depths) {
    out << fmt::format("{:6d}\n", depth);
  }
  out << std::endl;

  out << fmt::format("SCALARS BVH_node_height int 1\n");
  out << "LOOKUP_TABLE default\n";
  for (const auto height : heights) {
    out << fmt::format("{:6d}\n", height);
  }
  out << std::endl;
}

template <typename MeshType>
void WriteToVtk(const std::string& file_name,
                const BoundingVolumeHierarchy<MeshType>& bvh,
                const std::string& title) {
  std::ofstream file(file_name);
  if (file.fail()) {
    throw std::runtime_error(fmt::format("Cannot create file: {}.", file_name));
  }
  std::vector<const BvNode<MeshType>*> all_nodes;
  // depth = number of edges to the root node. root.depth = 0
  std::vector<int> all_depths;
  // height = largest number of edges to a leaf. leaf.height = 0
  std::vector<int> all_heights;
  GetAllNodes(bvh.root_node(), 0, all_nodes, all_depths, all_heights);

  WriteVtkHeader(file, title);
  WriteVtkUnstructuredGrid(file, all_nodes);
  WriteVtkCellDataDepthHeight(file, all_depths, all_heights);
  file.close();
}

}  // namespace

void WriteBVHToVtk(const std::string& file_name,
                   const BoundingVolumeHierarchy<VolumeMesh<double>>& bvh,
                   const std::string& title) {
  WriteToVtk(file_name, bvh, title);
}

void WriteBVHToVtk(const std::string& file_name,
                   const BoundingVolumeHierarchy<SurfaceMesh<double>>& bvh,
                   const std::string& title) {
  WriteToVtk(file_name, bvh, title);
}

}  // namespace internal
}  // namespace geometry
}  // namespace drake
