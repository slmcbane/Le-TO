#ifndef READ_MESH_HPP
#define READ_MESH_HPP

#include <cassert>
#include <variant>

#include "TetMesh.hpp"

template <int order>
constexpr size_t max_node_adjacencies()
{
    if constexpr (order == 1)
    {
        return 8;
    }
    else if constexpr (order == 2)
    {
        return 24;
    }
    else if constexpr (order == 3)
    {
        return 48;
    }
    else if constexpr (order == 4)
    {
        return 72;
    }
}

constexpr size_t max_element_adjacencies = 16;

using MeshVariant = std::variant<
    msh::TetMesh<double, max_element_adjacencies, max_node_adjacencies<1>(), 0, 0>,
    msh::TetMesh<double, max_element_adjacencies, max_node_adjacencies<2>(), 1, 0>,
    msh::TetMesh<double, max_element_adjacencies, max_node_adjacencies<3>(), 2, 1>,
    msh::TetMesh<double, max_element_adjacencies, max_node_adjacencies<4>(), 3, 3>>;

template <int order>
using MeshType = std::variant_alternative_t<order-1, MeshVariant>;

MeshVariant read_mesh(const char *name, int order);

#endif // READ_MESH_HPP
