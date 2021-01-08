#include "read_mesh.hpp"

MeshVariant read_mesh(const char *name, int order)
{
    assert(order == 1 || order == 2 || order == 3 || order == 4);
    if (order == 1)
    {
        auto mesh = msh::parse_gmsh_to_tetmesh<max_element_adjacencies,
                                          max_node_adjacencies<1>(), 0, 0>(name);
        mesh.renumber_nodes();
        return mesh;
    }
    else if (order == 2)
    {
        auto mesh = msh::parse_gmsh_to_tetmesh<max_element_adjacencies,
                                          max_node_adjacencies<2>(), 1, 0>(name);
        mesh.renumber_nodes();
        return mesh;
    }
    else if (order == 3)
    {
        auto mesh = msh::parse_gmsh_to_tetmesh<max_element_adjacencies,
                                          max_node_adjacencies<3>(), 2, 1>(name);
        mesh.renumber_nodes();
        return mesh;
    }
    else
    {
        auto mesh = msh::parse_gmsh_to_tetmesh<max_element_adjacencies,
                                          max_node_adjacencies<4>(), 3, 3>(name);
        mesh.renumber_nodes();
        return mesh;
    }
}