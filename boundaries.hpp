#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <vector>

#include "Galerkin/Galerkin.hpp"
#include "TetMesh.hpp"
#include "mesh_traits.hpp"

template <int Order>
struct LineMesh
{
    LineMesh(const std::vector<double> &p) : pts(p) {}

    auto get_elements_and_dofs() const
    {
        std::vector<Galerkin::Elements::IntervalElement<Order, double>> elements;
        std::vector<std::array<size_t, Order + 1>> dofs;
        std::array<size_t, Order + 1> tmp;
        size_t d = 0;
        for (size_t i = 0; i < pts.size() - 1; ++i)
        {
            elements.push_back(Galerkin::Elements::IntervalElement<Order, double>(pts[i], pts[i + 1]));
            for (size_t j = 0; j <= Order; ++j)
            {
                tmp[j] = d + j;
            }
            dofs.push_back(tmp);
            d += Order;
        }
        return std::pair(elements, dofs);
    }

  private:
    std::vector<double> pts;
};

template <class Mesh>
LineMesh<msh::element_order<Mesh>> build_boundary_mesh(const Mesh &mesh, std::size_t which)
{
    std::vector<double> pts{0.0};
    const auto &boundary = mesh.boundary(which);
    for (const auto &face : boundary.faces)
    {
        size_t first_node = boundary.nodes[face.nodes[0]];
        size_t second_node = boundary.nodes[*(face.nodes.end() - 1)];
        auto c1 = mesh.coord(first_node);
        auto c2 = mesh.coord(second_node);
        double dx = c2[0] - c1[0];
        double dy = c2[1] - c1[1];
        pts.push_back(pts.back() + std::sqrt(dx * dx + dy * dy));
    }
    return LineMesh<msh::element_order<Mesh>>(pts);
}

#endif // BOUNDARIES_HPP