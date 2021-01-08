#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP

#include "C0_triangles.hpp"
#include "Galerkin/Galerkin.hpp"
#include "mesh_traits.hpp"
#include "SymSparse.hpp"

using Galerkin::get;

#include <Eigen/Core>

#include <exception>

namespace Elasticity
{

namespace TwoD
{

template <class Element, class U, class V>
Eigen::Matrix2d pair_stiffness_integral(const Element &el, const U &u, const V &v,
    double lambda, double mu)
{
    auto uxvx = el.integrate(el.template partial<0>(u) * el.template partial<0>(v));
    auto uxvy = el.integrate(el.template partial<0>(u) * el.template partial<1>(v));
    auto uyvy = el.integrate(el.template partial<1>(u) * el.template partial<1>(v));
    auto uyvx = el.integrate(el.template partial<1>(u) * el.template partial<0>(v));

    Eigen::Matrix2d K;
    K(0, 0) = (lambda + 2*mu) * uxvx + mu * uyvy;
    K(1, 0) = lambda * uyvx + mu * uxvy;
    K(0, 1) = lambda * uxvy + mu * uyvx;
    K(1, 1) = (lambda + 2*mu) * uyvy + mu * uxvx;

    return K;
}

template <class Element>
constexpr size_t basis_size = Element::basis.size();

template <auto I, auto J, class Element>
void add_contribution(const Element &el, double lambda, double mu,
                      Eigen::Matrix<double, 2*basis_size<Element>, 2*basis_size<Element>> &K)
{
    constexpr auto u = get<I>(Element::basis);
    constexpr auto v = get<J>(Element::basis);
    auto local_K = pair_stiffness_integral(el, u, v, lambda, mu);
    auto row = I*2;
    auto col = J*2;
    K.block(row, col, 2, 2) = local_K;
    K.block(col, row, 2, 2) = local_K.transpose();
}

template <class Element>
Eigen::Matrix<double, 2*basis_size<Element>, 2*basis_size<Element>>
element_stiffness_matrix(const Element &el, double lambda, double mu)
{
    Eigen::Matrix<double, 2*basis_size<Element>, 2*basis_size<Element>> K;
    Galerkin::static_for<0, basis_size<Element>, 1>(
        [&](auto I)
        {
            Galerkin::static_for<I(), basis_size<Element>, 1>(
                [&](auto J)
                {
                    add_contribution<I(), J()>(el, lambda, mu, K);
                }
            );
        }
    );
    return K;
}

template <class Mesh>
auto instantiate_element(const Mesh &mesh, size_t which)
{
    constexpr int order = msh::element_order<Mesh>;
    const auto &el = mesh.element(which);
    return fem::c0::C0Triangle<order>(
        mesh.coord(el.control_nodes[0]),
        mesh.coord(el.control_nodes[1]),
        mesh.coord(el.control_nodes[2])
    );
}

constexpr auto lame_parameters(double E, double nu) noexcept
{
    double lambda = E * nu / ((1+nu) * (1-nu));
    double mu     = E / (2 * (1+nu));
    return std::make_pair(lambda, mu);
}

template <class Mesh>
auto assemble_stiffness(const Mesh &mesh, double E, double nu)
{
    SymSparse::SymmetricSparseMatrix<double, 2 * msh::max_node_adjacencies<Mesh>> K(mesh.num_nodes() * 2);
    auto [lambda, mu] = lame_parameters(E, nu);

    for (size_t i = 0; i < mesh.num_elements(); ++i)
    {
        const auto &el_info = mesh.element(i);
        const auto nn = el_info.node_numbers();
        const auto el = instantiate_element(mesh, i);
        const auto local_K = element_stiffness_matrix(el, lambda, mu);
        for (size_t j = 0; j < nn.size(); ++j)
        {
            for (size_t k = j; k < nn.size(); ++k)
            {
                K.insert_entry(nn[j]*2, nn[k]*2, local_K(2*j, 2*k));
                if (k != j)
                {
                    K.insert_entry(nn[j]*2+1, nn[k]*2, local_K(2*j+1, 2*k));
                }
                K.insert_entry(nn[j]*2, nn[k]*2+1, local_K(2*j, 2*k+1));
                K.insert_entry(nn[j]*2+1, nn[k]*2+1, local_K(2*j+1, 2*k+1));
            }
        }
    }
    return K;
}

template <class Mesh>
using StiffnessType = decltype(assemble_stiffness(std::declval<const Mesh &>(), 1.0, 1.0));

/*
 * Add a homogeneous Dirichlet condition on the boundary segment of the mesh
 * indexed by `which`. K is the stiffness matrix obtained from `assemble_stiffness`,
 * and `rhs` is the forcing vector.
 */
template <class Mesh, class RHS>
void impose_homogeneous_condition(const Mesh &mesh, StiffnessType<Mesh> &K, RHS &rhs, size_t which,
                                  double scale = 1.0)
{
    std::vector<size_t> adjacent;
    adjacent.reserve(2 * msh::max_node_adjacencies<Mesh>);
    const auto &boundary = mesh.boundary(which);

    for (auto n: boundary.nodes)
    {
        adjacent.clear();
        // Get all of the adjacent DOFs to this one; since there are two components
        // of displacement there are 2 degrees of freedom (2*n, 2*n+1) corresponding
        // to each node.
        for (auto n2: mesh.adjacent_nodes(n))
        {
            adjacent.push_back(2*n2);
            adjacent.push_back(2*n2+1);
        }
        K.eliminate_dof(2*n, 0.0, scale, rhs, adjacent);
        K.eliminate_dof(2*n+1, 0.0, scale, rhs, adjacent);
    }
}

template <class Mesh>
void impose_homogeneous_condition(
    const Mesh &mesh, StiffnessType<Mesh> &K, size_t which, double scale = 1.0)
{
    std::vector<size_t> adjacent;
    adjacent.reserve(2 * msh::max_node_adjacencies<Mesh>);
    const auto &boundary = mesh.boundary(which);

    for (auto n : boundary.nodes)
    {
        adjacent.clear();
        // Get all of the adjacent DOFs to this one; since there are two components
        // of displacement there are 2 degrees of freedom (2*n, 2*n+1) corresponding
        // to each node.
        for (auto n2 : mesh.adjacent_nodes(n))
        {
            adjacent.push_back(2 * n2);
            adjacent.push_back(2 * n2 + 1);
        }
        K.eliminate_dof(2 * n, 0.0, scale, adjacent);
        K.eliminate_dof(2 * n + 1, 0.0, scale, adjacent);
    }
}

struct OutOfBoundsIndex : public std::exception
{
    const char *msg;
    OutOfBoundsIndex(const char *m) : msg(m) {}
    const char *what() const noexcept
    {
        return msg;
    }
};

template <class Mesh, class RHS, class Vector>
void impose_dirichlet_condition(const Mesh &mesh, StiffnessType<Mesh> &K, RHS &rhs, size_t which,
                                const Vector &value, double scale = 1.0)
{
    std::vector<size_t> adjacent;
    adjacent.reserve(2 * msh::max_node_adjacencies<Mesh>);
    const auto &boundary = mesh.boundary(which);

    if (value.size() != 2*boundary.nodes.size())
    {
        throw OutOfBoundsIndex("Vector given for Dirichlet condition has wrong number of values");
    }

    size_t i = 0;
    for (auto n: boundary.nodes)
    {
        adjacent.clear();
        for (auto n2: mesh.adjacent_nodes(n))
        {
            adjacent.push_back(2*n2);
            adjacent.push_back(2*n2+1);
        }
        K.eliminate_dof(2*n, value[2*i], scale, rhs, adjacent);
        K.eliminate_dof(2*n+1, value[2*i+1], scale, rhs, adjacent);
        i += 1;
    }
}

template <class Force, class RHS>
void add_point_force(const Force &force, size_t node, RHS &F)
{
    if (2*node+1 > F.size())
    {
        throw OutOfBoundsIndex("Node is out of bounds for given forcing vector");
    }
    F[2*node] += force[0];
    F[2*node+1] += force[1];
}

template <class T, class RHS, int N, class IndexContainer, int... Options>
void add_point_forces(const Eigen::Matrix<T, 2, N, Options...> &force, const IndexContainer &nodes,
                      RHS &F)
{
    static_assert(sizeof...(Options) == 3);
    size_t col = 0;
    for (size_t n: nodes)
    {
        add_point_force(force.col(col), n, F);
        col += 1;
    }
}

} // namespace 2D

} // namespace Elasticity

#endif // LINEAR_ELASTICITY_HPP
