#ifndef MESH_TRAITS_HPP
#define MESH_TRAITS_HPP

#include "TetMesh.hpp"
#include <type_traits>

namespace msh
{

template <class T>
struct IsTetMesh : public std::false_type
{
};

template <class CoordT, size_t... Others>
struct IsTetMesh<TetMesh<CoordT, Others...>> : public std::true_type
{
};

template <size_t NPF, size_t IN>
constexpr int which_poly_order() noexcept
{
    if constexpr (NPF == 0 && IN == 0)
    {
        return 1;
    }
    else if constexpr (NPF == 1 && IN == 0)
    {
        return 2;
    }
    else if constexpr (NPF == 2 && IN == 1)
    {
        return 3;
    }
    else if constexpr (NPF == 3 && IN == 3)
    {
        return 4;
    }
    else
    {
        return -1;
    }
}

template <class Mesh>
struct ElementOrder : public std::integral_constant<int, -1>
{};

template <class CoordT, size_t MEA, size_t MNA, size_t NPF, size_t IN>
struct ElementOrder<TetMesh<CoordT, MEA, MNA, NPF, IN>> :
    public std::integral_constant<int, which_poly_order<NPF, IN>()>
{};

template <class Mesh>
constexpr int element_order = ElementOrder<Mesh>::value;

template <class Mesh>
struct MaxNodeAdjacencies
{};

template <class CoordT, size_t MEA, size_t MNA, size_t NPF, size_t IN>
struct MaxNodeAdjacencies<TetMesh<CoordT, MEA, MNA, NPF, IN>> :
    public std::integral_constant<size_t, MNA>
{};

template <class Mesh>
constexpr size_t max_node_adjacencies = MaxNodeAdjacencies<Mesh>::value;

} // namespace msh

#endif // MESH_TRAITS_HPP

