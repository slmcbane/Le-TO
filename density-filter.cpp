#include "density-filter.hpp"
#include "SmallVector.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace
{

template <class Mesh>
std::vector<DensityFilter> build_filters_impl(const Mesh &mesh, double r0);

} // namespace

std::vector<DensityFilter> build_filters(const MeshVariant &mesh, double r0)
{
    try
    {
        return std::visit([r0](const auto &mesh) { return build_filters_impl(mesh, r0); }, mesh);
    }
    catch (const smv::MaxSizeExceeded &)
    {
        fmt::print(stderr, "Exceeded max number of filter entries, {}\n", max_filter_elements);
        exit(2);
    }
}

namespace
{

template <class Mesh>
void process_element(
    const Mesh &, double, std::size_t, std::vector<std::unordered_map<std::size_t, double>> &);

DensityFilter construct_filter(const std::unordered_map<std::size_t, double> &);

template <class Mesh>
std::vector<DensityFilter> build_filters_impl(const Mesh &mesh, double r0)
{
    // element_maps holds a map, for each element eli, of distances of element
    // centroids from the centroid of eli. element_maps[eli][elj] is the distance
    // between the centroids. It will be zero for some eli, elj pairs, and not
    // exist for most of them.
    std::vector<std::unordered_map<std::size_t, double>> element_maps(mesh.num_elements());

    for (std::size_t eli = 0; eli < mesh.num_elements(); ++eli)
    {
        // Updates the element maps with new entries from processing this element.
        process_element(mesh, r0, eli, element_maps);
    }

    std::vector<DensityFilter> filters;
    filters.reserve(mesh.num_elements());
    std::transform(
        element_maps.begin(), element_maps.end(), std::back_inserter(filters),
        [&](const auto &map) { return construct_filter(map); });

    return filters;
}

template <class Mesh>
auto compute_centroid(const Mesh &mesh, std::size_t eli)
{
    const auto &el = mesh.element(eli);
    const auto transform = Galerkin::Transforms::triangle_transform(
        mesh.coord(el.control_nodes[0]), mesh.coord(el.control_nodes[1]), mesh.coord(el.control_nodes[2]));
    
    const auto area = transform.template quadrature<1>(
        []([[maybe_unused]] auto pt) { return 1; }
    );
    const auto integral_of_x = transform.template quadrature<1>(
        [&](auto pt) { return std::get<0>(transform(pt)); }
    );
    const auto integral_of_y = transform.template quadrature<1>(
        [&](auto pt) { return std::get<1>(transform(pt)); }
    );

    return std::array<double, 2>{integral_of_x / area, integral_of_y / area};
}

double euclidean_distance(std::array<double, 2> x, std::array<double, 2> y) noexcept
{
    const double dx = x[0] - y[0];
    const double dy = x[1] - y[1];
    return std::sqrt(dx * dx + dy * dy);
}

template <class Mesh>
void process_element_pair(
    const Mesh &mesh, double r0, std::size_t eli, std::size_t elj,
    std::vector<std::unordered_map<std::size_t, double>> &element_maps)
{
    // If elj is already in the element map for eli, nothing to do.
    // Check to make sure eli is included in element_maps[elj].
    if (element_maps[eli].count(elj) == 1)
    {
        assert(element_maps[elj].count(eli) == 1);
        return;
    }

    // Check the distance from eli to elj.
    const auto icentroid = compute_centroid(mesh, eli);
    const auto jcentroid = compute_centroid(mesh, elj);
    const auto r = euclidean_distance(icentroid, jcentroid);

    if (r >= r0)
    {
        if constexpr (debug_output)
        {
            fmt::print(
                FMT_STRING("Element {:d} has distance greater than r0 from element {:d}\n"), elj, eli);
        }
        element_maps[eli][elj] = 0;
        element_maps[elj][eli] = 0;
        return;
    }

    const auto weight = (r0 - r) / r0;
    element_maps[eli][elj] = weight;
    element_maps[elj][eli] = weight;

    if constexpr (debug_output)
    {
        fmt::print(
            FMT_STRING("Added {:d} to filter for {:d} (and vice versa) with weight {:E}\n"), elj, eli,
            weight);
    }

    for (std::size_t elk : mesh.element(elj).adjacent_elements)
    {
        process_element_pair(mesh, r0, eli, elk, element_maps);
    }
}

template <class Mesh>
void process_element(
    const Mesh &mesh, double r0, std::size_t eli,
    std::vector<std::unordered_map<std::size_t, double>> &element_maps)
{
    // Add my own weight.
    element_maps[eli][eli] = 1.0;

    for (std::size_t elj : mesh.element(eli).adjacent_elements)
    {
        process_element_pair(mesh, r0, eli, elj, element_maps);
    }
}

DensityFilter construct_filter(const std::unordered_map<std::size_t, double> &map)
{
    DensityFilter filter;
    double sum_weights = 0;
    for (const auto &[i, w] : map)
    {
        if (w == 0)
        {
            continue;
        }
        sum_weights += w;
        filter.entries.emplace_back(i, w);
    }
    for (auto &entry : filter.entries)
    {
        entry.weight /= sum_weights;
    }

    return filter;
}

} // namespace