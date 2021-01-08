#ifndef DENSITY_FILTER_HPP
#define DENSITY_FILTER_HPP

#include "Galerkin/Galerkin.hpp"
#include "SmallVector.hpp"
#include "read_mesh.hpp"

#include <vector>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

inline constexpr int max_filter_elements = 64;

inline constexpr bool debug_output = false;

/*
 * This structure defines the density filter for a single element. max_filter_elements,
 * above, defines the maximum number of elements that can be included in the filter.
 */
struct DensityFilter
{
    struct Entry
    {
        std::size_t index;
        double weight;

        constexpr Entry(std::size_t i, double w) noexcept : index{i}, weight{w} {}
        Entry() = default;
    };
    smv::SmallVector<Entry, max_filter_elements> entries;
};

std::vector<DensityFilter> build_filters(const MeshVariant &mesh, double r0);

#endif // DENSITY_FILTER_HPP
