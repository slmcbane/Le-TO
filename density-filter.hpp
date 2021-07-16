#ifndef DENSITY_FILTER_HPP
#define DENSITY_FILTER_HPP

#include "Galerkin/Galerkin.hpp"
#include "SmallVector.hpp"
#include "read_mesh.hpp"

#include <cassert>
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

template <class Vin, class Vout>
void filter_densities(Vout &filtered, const Vin &rho, const std::vector<DensityFilter> &filter)
{
    assert(static_cast<unsigned long>(filtered.size()) == static_cast<unsigned long>(rho.size()));
    assert(static_cast<unsigned long>(filtered.size()) == static_cast<unsigned long>(filter.size()));

    for (std::size_t i = 0; i < filtered.size(); ++i)
    {
        filtered[i] = 0;
        for (const auto [j, w] : filter[i].entries)
        {
            assert(w > 0);
            filtered[i] += w * rho[j];
        }
        filtered[i] = filtered[i] < 0 ? 0 : filtered[i];
    }
}

#endif // DENSITY_FILTER_HPP
