#include "density-filter.hpp"
#include "read_mesh.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

void serialize_filter(std::FILE *out, const DensityFilter &filter)
{
    int64_t count = filter.entries.size();
    smv::SmallVector<int64_t, max_filter_elements> is;
    smv::SmallVector<double, max_filter_elements> ws;

    for (const auto &entry : filter.entries)
    {
        is.push_back(entry.index);
        ws.push_back(entry.weight);
    }

    fwrite(&count, sizeof(count), 1, out);
    fwrite(is.data(), sizeof(int64_t), is.size(), out);
    fwrite(ws.data(), sizeof(double), ws.size(), out);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fmt::print("Please pass the name of a gmsh mesh file as command line argument"
                   " and filter radius as second command line argument\n");
        exit(1);
    }

    const MeshVariant mesh = read_mesh(argv[1], 1);
    const double r0 = std::stod(argv[2]);

    fmt::print(FMT_STRING("Building filter with radius {:E}\n"), r0);

    const std::vector<DensityFilter> filters = build_filters(mesh, r0);

    std::FILE *out = fopen("filters.dat", "wb");

    std::size_t i = 0;
    for (const auto &filter : filters)
    {
        serialize_filter(out, filter);
        fmt::print(FMT_STRING("Filter for element {:d}:\n"), i++);
        double total_weight = 0;
        for (const auto &entry : filter.entries)
        {
            total_weight += entry.weight;
            fmt::print(FMT_STRING("({:E}, {:d})  "), entry.weight, entry.index);
        }
        fmt::print(FMT_STRING("\nSum of weights: {:f}\n\n"), total_weight);
    }

    fclose(out);

    return 0;
}
