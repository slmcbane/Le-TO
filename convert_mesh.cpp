#include "read_mesh.hpp"
#include "options.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

int main(int argc, char *argv[])
{
    const toml::table options = parse_options_file("options.toml");

    if (options.contains("mesh_file"))
    {
        if (!options["mesh_file"].is_string())
        {
            fprintf(stderr, "Expected mesh_file as string in options\n");
            exit(1);
        }
    }
    else {
        fprintf(stderr, "Expected mesh_file as string in options\n");
        exit(1);
    }
    const std::string mesh_file = options["mesh_file"].value<std::string>().value();

    if (options.contains("converted_mesh_file"))
    {
        if (!options["converted_mesh_file"].is_string())
        {
            fprintf(stderr, "Expected converted_mesh_file as string in options\n");
            exit(2);
        }
    }
    else {
        fprintf(stderr, "Expected converted_mesh_file as string in options\n");
        exit(2);
    }
    const std::string converted_mesh_file = options["converted_mesh_file"].value<std::string>().value();

    if (options.contains("mesh_order"))
    {
        if (!options["mesh_order"].is_integer())
        {
            fprintf(stderr, "Expected integer mesh_order in options\n");
            exit(2);
        }
        int order = options["mesh_order"].value<int>().value();
        if (!(order >= 1 && order <= 4))
        {
            fprintf(stderr, "Expected mesh order to be 1-4\n");
            exit(2);
        }
    }
    const int order = options["mesh_order"].value<int>().value();

    auto mesh = read_mesh(mesh_file.c_str(), order);

    FILE *output = fopen(converted_mesh_file.c_str(), "w");
    std::visit(
        [=](const auto &m)
        {
            m.serialize(output);
        }, mesh
    );
    fclose(output);

    return 0;
}