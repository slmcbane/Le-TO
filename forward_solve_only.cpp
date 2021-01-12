#include "ersatz_stiffness.hpp"
#include "forward_model.hpp"
#include "options.hpp"
#include "save_eigen.hpp"

const ErsatzStiffness interp(3.0, 1e-3);

int main()
{
    auto options = parse_options_file("options.toml");
    std::string mesh_file = options.get("mesh_file")->as_string()->get();
    int order = options.get("mesh_order")->as_integer()->get();
    double force_magnitude = options.get("force_magnitude")->as_floating_point()->get();
    double filter_radius = options.get("filter_radius")->as_floating_point()->get();

    MaterialOptions mat_options;
    parse_material_options(options, mat_options);
    double E = *mat_options.youngs_modulus;
    double nu = *mat_options.poissons_ratio;
    auto [lambda, mu] = Elasticity::TwoD::lame_parameters(E, nu);

    Eigen::Vector2d force(0.0, -force_magnitude);

    std::unique_ptr<ModelInfoVariant> minfo =
        construct_model_info(mesh_file, order, force, interp, filter_radius, lambda, mu);

    Eigen::VectorXd rho(num_elements(*minfo));
    rho.fill(1);
    const auto &u = solve_forward(*minfo, rho);

    save_eigen(u, "u.dat");

    return 0;
}