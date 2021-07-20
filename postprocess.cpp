#include "options.hpp"
#include "evaluator.hpp"
#include "save_eigen.hpp"

double get_drop_tolerance(const toml::table &options)
{
    if (!options.contains("drop_tolerance"))
    {
        fmt::print(stderr, "drop_tolerance not found in options\n");
        exit(1);
    }
    else if(!options["drop_tolerance"].is_floating_point())
    {
        fmt::print(stderr, "Expected drop_tolerance as floating point\n");
        exit(2);
    }

    return options["drop_tolerance"].as_floating_point()->get();
}

int main()
{
    auto options = parse_options_file("options.toml");

    /* Model setup */
    std::string mesh_file = options.get("mesh_file")->as_string()->get();
    int order = options.get("mesh_order")->as_integer()->get();
    double force_magnitude = options.get("force_magnitude")->as_floating_point()->get();
    double filter_radius = options.get("filter_radius")->as_floating_point()->get();
    MaterialOptions mat_options;
    parse_material_options(options, mat_options);
    double E = mat_options.youngs_modulus.value();
    double nu = mat_options.poissons_ratio.value();
    double p_material = mat_options.simp_exponent.value();
    double eps_material = mat_options.epsilon_constitutive.value();
    auto [lambda, mu] = Elasticity::TwoD::lame_parameters(E, nu);
    Eigen::Vector2d force(0.0, -force_magnitude);
    Evaluator evaluator(
        mesh_file, order, force, ErsatzStiffness(p_material, eps_material), filter_radius, lambda, mu);

    /* Set up the parameters for stress constraints */
    auto rho = read_eigen("rho_filt.dat");
    const double drop_tol = get_drop_tolerance(options);
    for (long i = 0; i < rho.size(); ++i)
    {
        rho[i] = rho[i] > drop_tol ? 1 : 0;
    }

    evaluator.set_filtered_parameter_directly(rho.data());

    save_eigen(rho, "rho_pp.dat");
    save_eigen(evaluator.displacement(), "u.dat");
    save_eigen(evaluator.cell_centered_stress(), "cc_stress.dat");

    std::cout << "Max stress in the domain: " << evaluator.max_stress() << '\n';
    std::cout << "Objective value: " << evaluator.sum_mass_times_density() << "\n";

    return 0;
}
