#include "OptimizationProblem.hpp"
#include "options.hpp"
#include "save_eigen.hpp"

OptimizationType parse_optimization_type(const toml::table &options)
{
    if (!options.contains("optimization_type"))
    {
        fmt::print(stderr, "Missing optimization_type in options\n");
        exit(3);
    }
    else if (!options["optimization_type"].is_string())
    {
        fmt::print(stderr, "optimization_type should be a string\n");
        exit(2);
    }

    std::string ty = options["optimization_type"].as_string()->get();
    if (!(ty.size() == 3 || ty.size() == 4))
    {
        goto unrecognized;
    }

    switch (ty[1])
    {
    case 'W':
        if (ty[2] == 'S')
        {
            if (!(ty == "MWS"))
            {
                goto unrecognized;
            }
            return MWS;
        }
        else if (!(ty == "MWCS"))
        {
            goto unrecognized;
        }
        return MWCS;
    case 'C':
        if (ty == "MCW")
        {
            return MCW;
        }
        else if (ty == "MCWS")
        {
            return MCWS;
        }
        goto unrecognized;
    case 'S':
        if (ty == "MSW")
        {
            return MSW;
        }
    default:
        goto unrecognized;
    }
unrecognized:
    fmt::print(stderr, FMT_STRING("Unrecognized optimization type \"{:s}\"\n"), ty);
    exit(3);
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
    AggregationOptions agg_options;
    parse_aggregation_options(options, agg_options);
    double p_stress = mat_options.stress_exponent.value();
    double eps_stress = mat_options.epsilon_stress.value();
    evaluator.set_parameter(std::vector<double>(num_elements(evaluator.model_info()), 1).data());
    evaluator.set_stress_criterion(StressCriterionDefinition{
        ErsatzStiffness(p_stress, eps_stress),
        assign_agg_regions(evaluator.cell_centered_stress(), agg_options.num_aggregation_regions.value()),
        agg_options.aggregation_multiplier.value()});

    OptimizationOptions opt_options;
    parse_optimization_options(options, opt_options);

    OptimizationProblem problem(evaluator, opt_options, parse_optimization_type(options));
    nlopt::opt optimizer = problem.get_optimizer();
    std::vector<double> rho(num_elements(evaluator.model_info()), 1);
    double optimal_value;
    optimizer.optimize(rho, optimal_value);

    // save_eigen(problem->optimal_values(), "rho.dat");
    save_eigen(evaluator.filtered_parameter(), "rho_filt.dat");
    save_eigen(evaluator.cell_centered_stress(), "cc_stress.dat");
    save_eigen(evaluator.displacement(), "u.dat");

    std::cout << "Max stresses by region:\n" << evaluator.max_stresses() << "\n";

    return 0;
}
