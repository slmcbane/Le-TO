#include "IpIpoptApplication.hpp"
#include "OptimizationProblem.hpp"
#include "options.hpp"
#include "save_eigen.hpp"

#include <chrono>

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

void apply_options(Ipopt::IpoptApplication &app, const OptimizationOptions &opt_options)
{
    app.Options()->SetStringValue("hessian_approximation", "limited-memory");
    app.Options()->SetStringValue("mu_strategy", "adaptive");

    if (opt_options.max_iters)
    {
        app.Options()->SetIntegerValue("max_iter", *opt_options.max_iters);
    }

    if (opt_options.absolute_tol)
    {
        app.Options()->SetNumericValue("tol", *opt_options.absolute_tol);
    }

    if (opt_options.acceptable_tol)
    {
        app.Options()->SetNumericValue("acceptable_tol", *opt_options.acceptable_tol);
    }

    if (opt_options.acceptable_iters)
    {
        app.Options()->SetIntegerValue("acceptable_iter", *opt_options.acceptable_iters);
    }

    if (opt_options.verbosity_level)
    {
        app.Options()->SetIntegerValue("print_level", *opt_options.verbosity_level);
    }

    if (opt_options.theta_max_fact)
    {
        app.Options()->SetNumericValue("theta_max_fact", *opt_options.theta_max_fact);
    }

    if (opt_options.watchdog_shortened_iter_trigger)
    {
        app.Options()->SetIntegerValue(
            "watchdog_shortened_iter_trigger", *opt_options.watchdog_shortened_iter_trigger);
    }
}

namespace
{

std::vector<double> get_initial_condition(std::size_t n, const OptimizationOptions &options)
{
    if (options.initial_condition_file)
    {
        Eigen::VectorXd init = read_eigen((*options.initial_condition_file).c_str());
        assert(init.size() == n);
        return std::vector<double>(init.data(), init.data() + init.size());
    }
    else
    {
        return std::vector<double>(n, 1);
    }
}

} // namespace

int main()
{
    auto start = std::chrono::steady_clock::now();

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

    OptimizationOptions opt_options;
    parse_optimization_options(options, opt_options);

    /* Set up the parameters for stress constraints */
    AggregationOptions agg_options;
    parse_aggregation_options(options, agg_options);
    double p_stress = mat_options.stress_exponent.value();
    double eps_stress = mat_options.epsilon_stress.value();
    evaluator.set_parameter(get_initial_condition(num_elements(evaluator.model_info()), opt_options).data());
    double ks_alpha = evaluator.estimated_ks_alpha(agg_options.aggregation_multiplier.value(), 1.0);
    evaluator.set_stress_criterion(StressCriterionDefinition{
        ErsatzStiffness(p_stress, eps_stress),
        assign_agg_regions(evaluator.cell_centered_stress(), agg_options.num_aggregation_regions.value()),
        agg_options.aggregation_multiplier.value(), 
        Eigen::VectorXd::Constant(agg_options.num_aggregation_regions.value(), ks_alpha)});

    Ipopt::SmartPtr<OptimizationProblem> problem =
        new OptimizationProblem(evaluator, opt_options, parse_optimization_type(options));

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = new Ipopt::IpoptApplication();

    apply_options(*app, opt_options);

    app->OptimizeTNLP(problem);

    save_eigen(problem->optimal_values(), "rho.dat");
    save_eigen(evaluator.filtered_parameter(), "rho_filt.dat");
    save_eigen(evaluator.cell_centered_stress(), "cc_stress.dat");
    save_eigen(evaluator.displacement(), "u.dat");

    std::cout << "Max stress in the domain: " << evaluator.max_stress() << '\n';
    std::cout << "Max stresses by region:\n" << evaluator.max_stresses() << "\n";
    std::cout << "vs. constraint values:\n" << problem->constraint_values() << "\n";
    std::cout << "Objective value: " << problem->objective() << "\n";

    auto stop = std::chrono::steady_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << "s\n";

    return 0;
}
