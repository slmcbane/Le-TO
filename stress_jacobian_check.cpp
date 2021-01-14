#include "evaluator.hpp"
#include "options.hpp"
#include "stress.hpp"
#include "xorshift.hpp"
#include "save_eigen.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

constexpr double perturbation_magnitude = 1e-6;
const ErsatzStiffness stiffness_interp(3, 1e-3);
const ErsatzStiffness stress_interp(0.5, 1e-6);

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

    Evaluator evaluator(mesh_file, order, force, stiffness_interp, filter_radius, lambda, mu);

    AggregationOptions agg_options;
    parse_aggregation_options(options, agg_options);

    Eigen::VectorXd rho(num_elements(evaluator.model_info()));
    Eigen::VectorXd perturbation(rho.size());
    Xorshift64StarEngine random_engine;

    std::uniform_real_distribution<double> udist(
        0.9, std::nextafter(0.999, std::numeric_limits<double>::max()));
    std::normal_distribution<double> ndist;

    for (int i = 0; i < rho.size(); ++i)
    {
        rho[i] = udist(random_engine);
        perturbation[i] = ndist(random_engine);
    }

    evaluator.set_parameter(rho.data());

    // Using cell centered stresses from this evaluation, assign aggregation
    // regions.
    int nregions = *agg_options.num_aggregation_regions;
    auto agg_regions = assign_agg_regions(evaluator.cell_centered_stress(), nregions);
    evaluator.set_stress_criterion(StressCriterionDefinition{
        stress_interp, std::move(agg_regions), *agg_options.aggregation_multiplier});

    const auto ref_jac = evaluator.stress_agg_jacobian();
    const auto ref_aggs = evaluator.stress_aggregates();
    std::cout << ref_aggs << std::endl;

    rho += perturbation_magnitude * perturbation;
    evaluator.set_parameter(rho.data());
    const auto new_aggs = evaluator.stress_aggregates();

    const Eigen::VectorXd dd = ref_jac * perturbation;
    const Eigen::VectorXd dd_fd = (new_aggs - ref_aggs) / perturbation_magnitude;

    for (int i = 0; i < nregions; ++i)
    {
        fmt::print("Directional derivative of aggregate {}: {:E} vs {:E}\n", i+1, dd[i], dd_fd[i]);
    }

    save_eigen(evaluator.cell_centered_stress(), "sigma.dat");

    return 0;
}
