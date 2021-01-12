#include "evaluator.hpp"
#include "xorshift.hpp"
#include "options.hpp"
#include "save_eigen.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

constexpr double perturbation_magnitude = 1e-6;
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

    Evaluator evaluator(mesh_file, order, force, interp, filter_radius, lambda, mu);

    Eigen::VectorXd rho(num_elements(evaluator.model_info()));
    Eigen::VectorXd perturbation(rho.size());
    Xorshift64StarEngine random_engine;

    std::uniform_real_distribution<double> udist(0.0, 
        std::nextafter(1.0, std::numeric_limits<double>::max()));
    std::normal_distribution<double> ndist;

    for (int i = 0; i < rho.size(); ++i)
    {
        rho[i] = udist(random_engine);
        perturbation[i] = ndist(random_engine);
    }

    evaluator.set_parameter(rho.data());
    const double rc = evaluator.compliance();
    const Eigen::VectorXd rg = evaluator.compliance_gradient();

    rho += perturbation_magnitude * perturbation;
    evaluator.set_parameter(rho.data());
    const double nc = evaluator.compliance();

    const double dd = rg.dot(perturbation);
    const double dd_fd = (nc - rc) / perturbation_magnitude;

    fmt::print("Directional derivative analytically: {}\n", dd);
    fmt::print("Directional derivative using FD: {}\n", dd_fd);

    save_eigen(rho, "rho.dat");
    save_eigen(std::visit([](const auto &minfo) { return minfo.rho_filt; }, evaluator.model_info()), "rho_filt.dat");

    return 0;
}
