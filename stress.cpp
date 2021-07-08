#include "stress.hpp"
#include "2d.hpp"
#include "forward_model.hpp"
#include "tensor_product.hpp"

namespace
{

template <class Mesh>
auto vmcomputer(const Eigen::VectorXd &u, const Mesh &mesh, double lambda, double mu, std::size_t eli)
{
    const auto nn = mesh.element(eli).node_numbers();
    const auto el = Elasticity::TwoD::instantiate_element(mesh, eli);
    Eigen::Matrix<double, decltype(el)::basis.size() * 2, 1> coeffs;
    for (int i = 0; i < coeffs.size() / 2; ++i)
    {
        coeffs[2 * i] = u[2 * nn[i]];
        coeffs[2 * i + 1] = u[2 * nn[i] + 1];
    }
    return Elasticity::TwoD::VonMisesComputer<decltype(el)>(el, coeffs, lambda, mu);
}

template <class Mesh>
double
cell_centered_stress(const Eigen::VectorXd &u, const Mesh &mesh, double lambda, double mu, std::size_t eli)
{
    return vmcomputer(u, mesh, lambda, mu, eli).evaluate(-1.0 / 3, -1.0 / 3);
}

template <class Mesh>
auto cell_centered_stress_w_gradient(
    const Eigen::VectorXd &u, const Mesh &mesh, double lambda, double mu, std::size_t eli)
{
    return vmcomputer(u, mesh, lambda, mu, eli).evaluate_with_gradient(-1.0 / 3, -1.0 / 3);
}

template <class Mesh>
void cell_centered_stress(
    Eigen::VectorXd &dest, const Eigen::VectorXd &u, const Mesh &mesh, double lambda, double mu)
{
    assert(static_cast<unsigned>(dest.size()) == mesh.num_elements());
    assert(static_cast<unsigned>(u.size()) == 2 * mesh.num_nodes());
    for (size_t i = 0; i < mesh.num_elements(); ++i)
    {
        dest[i] = cell_centered_stress(u, mesh, lambda, mu, i);
    }
}

template <class Mesh>
void eliminate_essential_boundaries(Eigen::MatrixXd &workspace, const ModelInfo<Mesh> &minfo)
{
    assert(workspace.rows() == 2 * minfo.mesh.num_nodes());
    for (int j = 0; j < workspace.cols(); ++j)
    {
        for (std::size_t which : minfo.homogeneous_boundaries)
        {
            const auto &bound = minfo.mesh.boundary(which);
            for (auto n : bound.nodes)
            {
                workspace(2 * n, j) = 0;
                workspace(2 * n + 1, j) = 0;
            }
        }
    }
}

} // namespace

void cell_centered_stress(Eigen::VectorXd &dest, const ModelInfoVariant &minfo, double lambda, double mu)
{
    std::visit(
        [&, lambda, mu](const auto &minfo)
        { cell_centered_stress(dest, minfo.displacement, minfo.mesh, lambda, mu); },
        minfo);
}

void pnorm_stress_aggregates(
    Eigen::VectorXd &aggregates, Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def,
    const ModelInfoVariant &minfo, double lambda, double mu)
{
    aggregates = Eigen::VectorXd::Zero(def.agg_regions.n);
    auto counts = std::vector<int>(aggregates.size(), 0);
    std::visit(
        [&, lambda, mu](const auto &minfo)
        {
            cc_stress.resize(minfo.mesh.num_elements());
            const auto &u = minfo.displacement;
            for (std::size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
            {
                double sigma = cell_centered_stress(u, minfo.mesh, lambda, mu, eli) *
                               def.stiffness_interp(minfo.rho_filt[eli]);
                cc_stress[eli] = sigma;
                aggregates[def.agg_regions.assignments[eli]] += std::pow(sigma, def.p);
                counts[def.agg_regions.assignments[eli]] += 1;
            }
            for (int i = 0; i < aggregates.size(); ++i)
            {
                aggregates[i] = std::pow(aggregates[i] / counts[i], 1.0 / def.p);
            }
        },
        minfo);
}

AggregationRegions assign_agg_regions(const Eigen::VectorXd &cc_stress, std::size_t n)
{
    std::vector<std::size_t> indices;
    indices.reserve(cc_stress.size());
    for (long i = 0; i < cc_stress.size(); ++i)
    {
        indices.push_back(i);
    }

    std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return cc_stress[i] < cc_stress[j]; });
    AggregationRegions agg_regions;
    agg_regions.n = n;
    agg_regions.assignments.resize(indices.size());

    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        agg_regions.assignments[indices[i]] = i % n;
    }

    return agg_regions;
}

void pnorm_aggs_with_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def, const ModelInfoVariant &minfo,
    double lambda, double mu, Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2)
{
    aggs = Eigen::VectorXd::Zero(def.agg_regions.n);
    auto counts = std::vector<int>(aggs.size(), 0);
    std::visit(
        [&](const auto &minfo)
        {
            J = std::decay_t<decltype(J)>::Zero(def.agg_regions.n, minfo.mesh.num_elements());
            cc_stress.resize(minfo.mesh.num_elements());

            // The first term in the Jacobian will be accumulated in the rows of J.
            // Workspace is used to compute the adjoint right-hand sides, then the adjoint solve
            // is performed into J^T
            workspace = Eigen::MatrixXd::Zero(2 * minfo.mesh.num_nodes(), J.rows());
            const auto &u = minfo.displacement;
            const auto &s = def.stiffness_interp;
            for (std::size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
            {
                const long agg_index = def.agg_regions.assignments[eli];
                const auto &filt = minfo.filter[eli];
                // sigma is relaxed stress at eli; dsigmadu is its partial w.r.t. u.
                auto [sigma, dsigmadu] = cell_centered_stress_w_gradient(u, minfo.mesh, lambda, mu, eli);

                sigma *= s(minfo.rho_filt[eli]);
                cc_stress[eli] = sigma;
                dsigmadu *= s(minfo.rho_filt[eli]);

                // update p-norm aggregate accumulator.
                aggs[agg_index] += std::pow(sigma, def.p);
                counts[agg_index] += 1;

                // partial of p-norm aggregate w.r.t. this relaxed stress, modulo a constant.
                double dpndsigma = std::pow(sigma, def.p - 1);

                // partial of the relaxed stress w.r.t. *filtered* density.
                double dsigmadrho = s.derivative(minfo.rho_filt[eli]) * cc_stress[eli];

                // update the first Jacobian term with contributions for each unfiltered
                // density variable.
                for (auto [ri, w] : filt.entries)
                {
                    J(agg_index, ri) += dpndsigma * dsigmadrho * w;
                }

                // update the adjoint RHS for this aggregate.
                const auto &nn = minfo.mesh.element(eli).node_numbers();
                assert(nn.size() * 2 == static_cast<unsigned>(dsigmadu.rows()));
                for (std::size_t ni = 0; ni < nn.size(); ++ni)
                {
                    workspace(2 * nn[ni], agg_index) += dpndsigma * dsigmadu[2 * ni];
                    workspace(2 * nn[ni] + 1, agg_index) += dpndsigma * dsigmadu[2 * ni + 1];
                }
            }

            // If the above for loop is all correct, the adjoint RHS's and the first
            // term in the Jacobian are computed and we just need to do the adjoint solve,
            // subtract, and multiply by constants.
            // These constants are given by the p-norm aggregate divided by the sum inside the
            // p norm.
            for (int i = 0; i < aggs.size(); ++i)
            {
                double c = aggs[i];
                aggs[i] = std::pow(aggs[i] / counts[i], 1.0 / def.p);
                J.row(i) *= aggs[i] / c;
                workspace.col(i) *= aggs[i] / c;
            }
            eliminate_essential_boundaries(workspace, minfo);
            workspace2 = -minfo.factorized.solve(workspace);
            for (int i = 0; i < aggs.size(); ++i)
            {
                auto Jrow = J.row(i);
                evaluate_tensor_product(Jrow, workspace2.col(i), minfo);
            }
        },
        minfo);
}

namespace
{

template <class Element, class Interp>
struct KSComputer
{
    KSComputer(const Interp &interp, double p, double m, double rho) : interp(interp), p{p}, m{m}, rho{rho}
    {
    }

    template <class Computer, class... Args>
    double evaluate(const Computer &stress_computer, const Args &...args) const
    {
        const double sigma = evaluate_relaxed_stress(stress_computer, args...);
        return std::exp(p * (sigma - m));
    }

    /*
     * dst(0) holds exp(p*(sigma - m))
     * dst(1) gets exp(p*(sigma - m)) * \partial sigma / \partial rho
     * Remaining entries get exp(p*(sigma - m)) * \partial sigma / \partial u
     */
    template <class Computer, class... Args>
    void evaluate_with_gradient_terms(
        Eigen::Matrix<double, 2 * (Elasticity::TwoD::basis_size<Element> + 1), 1> &dst,
        const Computer &stress_computer, const Args &...args) const
    {
        const auto [sigma, sigma_rho, sigma_u] =
            evaluate_relaxed_stress_with_partials(stress_computer, args...);
        const double integrand = std::exp(p * (sigma - m));

        dst(0) = integrand;
        dst(1) = integrand * sigma_rho;
        dst.template tail<2 * Elasticity::TwoD::basis_size<Element>>() = integrand * sigma_u;
    }

    template <class Computer, class... Args>
    double evaluate_relaxed_stress(const Computer &stress_computer, const Args &...args) const
    {
        return interp(rho) * stress_computer.evaluate(args...);
    }

    template <class Computer, class... Args>
    auto evaluate_relaxed_stress_with_partials(const Computer &stress_computer, const Args &...args) const
    {
        double sigma_vm;
        Eigen::Matrix<double, 2 * Element::basis.size(), 1> grad;
        std::tie(sigma_vm, grad) = stress_computer.evaluate_with_gradient(args...);
        const double scaling = interp(rho);
        const double sigma = scaling * sigma_vm;
        const double sigma_rho = interp.derivative(rho) * sigma_vm;
        for (int i = 0; i < grad.size(); ++i)
        {
            grad[i] *= scaling;
        }
        return std::make_tuple(sigma, sigma_rho, grad);
    }

  private:
    Interp interp;
    double p, m, rho;
};

template <
    class Element, int Order = Galerkin::DefaultIntegrationOrder<Element>::order, class Interp,
    class Coeffs>
double integrate_over_element(
    const Element &el, const Coeffs &coeffs, const Interp &interp, double lambda, double mu, double p,
    double m, double rho)
{
    const KSComputer<Element, Interp> c(interp, p, m, rho);

    double result = 0;
    const auto quadrature_rule = el.coordinate_map().template quadrature_rule<Order>();
    const auto stress_computer = Elasticity::TwoD::VonMisesComputer<Element>(el, coeffs, lambda, mu);
    for (unsigned i = 0; i < quadrature_rule.points.size(); ++i)
    {
        double contrib = c.evaluate(stress_computer, quadrature_rule.points[i]);
        const double det = el.coordinate_map().detJ()(quadrature_rule.points[i]);
        result += quadrature_rule.weights[i] * contrib * det;
    }

    return result;
}

template <
    class Element, int Order = Galerkin::DefaultIntegrationOrder<Element>::order, class Interp,
    class Coeffs>
void integrate_over_element_with_partials(
    const Element &el, const Coeffs &coeffs, const Interp &interp, double lambda, double mu, double p,
    double m, double rho, Eigen::Matrix<double, 2 * (Element::basis.size() + 1), 1> &dst)
{
    const KSComputer<Element, Interp> c(interp, p, m, rho);

    std::decay_t<decltype(dst)> work;
    dst.fill(0);
    const auto quadrature_rule = el.coordinate_map().template quadrature_rule<Order>();

    const auto stress_computer = Elasticity::TwoD::VonMisesComputer<Element>(el, coeffs, lambda, mu);
    for (unsigned n = 0; n < quadrature_rule.points.size(); ++n)
    {
        c.evaluate_with_gradient_terms(work, stress_computer, quadrature_rule.points[n]);
        const double det = el.coordinate_map().detJ()(quadrature_rule.points[n]);
        const double w = quadrature_rule.weights[n];
        dst += w * work * det;
    }
}

template <class Mesh>
auto extract_coeffs(const Mesh &mesh, const Eigen::VectorXd &u, size_t eli)
{
    using Element = decltype(Elasticity::TwoD::instantiate_element(mesh, eli));
    assert(eli < mesh.num_elements());

    Eigen::Matrix<double, Element::basis.size() * 2, 1> coeffs;
    const auto &nn = mesh.element(eli).node_numbers();
    for (int i = 0; i < coeffs.size() / 2; ++i)
    {
        coeffs[2 * i] = u[2 * nn[i]];
        coeffs[2 * i + 1] = u[2 * nn[i] + 1];
    }

    return coeffs;
}

template <class Mesh>
Eigen::VectorXd get_max_quadrature_stresses(
    const StressCriterionDefinition &def, const ModelInfo<Mesh> &minfo, double lambda, double mu)
{
    Eigen::VectorXd maxes = Eigen::VectorXd::Zero(def.agg_regions.n);

    for (size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        const auto coeffs = extract_coeffs(minfo.mesh, minfo.displacement, eli);
        const auto el = Elasticity::TwoD::instantiate_element(minfo.mesh, eli);
        const auto comp =
            Elasticity::TwoD::VonMisesComputer<std::decay_t<decltype(el)>>(el, coeffs, lambda, mu);

        const auto rule = el.coordinate_map()
                              .template quadrature_rule<
                                  Galerkin::DefaultIntegrationOrder<std::decay_t<decltype(el)>>::order>();

        size_t which_agg = def.agg_regions.assignments[eli];

        for (auto p : rule.points)
        {
            double sigma = comp.evaluate(p);
            if (sigma > maxes[which_agg])
            {
                maxes[which_agg] = sigma;
            }
        }
    }

    return maxes;
}

template <class Mesh>
void ks_aggregates(
    Eigen::VectorXd &aggs, const StressCriterionDefinition &def, const ModelInfo<Mesh> &minfo,
    double lambda, double mu)
{
    assert(def.agg_regions.assignments.size() == minfo.mesh.num_elements());
    auto ms = get_max_quadrature_stresses(def, minfo, lambda, mu);
    aggs = Eigen::VectorXd::Zero(def.agg_regions.n);

    const auto &u = minfo.displacement;

    for (size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        auto coeffs = extract_coeffs(minfo.mesh, u, eli);
        auto which_agg = def.agg_regions.assignments[eli];
        aggs[which_agg] += integrate_over_element(
            Elasticity::TwoD::instantiate_element(minfo.mesh, eli), coeffs, def.stiffness_interp, lambda,
            mu, def.p, ms[which_agg], minfo.rho_filt[eli]);
    }

    aggs = ms.array() + (aggs.array().log() - def.alphas.array().log()) / def.p;
}

template <class Mesh>
void ks_aggregates_w_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    const StressCriterionDefinition &def, const ModelInfo<Mesh> &minfo, double lambda, double mu,
    Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2)
{
    assert(def.agg_regions.assignments.size() == minfo.mesh.num_elements());
    auto ms = get_max_quadrature_stresses(def, minfo, lambda, mu);
    aggs = Eigen::VectorXd::Zero(def.agg_regions.n);
    J = std::decay_t<decltype(J)>::Zero(def.agg_regions.n, minfo.mesh.num_elements());
    workspace = Eigen::MatrixXd::Zero(2 * minfo.mesh.num_nodes(), J.rows());
    const auto &u = minfo.displacement;

    for (size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        auto coeffs = extract_coeffs(minfo.mesh, u, eli);
        auto which_agg = def.agg_regions.assignments[eli];
        const auto &filt = minfo.filter[eli];
        auto el = Elasticity::TwoD::instantiate_element(minfo.mesh, eli);
        Eigen::Matrix<double, 2 * decltype(el)::basis.size() + 2, 1> integrated;

        integrate_over_element_with_partials(
            el, coeffs, def.stiffness_interp, lambda, mu, def.p, ms[which_agg], minfo.rho_filt[eli],
            integrated);

        aggs[which_agg] += integrated(0);
        for (auto [ri, w] : filt.entries)
        {
            J(which_agg, ri) += integrated(1) * w;
        }

        const auto &nn = minfo.mesh.element(eli).node_numbers();
        for (std::size_t ni = 0; ni < nn.size(); ++ni)
        {
            workspace(2 * nn[ni], which_agg) += integrated(2 * ni + 2);
            workspace(2 * nn[ni] + 1, which_agg) += integrated(2 * ni + 3);
        }
    }

    J.array().colwise() /= aggs.array();
    workspace.array().rowwise() /= aggs.transpose().array();
    aggs = ms.array() + (aggs.array().log() - def.alphas.array().log()) / def.p;

    eliminate_essential_boundaries(workspace, minfo);
    workspace2 = -minfo.factorized.solve(workspace);
    for (int i = 0; i < aggs.size(); ++i)
    {
        auto Jrow = J.row(i);
        evaluate_tensor_product(Jrow, workspace2.col(i), minfo);
    }
}

} // namespace

void ks_stress_aggregates(
    Eigen::VectorXd &aggs, const StressCriterionDefinition &def, const ModelInfoVariant &minfo,
    double lambda, double mu)
{
    std::visit([&, lambda, mu](const auto &minfo) { ks_aggregates(aggs, def, minfo, lambda, mu); }, minfo);
}

void ks_aggs_with_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    const StressCriterionDefinition &def, const ModelInfoVariant &minfo, double lambda, double mu,
    Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2)
{
    std::visit(
        [&, lambda, mu](const auto &minfo)
        { ks_aggregates_w_jacobian(aggs, J, def, minfo, lambda, mu, workspace, workspace2); },
        minfo);
}

namespace
{

template <class Mesh>
double
updated_max_stress(const ModelInfo<Mesh> &minfo, size_t eli, double lambda, double mu, double max_stress)
{
    const auto &el = Elasticity::TwoD::instantiate_element(minfo.mesh, eli);
    const auto u = extract_coeffs(minfo.mesh, minfo.displacement, eli);

    const auto stress_computer =
        Elasticity::TwoD::VonMisesComputer<std::decay_t<decltype(el)>>(el, u, lambda, mu);

    auto update_max = [&](auto... args)
    {
        double sigma = stress_computer.evaluate(args...);
        if (sigma > max_stress)
        {
            max_stress = sigma;
        }
    };

    // Nodal stress values.
    update_max(-1, -1);
    update_max(-1, 1);
    update_max(1, -1);

    // Quadrature points.
    const auto rule = el.coordinate_map()
                          .template quadrature_rule<
                              Galerkin::DefaultIntegrationOrder<std::decay_t<decltype(el)>>::order>();
    for (auto pt : rule.points)
    {
        update_max(pt);
    }

    return max_stress;
}

template <class Mesh>
double estimate_max_stress(const ModelInfo<Mesh> &minfo, double lambda, double mu)
{
    double max_stress = 0;

    for (size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        max_stress = updated_max_stress(minfo, eli, lambda, mu, max_stress);
    }

    return max_stress;
}

template <class Mesh>
double ks_alpha_contrib(
    const ModelInfo<Mesh> &minfo, size_t eli, double lambda, double mu, double p, double max_stress)
{
    const auto u = extract_coeffs(minfo.mesh, minfo.displacement, eli);
    const auto el = Elasticity::TwoD::instantiate_element(minfo.mesh, eli);
    const auto stress_computer = Elasticity::TwoD::VonMisesComputer<decltype(el)>(el, u, lambda, mu);

    const auto rule = el.coordinate_map()
                          .template quadrature_rule<
                              Galerkin::DefaultIntegrationOrder<std::decay_t<decltype(el)>>::order>();

    double integrated = 0;
    for (size_t i = 0; i < rule.points.size(); ++i)
    {
        const double integrand = std::exp(p * (stress_computer.evaluate(rule.points[i]) - max_stress));
        const double det = el.coordinate_map().detJ()(rule.points[i]);
        integrated += integrand * det * rule.weights[i];
    }

    return integrated;
}

template <class Mesh>
double estimate_ks_alpha(const ModelInfo<Mesh> &minfo, double lambda, double mu, double p)
{
    const double max_stress = estimate_max_stress(minfo, lambda, mu);
    fmt::print("  Max. stress in the domain is {:E}\n", max_stress);
    double integral = 0;

    for (size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
    {
        integral += ks_alpha_contrib(minfo, eli, lambda, mu, p, max_stress);
    }

    return integral;
}

} // namespace

double estimate_max_stress(const ModelInfoVariant &minfo, double lambda, double mu)
{
    return std::visit([=](const auto &minfo) { return estimate_max_stress(minfo, lambda, mu); }, minfo);
}

double estimate_ks_alpha(const ModelInfoVariant &minfo, double lambda, double mu, double p, double frac)
{
    fmt::print("Estimating K-S normalization...\n");
    double integral =
        std::visit([=](const auto &minfo) { return estimate_ks_alpha(minfo, lambda, mu, p); }, minfo);

    double alpha = integral * frac;
    fmt::print("Estimated alpha as {:E}\n", alpha);
    return alpha;
}
