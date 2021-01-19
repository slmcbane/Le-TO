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

template <class Matrix, class Mesh>
void eliminate_essential_boundaries(Matrix &workspace, const ModelInfo<Mesh> &minfo)
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
        [&, lambda, mu](const auto &minfo) {
            cell_centered_stress(dest, minfo.displacement, minfo.mesh, lambda, mu);
        },
        minfo);
}

void pnorm_stress_aggregates(
    Eigen::VectorXd &aggregates, Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def,
    const ModelInfoVariant &minfo, double lambda, double mu)
{
    aggregates = Eigen::VectorXd::Zero(def.agg_regions.n);
    auto counts = std::vector<int>(aggregates.size(), 0);
    std::visit(
        [&, lambda, mu](const auto &minfo) {
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

    std::size_t index = 0;
    for (std::size_t region = 0; region < n; ++region)
    {
        for (std::size_t i = 0; i < indices.size() / n; ++i)
        {
            agg_regions.assignments[indices[index++]] = region;
        }
    }
    for (; index < agg_regions.assignments.size(); ++index)
    {
        agg_regions.assignments[indices[index]] = n - 1;
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
        [&](const auto &minfo) {
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

template <class VM, class Interp>
struct InducedAggregateComputer
{
    VM vm_computer;
    Interp interp;
    double p, m, rho;

    InducedAggregateComputer(VM &&c, const Interp &interp, double p, double m, double rho)
        : vm_computer(std::move(c)), interp(interp), p{p}, m{m}, rho{rho}
    {
    }

    template <class... Args>
    std::array<double, 2> evaluate_numerator_and_denominator(const Args &... args) const
    {
        std::array<double, 2> terms;
        const double sigma = evaluate_relaxed_stress(args...);
        const double e = std::exp(p * (sigma - m));
        terms[0] = sigma * e;
        terms[1] = e;
        return terms;
    }

    /*
     * On return, dst(0, 0) holds the numerator term; dst(0, 1) holds the denominator
     * term; dst(1, 0) has the first term for the integral of g(sigma*)*dsigma/drho in eq. (14);
     * dst(1, 1) has the second term; then the remainder of each column holds the two adjoint
     * integrands.
     */
    template <class... Args>
    void evaluate_with_gradient_terms(
        Eigen::Matrix<double, VM::coeffs_size + 2, 2> &dst, const Args &... args) const
    {
        const auto [sigma, sigma_rho, sigma_u] = evaluate_relaxed_stress_with_partials(args...);
        const double e = std::exp(p * (sigma - m));
        const double num = sigma * e;
        const double den = e;

        if (std::isnan(num) || std::isinf(num))
        {
            printf("sigma = %E  m = %E  num = %E  den = %E\n", sigma, m, num, den);
        }

        dst(0, 0) = num;
        dst(1, 0) = den * sigma_rho;
        for (int i = 2; i < dst.rows(); ++i)
        {
            dst(i, 0) = den * sigma_u(i - 2);
        }
        dst(0, 1) = den;
        dst(1, 1) = num * sigma_rho;
        for (int i = 2; i < dst.rows(); ++i)
        {
            dst(i, 1) = num * sigma_u(i - 2);
        }
    }

    template <class... Args>
    double evaluate_relaxed_stress(const Args &... args) const
    {
        return interp(rho) * vm_computer.evaluate(args...);
    }

    template <class... Args>
    auto evaluate_relaxed_stress_with_partials(const Args &... args) const
    {
        double sigma_vm;
        Eigen::Matrix<double, VM::coeffs_size, 1> grad;
        std::tie(sigma_vm, grad) = vm_computer.evaluate_with_gradient(args...);
        const double scaling = interp(rho);
        const double sigma = scaling * sigma_vm;
        const double sigma_rho = interp.derivative(rho) * sigma_vm;
        for (int i = 0; i < grad.size(); ++i)
        {
            grad[i] *= scaling;
        }
        return std::make_tuple(sigma, sigma_rho, grad);
    }
};

const auto quad_rule = Galerkin::Quadrature::triangle_rule<double, 5>;

template <class Mesh, class Soln>
Eigen::Vector2d integrate_iagg_over_element(
    const Mesh &mesh, std::size_t eli, const Soln &u, const StressCriterionDefinition &def, double lambda,
    double mu, double m, const Eigen::VectorXd &rho_filt, Eigen::VectorXd &cc_stress)
{
    const auto c = InducedAggregateComputer(
        vmcomputer(u, mesh, lambda, mu, eli), def.stiffness_interp, def.p, m, rho_filt[eli]);

    Eigen::Vector2d result = Eigen::Vector2d::Zero();

    for (int i = 0; i < quad_rule.points.size(); ++i)
    {
        std::array<double, 2> contrib = c.evaluate_numerator_and_denominator(quad_rule.points[i]);
        const double det = c.vm_computer.element().coordinate_map().detJ()(quad_rule.points[i]);
        result[0] += quad_rule.weights[i] * contrib[0] * det;
        result[1] += quad_rule.weights[i] * contrib[1] * det;
    }

    cc_stress[eli] = c.vm_computer.evaluate(-1.0 / 3, -1.0 / 3) * def.stiffness_interp(rho_filt[eli]);

    return result;
}

template <class Mesh, class Soln>
auto integrate_over_element_with_partials(
    const Mesh &mesh, std::size_t eli, const Soln &u, const StressCriterionDefinition &def, double lambda,
    double mu, double m, const Eigen::VectorXd &rho_filt, Eigen::VectorXd &cc_stress)
{
    const auto c = InducedAggregateComputer(
        vmcomputer(u, mesh, lambda, mu, eli), def.stiffness_interp, def.p, m, rho_filt[eli]);

    Eigen::Matrix<double, decltype(c.vm_computer)::coeffs_size + 2, 2> terms;
    terms = decltype(terms)::Zero();

    decltype(terms) work;

    for (int n = 0; n < quad_rule.points.size(); ++n)
    {
        c.evaluate_with_gradient_terms(work, quad_rule.points[n]);
        const double det = c.vm_computer.element().coordinate_map().detJ()(quad_rule.points[n]);
        const double w = quad_rule.weights[n];
        terms += w * det * work;
    }

    cc_stress[eli] = c.vm_computer.evaluate(-1.0 / 3, -1.0 / 3) * def.stiffness_interp(rho_filt[eli]);

    return terms;
}

template <class Mesh, class Soln>
auto accumulate_on_mesh(
    const Mesh &mesh, const Soln &u, const StressCriterionDefinition &def, double lambda, double mu,
    const Eigen::VectorXd &m, const Eigen::VectorXd &rho_filt, Eigen::VectorXd &cc_stress,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J, Eigen::MatrixXd &workspace,
    Eigen::MatrixXd &workspace2)
{
    Eigen::Matrix<double, Eigen::Dynamic, 2> num_den;
    num_den = decltype(num_den)::Zero(def.agg_regions.n, 2);

    J = std::decay_t<decltype(J)>::Zero(def.agg_regions.n, mesh.num_elements());
    workspace = Eigen::MatrixXd::Zero(2 * mesh.num_nodes(), 2 * def.agg_regions.n);
    workspace2 = Eigen::MatrixXd::Zero(mesh.num_elements(), def.agg_regions.n);

    for (std::size_t eli = 0; eli < mesh.num_nodes(); ++eli)
    {
        const auto agg_index = def.agg_regions.assignments[eli];
        const auto el_contrib = integrate_over_element_with_partials(
            mesh, eli, u, def, lambda, mu, m[agg_index], rho_filt, cc_stress);

        num_den.row(agg_index) += el_contrib.row(0);
        J(agg_index, eli) += el_contrib(1, 0);
        workspace2(eli, agg_index) += el_contrib(1, 1);

        const auto adjoint_contribs = el_contrib.bottomRows(el_contrib.rows() - 2);
        const auto &nn = mesh.element(eli).node_numbers();

        assert(nn.size() * 2 == adjoint_contribs.rows());
        for (std::size_t n = 0; n < nn.size(); ++n)
        {
            workspace(2 * nn[n], agg_index) += adjoint_contribs(2 * n, 0);
            workspace(2 * nn[n] + 1, agg_index) += adjoint_contribs(2 * n + 1, 0);
            workspace(2 * nn[n], agg_index + def.agg_regions.n) += adjoint_contribs(2 * n, 1);
            workspace(2 * nn[n] + 1, agg_index + def.agg_regions.n) += adjoint_contribs(2 * n + 1, 1);
        }
    }

    Eigen::VectorXd aggregated = num_den.col(0).cwiseQuotient(num_den.col(1));
    Eigen::VectorXd one_minus_p_times_aggregate = -def.p * aggregated.array() + 1;
    for (std::size_t i = 0; i < def.agg_regions.n; ++i)
    {
        const double one_minus_p = -def.p * aggregated(i) + 1;
        const double one_minus_p_over_d = one_minus_p / num_den(i, 1);
        const double p_over_d = def.p / num_den(i, 1);
        for (std::size_t j = 0; j < mesh.num_elements(); ++j)
        {
            J(i, j) = one_minus_p_over_d * J(i, j) + p_over_d * workspace2(j, i);
        }

        assert(static_cast<unsigned>(workspace.rows()) == 2 * mesh.num_nodes());
        for (std::size_t j = 0; j < 2 * mesh.num_nodes(); ++j)
        {
            workspace(j, i) = one_minus_p_over_d * workspace(j, i) + p_over_d * workspace(j, i + def.agg_regions.n);
        }
    }

    return aggregated;
}

template <class Mesh, class Solution, class Rule>
Eigen::VectorXd max_relaxed_quadrature_stress(
    const Mesh &mesh, const Solution &soln, double lambda, double mu, const Eigen::VectorXd &rho_filt,
    const Rule &rule, const StressCriterionDefinition &def)
{
    assert(soln.size() == 2 * mesh.num_nodes());
    Eigen::VectorXd max_stress = Eigen::VectorXd::Zero(def.agg_regions.n);

    for (size_t i = 0; i < mesh.num_elements(); ++i)
    {
        const auto stress_computer = vmcomputer(soln, mesh, lambda, mu, i);
        const double scaling = def.stiffness_interp(rho_filt[i]);
        const long which_region = def.agg_regions.assignments[i];

        for (int n = 0; n < rule.points.size(); ++n)
        {
            double sigma = scaling * stress_computer.evaluate(rule.points[n]);
            if (sigma > max_stress[which_region])
            {
                max_stress[which_region] = sigma;
            }
        }
    }
    return max_stress;
}

} // namespace

void induced_stress_aggregates(
    Eigen::VectorXd &aggs, Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def,
    const ModelInfoVariant &minfo, double lambda, double mu)
{
    Eigen::Matrix<double, Eigen::Dynamic, 2> num_den;
    num_den = decltype(num_den)::Zero(def.agg_regions.n, 2);
    std::visit(
        [&](const auto &minfo) {
            const auto max_stresses = max_relaxed_quadrature_stress(
                minfo.mesh, minfo.displacement, lambda, mu, minfo.rho_filt, quad_rule, def);

            for (std::size_t eli = 0; eli < minfo.mesh.num_elements(); ++eli)
            {
                const auto agg_index = def.agg_regions.assignments[eli];
                num_den.row(agg_index) += integrate_iagg_over_element(
                    minfo.mesh, eli, minfo.displacement, def, lambda, mu,
                    max_stresses[def.agg_regions.assignments[eli]], minfo.rho_filt, cc_stress);
            }

            aggs = num_den.col(0).cwiseQuotient(num_den.col(1));
        },
        minfo);
}

void induced_aggs_with_jacobian(
    Eigen::VectorXd &aggs, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &J,
    Eigen::VectorXd &cc_stress, const StressCriterionDefinition &def, const ModelInfoVariant &minfo,
    double lambda, double mu, Eigen::MatrixXd &workspace, Eigen::MatrixXd &workspace2)
{
    std::visit(
        [&](const auto &minfo) {
            const auto max_stresses = max_relaxed_quadrature_stress(
                minfo.mesh, minfo.displacement, lambda, mu, minfo.rho_filt, quad_rule, def);

            aggs = accumulate_on_mesh(
                minfo.mesh, minfo.displacement, def, lambda, mu, max_stresses, minfo.rho_filt, cc_stress, J,
                workspace, workspace2);

            auto adjoint_rhs = workspace.leftCols(def.agg_regions.n);
            eliminate_essential_boundaries(adjoint_rhs, minfo);
            workspace2 = minfo.factorized.solve(adjoint_rhs);
            for (int i = 0; i < aggs.size(); ++i)
            {
                auto Jrow = J.row(i);
                evaluate_tensor_product(Jrow, workspace2.col(i), minfo);
            }
        },
        minfo);
}