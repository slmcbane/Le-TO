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