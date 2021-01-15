#include "evaluator.hpp"
#include "2d.hpp"
#include "stress.hpp"

void Evaluator::set_parameter(const double *rho)
{
    update_model_info(*m_minfo, rho);
    parameter_value = Eigen::Map<const Eigen::VectorXd>(rho, num_elements(*m_minfo));
    parameter_set = true;
    solved_forward = compliance_computed = compliance_gradient_computed = false;
    cc_stress_computed = aggregates_computed = aggj_computed = false;
}

void Evaluator::solve_forward()
{
    assert(parameter_set && "Set parameter before solving");

    std::visit(
        [](auto &minfo) { minfo.displacement = minfo.factorized.solve(minfo.nodal_forcing); }, *m_minfo);

    solved_forward = true;
}

const Eigen::VectorXd &Evaluator::displacement()
{
    if (!solved_forward)
    {
        solve_forward();
    }
    const Eigen::VectorXd *u = std::visit([](const auto &minfo) { return &minfo.displacement; }, *m_minfo);
    return *u;
}

double Evaluator::compliance()
{
    if (!compliance_computed)
    {
        compute_compliance();
    }
    return compliance_value;
}

void Evaluator::compute_compliance()
{
    const Eigen::SparseMatrix<double> &K =
        *std::visit([](const auto &minfo) { return &minfo.forward_stiffness_eigen; }, *m_minfo);

    const Eigen::VectorXd &u = displacement();

    compliance_value = u.dot(K * u);
    compliance_computed = true;
}

void Evaluator::compute_compliance_gradient()
{
    compliance_gradient_value = Eigen::VectorXd::Zero(num_elements(*m_minfo));

    std::visit(
        [&](const auto &minfo) {
            evaluate_tensor_product(compliance_gradient_value, -displacement(), minfo);
        },
        *m_minfo);

    compliance_gradient_computed = true;
}

const Eigen::VectorXd &Evaluator::compliance_gradient()
{
    if (!compliance_gradient_computed)
    {
        compute_compliance_gradient();
    }

    return compliance_gradient_value;
}

const Eigen::VectorXd &Evaluator::cell_centered_stress()
{
    if (!cc_stress_computed)
    {
        compute_cc_stress();
    }

    return cc_stress_value;
}

void Evaluator::compute_cc_stress()
{
    if (!solved_forward)
    {
        solve_forward();
    }
    cc_stress_value.resize(num_elements(*m_minfo));
    ::cell_centered_stress(cc_stress_value, *m_minfo, lambda, mu);
    cc_stress_computed = true;
}

const Eigen::VectorXd &Evaluator::stress_aggregates()
{
    if (!aggregates_computed)
    {
        compute_aggregates();
    }

    return aggregate_values;
}

void Evaluator::compute_aggregates()
{
    check_stress_defined();
    if (!solved_forward)
    {
        solve_forward();
    }

    pnorm_stress_aggregates(aggregate_values, cc_stress_value, *stress_criterion, *m_minfo, lambda, mu);

    cc_stress_computed = true;
    aggregates_computed = true;
}

const decltype(Evaluator::agg_jacobian) &Evaluator::stress_agg_jacobian()
{
    if (!aggj_computed)
    {
        compute_aggregates_and_jac();
    }

    return agg_jacobian;
}

void Evaluator::compute_aggregates_and_jac()
{
    check_stress_defined();
    if (!solved_forward)
    {
        solve_forward();
    }

    pnorm_aggs_with_jacobian(
        aggregate_values, agg_jacobian, cc_stress_value, *stress_criterion, *m_minfo, lambda, mu, workspace,
        workspace2);

    cc_stress_computed = true;
    aggregates_computed = true;
    aggj_computed = true;
}

void Evaluator::compute_relative_areas()
{
    double total_area = 0;
    std::visit(
        [&](const auto &minfo) {
            const auto &mesh = minfo.mesh;
            for (std::size_t eli = 0; eli < mesh.num_elements(); ++eli)
            {
                const auto el = Elasticity::TwoD::instantiate_element(mesh, eli);
                relative_areas[eli] = el.integrate(
                    Galerkin::Functions::ConstantFunction<int>(1), Galerkin::IntegrationOrder<1>{});
                total_area += relative_areas[eli];
            }
        },
        *m_minfo);
    for (double &x : relative_areas)
    {
        x /= total_area;
    }
}

double Evaluator::sum_mass_times_density() const
{
    const auto &rho = *std::visit([&](const auto &minfo) { return &minfo.rho_filt; }, *m_minfo);
    double sum = 0;
    for (std::size_t eli = 0; eli < static_cast<unsigned>(rho.size()); ++eli)
    {
        sum += rho[eli] * relative_areas[eli];
    }
    return sum;
}

Eigen::VectorXd Evaluator::max_stresses()
{
    check_stress_defined();
    const Eigen::VectorXd &cc_stress = cell_centered_stress();

    const auto &assignments = stress_criterion->agg_regions.assignments;

    Eigen::VectorXd max_stress = Eigen::VectorXd::Zero(stress_criterion->agg_regions.n);
    for (std::size_t eli = 0; eli < assignments.size(); ++eli)
    {
        std::size_t agg_index = assignments[eli];
        if (cc_stress[agg_index] > max_stress[agg_index])
        {
            max_stress[agg_index] = cc_stress[agg_index];
        }
    }
    return max_stress;
}

void Evaluator::reassign_aggregation_regions()
{
    check_stress_defined();
    stress_criterion->agg_regions =
        assign_agg_regions(cell_centered_stress(), stress_criterion->agg_regions.n);
}
