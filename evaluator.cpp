#include "evaluator.hpp"
#include "stress.hpp"

void Evaluator::set_parameter(const double *rho)
{
    update_model_info(*m_minfo, rho);
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