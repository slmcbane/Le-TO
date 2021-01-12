#include "evaluator.hpp"

void Evaluator::set_parameter(const double *rho)
{
    update_model_info(*m_minfo, rho);
    parameter_set = true;
    solved_forward = compliance_computed = compliance_gradient_computed = false;
}

void Evaluator::solve_forward()
{
    assert(parameter_set && "Set parameter before solving");

    std::visit([](auto &minfo)
    {
        minfo.displacement = minfo.factorized.solve(minfo.nodal_forcing);
    }, *m_minfo);

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
    const Eigen::SparseMatrix<double> &K = *std::visit(
        [](const auto &minfo) { return &minfo.forward_stiffness_eigen; },
        *m_minfo
    );

    const Eigen::VectorXd &u = displacement();

    compliance_value = u.dot(K * u);
    compliance_computed = true;
}

void Evaluator::compute_compliance_gradient()
{
    compliance_gradient_value.resize(num_elements(*m_minfo));

    std::visit([&](const auto &minfo)
    {
        evaluate_tensor_product(compliance_gradient_value, -displacement(), minfo);
    }, *m_minfo);

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
