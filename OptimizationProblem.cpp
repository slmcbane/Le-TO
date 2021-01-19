#include "OptimizationProblem.hpp"

double OptimizationProblem::nlopt_f(unsigned n, const double *x, double *grad, void *f_data)
{
    OptimizationProblem *problem = reinterpret_cast<OptimizationProblem *>(f_data);

    double objective;
    problem->eval_f(n, x, true, objective);
    fmt::print(FMT_STRING("New objective value: {:E}\n"), objective);

    if (grad != nullptr)
    {
        problem->eval_grad_f(n, x, false, grad);
    }

    return objective;
}

void OptimizationProblem::nlopt_g(
    unsigned, double *result, unsigned, const double *x, double *grad, void *f_data)
{
    OptimizationProblem *problem = reinterpret_cast<OptimizationProblem *>(f_data);
    problem->maybe_update_parameter(x, true);

    switch (problem->m_problem_type)
    {
    case MWS:
    {
        if (grad != nullptr)
        {
            const auto &J = problem->m_evaluator.stress_agg_jacobian();
            Eigen::Map<std::decay_t<decltype(J)>>(grad, J.rows(), J.cols()) =
                problem->m_stress_normalization.asDiagonal() * J;
        }
        const Eigen::ArrayXd constraints =
            problem->m_evaluator.stress_aggregates().cwiseProduct(problem->m_stress_normalization);
        Eigen::Map<Eigen::VectorXd>(result, constraints.size()) =
            constraints - problem->m_options.stress_limit.value();

        fmt::print("Stress constraint values:\n  [");
        for (int i = 0; i < constraints.size() - 1; ++i)
        {
            fmt::print(FMT_STRING(" {:E},"), constraints(i));
        }
        fmt::print(FMT_STRING(" {:E}]\n\n"), constraints(constraints.size() - 1));
        break;
    }
    case MWCS:
    {
        if (grad != nullptr)
        {
            const auto &J = problem->m_evaluator.stress_agg_jacobian();
            const auto &gc = problem->m_evaluator.compliance_gradient();
            auto full_jac = Eigen::Map<std::decay_t<decltype(J)>>(grad, J.rows() + 1, J.cols());
            full_jac.row(0) = gc.transpose();
            full_jac.bottomRows(J.rows()) = problem->m_stress_normalization.asDiagonal() * J;
        }
        const Eigen::ArrayXd constraints =
            problem->m_evaluator.stress_aggregates().cwiseProduct(problem->m_stress_normalization);
        result[0] = problem->m_evaluator.compliance() - problem->m_options.compliance_limit.value();
        Eigen::Map<Eigen::VectorXd>(result + 1, constraints.size()) =
            constraints - problem->m_options.stress_limit.value();

        fmt::print("Stress constraint values:\n  [");
        for (int i = 0; i < constraints.size() - 1; ++i)
        {
            fmt::print(FMT_STRING(" {:E},"), constraints(i));
        }
        fmt::print(FMT_STRING(" {:E}]\n"), constraints(constraints.size() - 1));
        fmt::print(FMT_STRING("Compliance constraint value: {:E}\n\n"), problem->m_evaluator.compliance());
        break;
    }
    case MCW:
    {
        if (grad != nullptr)
        {
            const auto &gw = problem->m_evaluator.relative_masses();
            memcpy(grad, gw.data(), sizeof(double) * gw.size());
        }
        double total_mass = problem->m_evaluator.sum_mass_times_density();
        result[0] = total_mass - problem->m_options.mass_limit.value();
        fmt::print(FMT_STRING("Mass constraint value: {:E}\n\n"), total_mass);
        break;
    }
    default:
        fmt::print(stderr, "Unimplemented optimization type\n");
        throw std::exception{};
    }
}

nlopt::opt OptimizationProblem::get_optimizer()
{
    unsigned n = num_elements(m_evaluator.model_info());
    nlopt::opt optimizer(nlopt::LD_MMA, n);
    optimizer.set_min_objective(OptimizationProblem::nlopt_f, this);

    optimizer.set_lower_bounds(0);
    optimizer.set_upper_bounds(1);

    unsigned m =
        m_problem_type == MWS
            ? m_evaluator.num_aggregates()
            : m_problem_type == MCW ? 1 : m_problem_type == MWCS ? m_evaluator.num_aggregates() + 1 : -1;
    optimizer.add_inequality_mconstraint(OptimizationProblem::nlopt_g, this, std::vector<double>(m));

    if (m_options.max_iters)
    {
        optimizer.set_maxeval(*m_options.max_iters);
    }

    if (m_options.mean_change_tol)
    {
        optimizer.set_xtol_rel(*m_options.mean_change_tol);
    }

    if (m_options.absolute_tol)
    {
        optimizer.set_ftol_rel(*m_options.absolute_tol);
    }

    return optimizer;
}

/*
bool OptimizationProblem::get_nlp_info(
    unsigned &n, unsigned &m, unsigned &nnz_jac_g, unsigned &nnz_h_lag, IndexStyleEnum &unsigned_style)
{
    n = num_elements(m_evaluator.model_info());

    switch (m_problem_type)
    {
    case MWS:
        m = m_evaluator.num_aggregates();
        break;
    case MWCS:
    case MCWS:
        m = m_evaluator.num_aggregates() + 1;
        break;
    case MCW:
    case MSW:
        m = 1;
        break;
    default:
        return false;
    }

    nnz_jac_g = n * m;
    nnz_h_lag = 0;
    unsigned_style = C_STYLE;

    return true;
} */

bool OptimizationProblem::get_bounds_info(
    unsigned n, double *x_l, double *x_u, unsigned m, double *g_l, double *g_u)
{
    for (unsigned i = 0; i < n; ++i)
    {
        x_l[i] = 0;
        x_u[i] = 1;
    }

    switch (m_problem_type)
    {
    case MWS:
        for (unsigned i = 0; i < m; ++i)
        {
            g_l[i] = -1e20;
            g_u[i] = m_options.stress_limit.value();
        }
        break;
    case MWCS:
        g_l[0] = -1e20;
        g_u[0] = m_options.compliance_limit.value();
        for (unsigned i = 1; i < m; ++i)
        {
            g_l[i] = -1e20;
            g_u[i] = m_options.stress_limit.value();
        }
        break;
    case MCW:
        g_l[0] = -1e20;
        g_u[0] = m_options.mass_limit.value();
        break;
    default:
        return false;
    }

    return true;
}

bool OptimizationProblem::get_starting_point(
    unsigned n, bool init_x, double *x, bool init_z, double *, double *, unsigned, bool init_lambda,
    double *)
{
    assert(!init_z && !init_lambda && init_x && x != nullptr);
    assert(n == m_initial_condition.size());
    for (unsigned i = 0; i < n; ++i)
    {
        x[i] = m_initial_condition[i];
    }

    return true;
}

bool OptimizationProblem::eval_f(unsigned, const double *x, bool new_x, double &obj_value)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    case MWCS:
        obj_value = m_evaluator.sum_mass_times_density();
        return true;
    case MCW:
        obj_value = m_evaluator.compliance();
        return true;
    default:
        return false;
    }
}

bool OptimizationProblem::eval_grad_f(unsigned n, const double *x, bool new_x, double *grad_f)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    case MWCS:
    {
        const std::vector<double> &rel_masses = m_evaluator.relative_masses();
        for (unsigned i = 0; i < n; ++i)
        {
            grad_f[i] = rel_masses[i];
        }
        return true;
    }
    case MCW:
    {
        const Eigen::VectorXd &grad = m_evaluator.compliance_gradient();
        assert(grad.size() == n);
        for (unsigned i = 0; i < n; ++i)
        {
            grad_f[i] = grad[i];
        }
        return true;
    }
    default:
        return false;
    }
}

bool OptimizationProblem::eval_g(unsigned, const double *x, bool new_x, unsigned m, double *g)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    {
        const Eigen::VectorXd &aggs = m_evaluator.stress_aggregates();
        for (unsigned i = 0; i < m; ++i)
        {
            g[i] = aggs[i] * m_stress_normalization(i);
        }
        return true;
    }
    case MWCS:
    {
        const Eigen::VectorXd &aggs = m_evaluator.stress_aggregates();
        g[0] = m_evaluator.compliance();
        for (unsigned i = 1; i < m; ++i)
        {
            g[i] = aggs[i - 1] * m_stress_normalization(i);
        }
        return true;
    }
    case MCW:
        g[0] = m_evaluator.sum_mass_times_density();
        return true;
    default:
        return false;
    }
}

bool OptimizationProblem::eval_jac_g(
    unsigned n, const double *x, bool new_x, unsigned m, unsigned nele_jac, unsigned *iRow, unsigned *jCol,
    double *values)
{
    if (x == nullptr)
    {
        assert(iRow != nullptr && jCol != nullptr && values == nullptr);
        unsigned offset = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            for (unsigned j = 0; j < n; ++j)
            {
                iRow[offset] = i;
                jCol[offset] = j;
                offset += 1;
            }
        }
        return true;
    }
    else
    {
        maybe_update_parameter(x, new_x);
        switch (m_problem_type)
        {
        case MWS:
        {
            const auto &J = m_evaluator.stress_agg_jacobian();
            std::size_t offset = 0;
            for (long i = 0; i < m; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    values[offset++] = J(i, j) * m_stress_normalization[i];
                }
            }
            return true;
        }
        case MWCS:
        {
            const auto &cg = m_evaluator.compliance_gradient();
            memcpy(values, cg.data(), sizeof(double) * n);
            const auto &J = m_evaluator.stress_agg_jacobian();
            std::size_t offset = 0;
            for (long i = 0; i < m; ++i)
            {
                for (long j = 0; j < n; ++j)
                {
                    values[n + offset++] = J(i, j) * m_stress_normalization[i];
                }
            }
            return true;
        }
        case MCW:
        {
            const auto &grad = m_evaluator.relative_masses();
            memcpy(values, grad.data(), sizeof(double) * n);
            return true;
        }
        default:
            return false;
        }
    }
}

/*
bool OptimizationProblem::intermediate_callback(
    Ipopt::AlgorithmMode mode, unsigned, double, double, double, double, double, double, double, double,
unsigned, const Ipopt::IpoptData *, Ipopt::IpoptCalculatedQuantities *)
{
    if (mode == Ipopt::AlgorithmMode::RestorationPhaseMode)
    {
        return true;
    }
    update_mean_changes();
    if (m_iter > m_options.mean_change_iters.value() && m_mean_change.mean() <
m_options.mean_change_tol.value())
    {
        std::cout << m_mean_change << std::endl;
        return false;
    }

    update_stress_normalization();
    maybe_update_region_definitions();

    m_iter += 1;

    return true;
} */

void OptimizationProblem::update_mean_changes()
{
    m_mean_change[m_iter % m_mean_change.size()] =
        (m_evaluator.parameter() - m_last_parameter).cwiseAbs().mean();
    m_last_parameter = m_evaluator.parameter();
}

void OptimizationProblem::update_stress_normalization()
{
    Eigen::VectorXd max_stresses = m_evaluator.max_stresses();
    m_stress_normalization = m_stress_alpha * max_stresses.cwiseQuotient(m_evaluator.stress_aggregates()) +
                             (1 - m_stress_alpha) * m_stress_normalization;
    fmt::print(FMT_STRING("Updated stress normalizations:\n  ["));
    for (int i = 0; i < m_stress_normalization.size() - 1; ++i)
    {
        fmt::print(FMT_STRING(" {:E},"), m_stress_normalization(i));
    }
    fmt::print(FMT_STRING(" {:E}]\n\n"), m_stress_normalization(m_stress_normalization.size() - 1));
}

void OptimizationProblem::maybe_update_region_definitions()
{
    if (m_iter % m_reassign_interval == 0 && m_iter != 0)
    {
        fmt::print(FMT_STRING("Reassigning aggregation regions...\n\n"));
        m_evaluator.reassign_aggregation_regions();
    }
}
