#include "OptimizationProblem.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

bool OptimizationProblem::get_nlp_info(
    Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag, IndexStyleEnum &index_style)
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
    index_style = C_STYLE;

    return true;
}

bool OptimizationProblem::get_bounds_info(
    Index n, double *x_l, double *x_u, Index m, double *g_l, double *g_u)
{
    for (Index i = 0; i < n; ++i)
    {
        x_l[i] = 0;
        x_u[i] = 1;
    }

    switch (m_problem_type)
    {
    case MWS:
        for (Index i = 0; i < m; ++i)
        {
            g_l[i] = -1e20;
            g_u[i] = 1.0;
        }
        break;
    case MWCS:
        g_l[0] = -1e20;
        g_u[0] = m_options.compliance_limit.value();
        for (Index i = 1; i < m; ++i)
        {
            g_l[i] = -1e20;
            g_u[i] = 1.0;
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
    Index n, bool init_x, double *x, bool init_z, double *, double *, Index, bool init_lambda, double *)
{
    assert(!init_z && !init_lambda && init_x && x != nullptr);
    assert(n == m_initial_condition.size());
    for (Index i = 0; i < n; ++i)
    {
        x[i] = m_initial_condition[i];
    }

    return true;
}

bool OptimizationProblem::eval_f(Index, const double *x, bool new_x, double &obj_value)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    case MWCS:
        obj_value = m_evaluator.sum_mass_times_density();
        fmt::print("Current objective value: {:E}\n", obj_value);
        return true;
    case MCW:
        obj_value = m_evaluator.compliance();
        return true;
    default:
        return false;
    }
}

bool OptimizationProblem::eval_grad_f(Index n, const double *x, bool new_x, double *grad_f)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    case MWCS:
    {
        const std::vector<double> &rel_masses = m_evaluator.relative_masses();
        for (Index i = 0; i < n; ++i)
        {
            grad_f[i] = rel_masses[i];
        }
        return true;
    }
    case MCW:
    {
        const Eigen::VectorXd &grad = m_evaluator.compliance_gradient();
        assert(grad.size() == n);
        for (Index i = 0; i < n; ++i)
        {
            grad_f[i] = grad[i];
        }
        return true;
    }
    default:
        return false;
    }
}

bool OptimizationProblem::eval_g(Index, const double *x, bool new_x, Index m, double *g)
{
    maybe_update_parameter(x, new_x);
    switch (m_problem_type)
    {
    case MWS:
    {
        const Eigen::VectorXd &aggs = m_evaluator.stress_aggregates();
        for (Index i = 0; i < m; ++i)
        {
            g[i] = aggs[i] * m_stress_normalization(i);
        }
        return true;
    }
    case MWCS:
    {
        const Eigen::VectorXd &aggs = m_evaluator.stress_aggregates();
        g[0] = m_evaluator.compliance();
        for (Index i = 1; i < m; ++i)
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
    Index n, const double *x, bool new_x, Index m, Index nele_jac, Index *iRow, Index *jCol, double *values)
{
    if (x == nullptr)
    {
        assert(iRow != nullptr && jCol != nullptr && values == nullptr);
        Index offset = 0;
        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n; ++j)
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

bool OptimizationProblem::intermediate_callback(
    Ipopt::AlgorithmMode mode, Index, double, double, double, double, double, double, double, double, Index,
    const Ipopt::IpoptData *, Ipopt::IpoptCalculatedQuantities *)
{
    if (mode == Ipopt::AlgorithmMode::RestorationPhaseMode)
    {
        return true;
    }
    update_mean_changes();
    if (m_iter > m_options.mean_change_iters.value() &&
        m_mean_change.mean() < m_options.mean_change_tol.value())
    {
        std::cout << m_mean_change << std::endl;
        return false;
    }

    if (m_problem_type != MCW)
    {
        const Eigen::VectorXd &aggs = m_evaluator.stress_aggregates();
        fmt::print("Agg. values: [");
        for (int i = 0; i < aggs.size(); ++i)
        {
            fmt::print((i == aggs.size() - 1) ? "{:E}]\n" : "{:E}, ", aggs[i]);
        }

        update_stress_normalization();
        fmt::print("With updated scaling: [");
        for (int i = 0; i < aggs.size(); ++i)
        {
            fmt::print((i == aggs.size() - 1) ? "{:E}]\n" : "{:E}, ", aggs[i] * m_stress_normalization[i]);
        }

        maybe_update_region_definitions();
    }

    m_iter += 1;

    return true;
}

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
}

void OptimizationProblem::maybe_update_region_definitions()
{
    if (m_iter % m_reassign_interval == 0 && m_iter != 0)
    {
        m_evaluator.reassign_aggregation_regions();
    }
}
