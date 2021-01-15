#ifndef OPTIMIZATION_PROBLEM_HPP
#define OPTIMIZATION_PROBLEM_HPP

#include "IpTNLP.hpp"

#include "evaluator.hpp"
#include "options.hpp"

using Ipopt::Index;

class OptimizationProblem : public Ipopt::TNLP
{
  public:
    OptimizationProblem(
        Evaluator &evaluator, const OptimizationOptions &opt_options, OptimizationType problem_type)
        : m_evaluator(evaluator), m_options(opt_options),
          m_initial_condition(num_elements(evaluator.model_info())), m_problem_type(problem_type)
    {
        m_initial_condition.fill(1);
    }

    bool get_nlp_info(Index &, Index &, Index &, Index &, IndexStyleEnum &);

    bool get_bounds_info(Index, double *, double *, Index, double *, double *);

    bool get_starting_point(Index, bool, double *, bool, double *, double *, Index, bool, double *);

    bool eval_f(Index, const double *, bool, double &);

    bool eval_grad_f(Index, const double *, bool, double *);

    bool eval_g(Index, const double *, bool, Index, double *);

    bool eval_jac_g(Index, const double *, bool, Index, Index, Index *, Index *, double *);

    void finalize_solution(
        Ipopt::SolverReturn, Index n, const double *x, const double *, const double *, Index m,
        const double *g, const double *, double obj_value, const Ipopt::IpoptData *,
        Ipopt::IpoptCalculatedQuantities *)
    {
        m_optimal_values = std::vector<double>(x, x + n);
        m_constraint_values = std::vector<double>(g, g + m);
        m_obj_value = obj_value;
    }

    const std::vector<double> &optimal_values() const { return m_optimal_values; }
    const std::vector<double> &constraint_values() const { return m_constraint_values; }
    double objective() const { return m_obj_value; }

  private:
    Evaluator &m_evaluator;
    OptimizationOptions m_options;
    Eigen::VectorXd m_initial_condition;
    OptimizationType m_problem_type;

    void maybe_update_parameter(const double *x, bool new_x)
    {
        if (new_x)
        {
            m_evaluator.set_parameter(x);
        }
    }

    std::vector<double> m_optimal_values, m_constraint_values;
    double m_obj_value;
};

#endif // OPTIMIZATION_PROBLEM_HPP
