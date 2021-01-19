#ifndef OPTIMIZATION_PROBLEM_HPP
#define OPTIMIZATION_PROBLEM_HPP

#include "nlopt.hpp"

#include "evaluator.hpp"
#include "options.hpp"
class OptimizationProblem
{
  public:
    OptimizationProblem(
        Evaluator &evaluator, const OptimizationOptions &opt_options, OptimizationType problem_type)
        : m_evaluator(evaluator), m_options(opt_options),
          m_initial_condition(num_elements(evaluator.model_info())), m_problem_type(problem_type),
          m_reassign_interval(m_options.reassign_regions_interval.value()), m_iter{0},
          m_stress_alpha(m_options.stress_alpha.value()),
          m_stress_normalization(m_evaluator.num_aggregates()),
          m_mean_change(m_options.mean_change_iters.value())
    {
        m_initial_condition.fill(1);
        m_stress_normalization.fill(1);
    }

    nlopt::opt get_optimizer();

    // bool get_nlp_info(unsigned &, unsigned &, unsigned &, unsigned &, unsignedStyleEnum &);

    bool get_bounds_info(unsigned, double *, double *, unsigned, double *, double *);

    bool get_starting_point(unsigned, bool, double *, bool, double *, double *, unsigned, bool, double *);

    bool eval_f(unsigned, const double *, bool, double &);

    bool eval_grad_f(unsigned, const double *, bool, double *);

    bool eval_g(unsigned, const double *, bool, unsigned, double *);

    bool eval_jac_g(unsigned, const double *, bool, unsigned, unsigned, unsigned *, unsigned *, double *);

/*
    void finalize_solution(
        Ipopt::SolverReturn, unsigned n, const double *x, const double *, const double *, unsigned m,
        const double *g, const double *, double obj_value, const Ipopt::IpoptData *,
        Ipopt::IpoptCalculatedQuantities *)
    {
        m_optimal_values = Eigen::Map<const Eigen::VectorXd>(x, n);
        m_constraint_values = Eigen::Map<const Eigen::VectorXd>(g, m);
        m_obj_value = obj_value;
    } */

    const Eigen::VectorXd &optimal_values() const { return m_optimal_values; }
    const Eigen::VectorXd &constraint_values() const { return m_constraint_values; }
    double objective() const { return m_obj_value; }

/*
    bool intermediate_callback(
        Ipopt::AlgorithmMode, unsigned, double, double, double, double, double, double, double, double, unsigned,
        const Ipopt::IpoptData *, Ipopt::IpoptCalculatedQuantities *); */

  private:
    Evaluator &m_evaluator;
    OptimizationOptions m_options;
    Eigen::VectorXd m_initial_condition;
    OptimizationType m_problem_type;

    // How many iterations between reassigning aggregation regions.
    int m_reassign_interval;

    // Current optimization iteration.
    std::size_t m_iter;

    // Normalization for the stress criteria.
    double m_stress_alpha; // Control the rate at which the stress normalization is adjusted.
    Eigen::VectorXd m_stress_normalization;

    // Convergence history for early stopping (when mean change is small enough)
    Eigen::VectorXd m_mean_change;
    Eigen::VectorXd m_last_parameter;

    void maybe_update_parameter(const double *x, bool new_x)
    {
        m_last_parameter = Eigen::Map<const Eigen::VectorXd>(x, num_elements(m_evaluator.model_info()));
        if (new_x)
        {
            m_evaluator.set_parameter(x);
            update_stress_normalization();
            maybe_update_region_definitions();
        }
    }

    Eigen::VectorXd m_optimal_values, m_constraint_values;
    double m_obj_value;

    void update_mean_changes();
    void update_stress_normalization();
    void maybe_update_region_definitions();

    static double nlopt_f(unsigned, const double *, double *, void *);
    static void nlopt_g(unsigned, double *, unsigned, const double *, double *, void *);
};

#endif // OPTIMIZATION_PROBLEM_HPP
